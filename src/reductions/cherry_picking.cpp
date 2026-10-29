#include "reductions/reductions.hpp"
#include "util/partition_ops.hpp"
#include "util/tree_ops.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "maffe/common.hpp"

namespace maffe {
namespace {

using Pair = std::array<int, 2>;

[[nodiscard]] std::vector<int> forest_roots(
    const Tree& tree,
    const std::vector<std::array<int, 2>>& include_cut_children
) {
    std::vector<int> roots = {tree.root()};
    const auto collect = [&](this auto&& self, const int node) -> void {
        if (node < tree.leaves())
            return;
        const auto [left, right] = include_cut_children[node];
        for (const int child : {left, right}) {
            if (child < 0)
                continue;
            if (tree.edge_state[child] == EdgeState::CUT)
                roots.push_back(child);
            self(child);
        }
    };
    collect(tree.root());
    return roots;
}

Tree reduce_tree(
    const Tree& tree,
    const std::vector<std::array<int, 2>>& children,
    const std::vector<int>& rep,
    const std::vector<int>& leaf_of_rep,
    const int leaves
) {
    Tree reduced{
        .parent = std::vector(leaves, -2),
        .edge_state = std::vector(leaves, EdgeState::UNKNOWN),
    };
    const auto merge_state = [](const EdgeState a, const EdgeState b) {
        if (a == EdgeState::UNKNOWN)
            return b;
        if (b == EdgeState::UNKNOWN || a == b)
            return a;
        throw std::runtime_error("cherry reduction encountered conflicting edge states on a compressed path");
    };

    const auto top = [&](const int node) {
        const int id = rep[node];
        if (id < 0 || leaf_of_rep[id] < 0)
            return false;

        const int parent = tree.parent[node];
        return parent < 0 || tree.edge_state[node] == EdgeState::CUT || rep[parent] < 0 || leaf_of_rep[rep[parent]] < 0;
    };

    int next = leaves;
    const auto build = [&](this auto&& self, const int node) -> std::pair<int, EdgeState> {
        if (top(node)) {
            const int leaf = leaf_of_rep[rep[node]];
            return {leaf, tree.edge_state[node]};
        }

        auto [left, right] = children[node];
        const auto [a, a_state] = left < 0 ? std::pair{-1, EdgeState::UNKNOWN} : self(left);
        const auto [b, b_state] = right < 0 ? std::pair{-1, EdgeState::UNKNOWN} : self(right);
        if (a < 0 || b < 0) {
            const int child = a < 0 ? b : a;
            const EdgeState child_state = a < 0 ? b_state : a_state;
            return {
                child,
                child < 0 ? EdgeState::UNKNOWN : merge_state(child_state, tree.edge_state[node]),
            };
        }

        const int current = next++;
        reduced.parent.resize(next, -2);
        reduced.edge_state.resize(next, EdgeState::UNKNOWN);
        reduced.parent[a] = current;
        reduced.parent[b] = current;
        reduced.edge_state[a] = a_state;
        reduced.edge_state[b] = b_state;
        return {current, tree.edge_state[node]};
    };

    const auto roots = forest_roots(
        tree,
        detail::tree_children(tree, detail::CutHandling::INCLUDE_CUT)
    );
    auto [root, root_state] = build(roots.front());
    static_cast<void>(root_state);
    for (int i = 1; i < static_cast<int>(roots.size()); ++i) {
        auto [component, component_state] = build(roots[i]);
        static_cast<void>(component_state);
        const int current = next++;
        reduced.parent.resize(next, -2);
        reduced.edge_state.resize(next, EdgeState::UNKNOWN);
        reduced.parent[root] = current;
        reduced.edge_state[root] = EdgeState::UNKNOWN;
        reduced.parent[component] = current;
        reduced.edge_state[component] = EdgeState::CUT;
        root = current;
    }
    reduced.parent[root] = -1;
    reduced.edge_state[root] = EdgeState::UNKNOWN;
    return reduced;
}

} // namespace

std::optional<Reduced> try_cherry_picking(const AnnotatedInstance& instance) {
    const int tree_count = static_cast<int>(instance.trees.size());
    const int leaves = instance.trees.front().leaves();
    if (tree_count < 2 || leaves < 2)
        return std::nullopt;

    std::vector<std::vector<std::array<int, 2>>> children(tree_count);
    std::vector<std::vector<int>> rep(tree_count);
    std::vector<std::vector<int>> roots(tree_count);
    std::vector<Pair> parts(leaves, {-1, -1});
    std::map<Pair, int> ids;

    for (int i = 0; i < tree_count; ++i) {
        const auto& tree = instance.trees[i];
        children[i] = detail::tree_children(tree);
        roots[i] = forest_roots(
            tree,
            detail::tree_children(tree, detail::CutHandling::INCLUDE_CUT)
        );
        rep[i].resize(tree.vertices());
        for (int leaf = 0; leaf < leaves; ++leaf)
            rep[i][leaf] = leaf;

        for (int node = leaves; node < tree.vertices(); ++node) {
            auto [left, right] = children[i][node];
            const int a = left < 0 ? -1 : rep[i][left];
            const int b = right < 0 ? -1 : rep[i][right];
            if (a < 0 || b < 0) {
                rep[i][node] = a < 0 ? b : a;
                continue;
            }

            Pair part{a, b};
            if (part[1] < part[0])
                std::swap(part[0], part[1]);

            const auto [it, inserted] = ids.emplace(part, static_cast<int>(parts.size()));
            if (inserted)
                parts.push_back(part);
            rep[i][node] = it->second;
        }
    }

    std::vector<int> count(parts.size());
    std::vector<int> seen(parts.size(), -1);
    for (int i = 0; i < tree_count; ++i) {
        for (const int id : rep[i]) {
            if (id < 0 || seen[id] == i)
                continue;
            seen[id] = i;
            ++count[id];
        }
    }

    std::vector<bool> common(parts.size());
    for (std::size_t id = 0; id < parts.size(); ++id)
        common[id] = count[id] == tree_count;

    std::vector<std::vector<int>> expanded(parts.size());
    std::vector<char> done(parts.size());
    const auto expand = [&](this auto&& self, const int id) -> const std::vector<int>& {
        if (done[id])
            return expanded[id];

        done[id] = true;
        auto& block = expanded[id];
        if (id < leaves) {
            block.push_back(id);
            return block;
        }

        const auto [left, right] = parts[id];
        const auto& a = self(left);
        const auto& b = self(right);
        block.resize(a.size() + b.size());
        std::merge(a.begin(), a.end(), b.begin(), b.end(), block.begin());
        return block;
    };

    const auto top = [&](const int i, const int node) {
        const int id = rep[i][node];
        if (id < 0 || !common[id])
            return false;

        const int parent = instance.trees[i].parent[node];
        return parent < 0 || instance.trees[i].edge_state[node] == EdgeState::CUT || rep[i][parent] < 0 || !common[rep[i][parent]];
    };

    std::vector<int> top_count(parts.size());
    std::vector<int> top_seen(parts.size(), -1);
    for (int i = 0; i < tree_count; ++i) {
        for (int node = 0; node < instance.trees[i].vertices(); ++node) {
            const int id = rep[i][node];
            if (id < 0 || !top(i, node) || top_seen[id] == i)
                continue;
            top_seen[id] = i;
            ++top_count[id];
        }
    }

    std::vector<int> chosen;
    std::vector<char> chosen_seen(parts.size());
    const auto collect = [&](this auto&& self, const int node) -> void {
        const int id = rep[0][node];
        if (id >= 0 && top(0, node) && top_count[id] == tree_count && !chosen_seen[id] && expand(id).size() == 2) {
            chosen_seen[id] = true;
            chosen.push_back(id);
            return;
        }
        if (node < instance.trees[0].leaves()) {
            if (!chosen_seen[node]) {
                chosen_seen[node] = true;
                chosen.push_back(node);
            }
            return;
        }
        const auto [left, right] = children[0][node];
        if (left >= 0)
            self(left);
        if (right >= 0)
            self(right);
    };
    for (const int root : roots[0])
        collect(root);
    if (chosen.size() == static_cast<std::size_t>(leaves))
        return std::nullopt;

    for (const int id : chosen)
        expand(id);

    std::vector<int> leaf_of_rep(parts.size(), -1);
    for (std::size_t leaf = 0; leaf < chosen.size(); ++leaf)
        leaf_of_rep[chosen[leaf]] = static_cast<int>(leaf);

    AnnotatedInstance reduced;
    reduced.trees.reserve(tree_count);
    for (int i = 0; i < tree_count; ++i)
        reduced.trees.push_back(reduce_tree(instance.trees[i], children[i], rep[i], leaf_of_rep, static_cast<int>(chosen.size())));
    const int removed_leaves = leaves - reduced.trees.front().leaves();

    return Reduced{
        .instance = std::move(reduced),
        .lift =
            [chosen = std::move(chosen), expanded = std::move(expanded)](Result result) {
                if (!result.feasible)
                    return result;

                Result lifted;
                lifted.feasible = true;
                lifted.partition.reserve(result.partition.size());
                for (const auto& block : result.partition) {
                    auto& out = lifted.partition.emplace_back();
                    for (const int leaf : block) {
                        const auto& part = expanded[chosen[leaf]];
                        out.insert(out.end(), part.begin(), part.end());
                    }
                }
                detail::sort_partition_blocks(lifted.partition);
                return lifted;
            },
        .reduction_count = removed_leaves,
    };
}

} // namespace maffe
