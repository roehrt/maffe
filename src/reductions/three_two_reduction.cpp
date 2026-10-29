#include "maffe/common.hpp"
#include "reductions/reductions.hpp"
#include "util/tree_ops.hpp"

#include <algorithm>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

namespace maffe {
namespace {

struct ThreeTwoCandidate {
    int cut_tree = -1;
    int a = -1;
    int b = -1;
    int cut_edge = -1;
    int subtree = 0;
};

[[nodiscard]] bool valid_tree_count(const AnnotatedInstance& instance) {
    return instance.trees.size() == 2 && instance.trees.front().leaves() >= 3;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] std::optional<int> lowest_common_ancestor(
    const Tree& tree,
    int a,
    int b
) {
    std::vector<char> seen(tree.vertices(), false);
    for (int node = a;;) {
        seen[node] = true;
        if (tree.parent[node] < 0 || tree.edge_state[node] == EdgeState::CUT)
            break;
        node = tree.parent[node];
    }
    for (int node = b;;) {
        if (seen[node])
            return node;
        if (tree.parent[node] < 0 || tree.edge_state[node] == EdgeState::CUT)
            break;
        node = tree.parent[node];
    }
    return std::nullopt;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

[[nodiscard]] std::optional<int> unique_hanging_subtree(
    const Tree& tree,
    const std::span<const std::array<int, 2>> children,
    const int a,
    const int b
) {
    const auto join = lowest_common_ancestor(tree, a, b);
    if (!join)
        return std::nullopt;

    int hanging = -1;
    const auto scan_branch = [&](int leaf) -> bool {
        int child = leaf;
        for (int parent = tree.parent[child];
             parent >= 0 && tree.edge_state[child] != EdgeState::CUT && parent != *join;
             parent = tree.parent[child]) {
            const int sibling = detail::other_child(children, parent, child);
            if (sibling >= 0) {
                if (hanging >= 0)
                    return false;
                hanging = sibling;
            }
            child = parent;
        }
        if (tree.parent[child] >= 0 && tree.edge_state[child] == EdgeState::CUT && child != *join)
            return false;
        return true;
    };

    if (!scan_branch(a) || !scan_branch(b) || hanging < 0)
        return std::nullopt;
    return hanging;
}

[[nodiscard]] std::optional<ThreeTwoCandidate> find_best_three_two_candidate(const AnnotatedInstance& instance) {
    if (!valid_tree_count(instance))
        return {};
    std::vector children = {
        detail::tree_children(instance.trees[0]),
        detail::tree_children(instance.trees[1]),
    };
    std::vector subtree = {
        detail::subtree_leaf_counts(instance.trees[0], children[0]),
        detail::subtree_leaf_counts(instance.trees[1], children[1]),
    };

    std::optional<ThreeTwoCandidate> best;
    for (int cherry_tree = 0; cherry_tree < 2; ++cherry_tree) {
        const int other_tree = cherry_tree ^ 1;
        for (int node = instance.trees[cherry_tree].leaves(); node < instance.trees[cherry_tree].vertices(); ++node) {
            const auto [left, right] = children[cherry_tree][node];
            if (left < 0 || right < 0)
                continue;
            if (left >= instance.trees[cherry_tree].leaves() || right >= instance.trees[cherry_tree].leaves())
                continue;

            const int a = std::min(left, right);
            const int b = std::max(left, right);
            const auto hanging = unique_hanging_subtree(instance.trees[other_tree], children[other_tree], a, b);
            if (!hanging)
                continue;

            const ThreeTwoCandidate candidate{
                .cut_tree = other_tree,
                .a = a,
                .b = b,
                .cut_edge = *hanging,
                .subtree = subtree[other_tree][*hanging],
            };
            if (!best || std::tie(best->subtree, best->cut_tree, best->cut_edge, best->a, best->b) <
                std::tie(candidate.subtree, candidate.cut_tree, candidate.cut_edge, candidate.a, candidate.b))
                best = candidate;
        }
    }
    return best;
}

} // namespace

std::optional<Reduced> try_three_two_reduction(const AnnotatedInstance& instance) {
    if (!valid_tree_count(instance))
        return std::nullopt;

    AnnotatedInstance reduced = instance;
    int reductions = 0;
    while (true) {
        const auto candidate = find_best_three_two_candidate(reduced);
        if (!candidate)
            break;
        reduced.trees[candidate->cut_tree].edge_state[candidate->cut_edge] = EdgeState::CUT;
        ++reductions;
    }
    if (reductions == 0)
        return std::nullopt;

    return Reduced{
        .instance = std::move(reduced),
        .lift = [](Result result) { return result; },
        .reduction_count = reductions,
    };
}

} // namespace maffe
