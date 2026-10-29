#include "maffe/common.hpp"
#include "reductions/reductions.hpp"
#include "util/tree_ops.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

namespace maffe {
namespace {

struct FiveThreeCandidate {
    int cut_tree = -1;
    int a = -1;
    int b = -1;
    int c = -1;
    std::array<int, 2> cut_edges{-1, -1};
    int subtree = 0;
};

struct Triple {
    int a = -1;
    int b = -1;
    int c = -1;
};

[[nodiscard]] bool valid_tree_count(const AnnotatedInstance& instance) {
    return instance.trees.size() == 2 && instance.trees.front().leaves() >= 5;
}

[[nodiscard]] bool is_leaf(const Tree& tree, const int node) {
    return node >= 0 && node < tree.leaves();
}

[[nodiscard]] std::optional<int> sibling_with_leaf(
    const Tree& tree,
    const std::span<const std::array<int, 2>> children,
    const int node,
    const int leaf
) {
    if (!is_leaf(tree, leaf) || node < tree.leaves())
        return std::nullopt;
    const auto [left, right] = children[node];
    if (left == leaf && right >= 0)
        return right;
    if (right == leaf && left >= 0)
        return left;
    return std::nullopt;
}

[[nodiscard]] std::optional<Triple> triple_at(
    const Tree& tree,
    const std::span<const std::array<int, 2>> children,
    const int node
) {
    if (node < tree.leaves())
        return {};

    auto [pair, c] = children[node];
    if (is_leaf(tree, pair))
        std::swap(pair, c);
    if (!is_leaf(tree, c) || pair < tree.leaves())
        return std::nullopt;
    const auto [a, b] = children[pair];
    if (!is_leaf(tree, a) || !is_leaf(tree, b))
        return std::nullopt;
    return Triple{.a = a, .b = b, .c = c};
}

[[nodiscard]] std::optional<std::array<int, 2>> paired_hanging_subtrees(
    const Tree& tree,
    const std::span<const std::array<int, 2>> children,
    const Triple triple
) {
    const int c_parent = tree.parent[triple.c];
    if (c_parent < 0 || tree.edge_state[triple.c] == EdgeState::CUT)
        return std::nullopt;

    const int pair = detail::other_child(children, c_parent, triple.c);
    if (pair < tree.leaves())
        return std::nullopt;

    const auto [left, right] = children[pair];
    const auto left_a = sibling_with_leaf(tree, children, left, triple.a);
    const auto right_b = sibling_with_leaf(tree, children, right, triple.b);
    if (left_a && right_b)
        return std::array{*left_a, *right_b};

    const auto left_b = sibling_with_leaf(tree, children, left, triple.b);
    const auto right_a = sibling_with_leaf(tree, children, right, triple.a);
    if (left_b && right_a)
        return std::array{*right_a, *left_b};

    return std::nullopt;
}

[[nodiscard]] std::optional<FiveThreeCandidate> find_best_five_three_candidate(const AnnotatedInstance& instance) {
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

    std::optional<FiveThreeCandidate> best;
    for (int triple_tree = 0; triple_tree < 2; ++triple_tree) {
        const int cut_tree = triple_tree ^ 1;
        for (int node = instance.trees[triple_tree].leaves(); node < instance.trees[triple_tree].vertices(); ++node) {
            const auto found = triple_at(instance.trees[triple_tree], children[triple_tree], node);
            if (!found)
                continue;
            for (const Triple triple : {*found, Triple{found->b, found->a, found->c}}) {
                auto cuts = paired_hanging_subtrees(instance.trees[cut_tree], children[cut_tree], triple);
                if (!cuts)
                    continue;
                std::ranges::sort(*cuts);
                const FiveThreeCandidate candidate{
                    .cut_tree = cut_tree,
                    .a = triple.a,
                    .b = triple.b,
                    .c = triple.c,
                    .cut_edges = *cuts,
                    .subtree = subtree[cut_tree][(*cuts)[0]] + subtree[cut_tree][(*cuts)[1]],
                };
                if (!best || std::tie(best->subtree, best->cut_tree, best->cut_edges, best->a, best->b, best->c) <
                    std::tie(candidate.subtree, candidate.cut_tree, candidate.cut_edges, candidate.a, candidate.b, candidate.c))
                    best = candidate;
            }
        }
    }
    return best;
}

} // namespace

std::optional<Reduced> try_five_three_reduction(const AnnotatedInstance& instance) {
    if (!valid_tree_count(instance))
        return std::nullopt;

    AnnotatedInstance reduced = instance;
    int reductions = 0;
    while (true) {
        const auto candidate = find_best_five_three_candidate(reduced);
        if (!candidate)
            break;
        for (const int edge : candidate->cut_edges) {
            if (reduced.trees[candidate->cut_tree].edge_state[edge] != EdgeState::CUT) {
                reduced.trees[candidate->cut_tree].edge_state[edge] = EdgeState::CUT;
                ++reductions;
            }
        }
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
