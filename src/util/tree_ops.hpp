#pragma once

#include "maffe/common.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

namespace maffe::detail {

enum class CutHandling : std::uint8_t {
    INCLUDE_CUT,
    EXCLUDE_CUT,
};

enum class BinaryValidation : std::uint8_t {
    NONE,
    REQUIRE_BINARY,
};

[[nodiscard]] inline std::vector<std::array<int, 2>> tree_children(
    const Tree& tree,
    const CutHandling cut_handling = CutHandling::EXCLUDE_CUT,
    const BinaryValidation validation = BinaryValidation::NONE
) {
    std::vector<std::array<int, 2>> children(tree.vertices(), {-1, -1});
    for (int u = 0; u < tree.vertices(); ++u) {
        const int parent = tree.parent[u];
        if (parent < 0)
            continue;
        if (cut_handling == CutHandling::EXCLUDE_CUT && tree.edge_state[u] == EdgeState::CUT)
            continue;

        auto& [left, right] = children[parent];
        if (left < 0)
            left = u;
        else if (right < 0 || validation != BinaryValidation::REQUIRE_BINARY)
            right = u;
        else
            throw std::invalid_argument("tree is not binary");
    }
    return children;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] inline int other_child(
    const std::span<const std::array<int, 2>> children,
    const int parent,
    const int child
) {
    const auto [left, right] = children[parent];
    return left ^ right ^ child;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

[[nodiscard]] inline std::vector<int> subtree_leaf_counts(
    const Tree& tree,
    const std::span<const std::array<int, 2>> children
) {
    std::vector<int> count(tree.vertices(), 0);
    for (int leaf = 0; leaf < tree.leaves(); ++leaf)
        count[leaf] = 1;
    for (int node = tree.leaves(); node < tree.vertices(); ++node) {
        const auto [left, right] = children[node];
        count[node] = (left >= 0 ? count[left] : 0) + (right >= 0 ? count[right] : 0);
    }
    return count;
}

[[nodiscard]] inline Tree restrict_tree_by_leaf_map(
    const Tree& tree,
    const std::vector<int>& leaf_map,
    const char* const empty_error = "restricted tree removed all leaves"
) {
    const auto leaves = std::ranges::count_if(leaf_map, [](const int value) { return value >= 0; });
    Tree reduced{
        .parent = std::vector<int>(leaves, -2),
        .edge_state = std::vector<EdgeState>(leaves, EdgeState::UNKNOWN),
    };
    const auto children = tree_children(tree, CutHandling::INCLUDE_CUT);
    int next = static_cast<int>(reduced.parent.size());
    const auto copy = [&](this auto&& self, const int node) -> int {
        if (node < tree.leaves())
            return leaf_map[node];

        const auto [left, right] = children[node];
        const int a = self(left);
        const int b = self(right);
        if (a < 0 || b < 0) {
            const int child = a < 0 ? b : a;
            if (child >= 0 && tree.edge_state[node] != EdgeState::UNKNOWN)
                reduced.edge_state[child] = tree.edge_state[node];
            return child;
        }

        const int current = next++;
        reduced.parent.resize(next, -2);
        reduced.edge_state.resize(next, EdgeState::UNKNOWN);
        reduced.parent[a] = current;
        reduced.parent[b] = current;
        return current;
    };

    const int root = copy(tree.root());
    if (root < 0)
        throw std::runtime_error(empty_error);
    reduced.parent[root] = -1;
    reduced.edge_state[root] = EdgeState::UNKNOWN;
    return reduced;
}

[[nodiscard]] inline AnnotatedInstance restrict_instance_by_leaf_map(
    const AnnotatedInstance& instance,
    const std::vector<int>& leaf_map,
    const char* const empty_error = "restricted tree removed all leaves"
) {
    AnnotatedInstance reduced;
    reduced.trees.reserve(instance.trees.size());
    for (const auto& tree : instance.trees)
        reduced.trees.push_back(restrict_tree_by_leaf_map(tree, leaf_map, empty_error));
    return reduced;
}

} // namespace maffe::detail
