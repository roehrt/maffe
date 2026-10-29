#pragma once

#include "maffe/common.hpp"
#include "util/tree_ops.hpp"

#include <algorithm>
#include <format>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace maffe::detail {

struct BinaryTreeView {
    int leaves = 0;
    std::vector<int> parent;
    std::vector<std::array<int, 2>> child;
};

[[nodiscard]] inline BinaryTreeView build_binary_tree_view(
    const Tree& tree,
    const CutHandling cut_handling = CutHandling::EXCLUDE_CUT
) {
    return BinaryTreeView{
        .leaves = tree.leaves(),
        .parent = tree.parent,
        .child = tree_children(tree, cut_handling),
    };
}

[[nodiscard]] inline std::vector<BinaryTreeView> build_binary_tree_views(
    const AnnotatedInstance& instance,
    const CutHandling cut_handling = CutHandling::EXCLUDE_CUT
) {
    std::vector<BinaryTreeView> trees;
    trees.reserve(instance.trees.size());
    for (const auto& tree : instance.trees)
        trees.push_back(build_binary_tree_view(tree, cut_handling));
    return trees;
}

struct InducedSubtree {
    std::vector<int> count;
    int root;
};

[[nodiscard]] inline InducedSubtree induced_subtree(const BinaryTreeView& tree, const std::vector<int>& leaves) {
    std::vector<int> count(static_cast<int>(tree.parent.size()), 0);
    for (const int leaf : leaves)
        count[leaf] = 1;
    for (int u = tree.leaves; u < static_cast<int>(tree.parent.size()); ++u) {
        const auto [left, right] = tree.child[u];
        count[u] = (left >= 0 ? count[left] : 0) + (right >= 0 ? count[right] : 0);
    }

    int root = leaves.front();
    while (root >= 0 && count[root] < static_cast<int>(leaves.size()))
        root = tree.parent[root];
    return {std::move(count), root};
}

[[nodiscard]] inline std::string induced_signature(const BinaryTreeView& tree, const std::vector<int>& leaves) {
    if (leaves.empty())
        return {};
    const auto [count, root] = induced_subtree(tree, leaves);
    if (root < 0)
        return "!";

    const auto build = [&](this auto&& self, const int u) -> std::string {
        if (count[u] == 0)
            return {};
        if (u < tree.leaves)
            return std::format("{}", u);

        std::string left;
        std::string right;
        if (tree.child[u][0] >= 0)
            left = self(tree.child[u][0]);
        if (tree.child[u][1] >= 0)
            right = self(tree.child[u][1]);
        if (left.empty())
            return right;
        if (right.empty())
            return left;
        if (right < left)
            std::swap(left, right);
        return "(" + left + "," + right + ")";
    };
    return build(root);
}

[[nodiscard]] inline std::vector<int> used_vertices(const BinaryTreeView& tree, const std::vector<int>& leaves) {
    if (leaves.size() < 2)
        return {};

    const auto [count, root] = induced_subtree(tree, leaves);
    if (root < 0)
        return {-1};

    std::vector<int> vertices;
    const auto mark = [&](this auto&& self, const int u) -> void {
        if (u < tree.leaves)
            return;
        const auto [left, right] = tree.child[u];
        const bool in_left = left >= 0 && count[left] > 0;
        const bool in_right = right >= 0 && count[right] > 0;
        if (!in_left && !in_right)
            return;
        vertices.push_back(u);
        if (in_left)
            self(left);
        if (in_right)
            self(right);
    };
    mark(root);
    return vertices;
}

[[nodiscard]] inline bool valid_component_all_trees(
    const std::span<const BinaryTreeView> trees,
    const std::vector<int>& block
) {
    if (block.empty())
        return false;
    const std::string reference = induced_signature(trees.front(), block);
    for (int tree = 1; tree < static_cast<int>(trees.size()); ++tree) {
        if (induced_signature(trees[tree], block) != reference)
            return false;
    }
    return true;
}

[[nodiscard]] inline bool partition_feasible(
    const std::span<const BinaryTreeView> trees,
    const std::vector<std::vector<int>>& partition
) {
    for (const auto& tree : trees) {
        std::vector<int> used(static_cast<int>(tree.parent.size()), 0);
        for (const auto& block : partition) {
            for (const int u : used_vertices(tree, block)) {
                if (u < 0)
                    return false;
                if (++used[u] > 1)
                    return false;
            }
        }
    }
    for (const auto& block : partition) {
        if (!valid_component_all_trees(trees, block))
            return false;
    }
    return true;
}

} // namespace maffe::detail
