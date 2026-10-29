#pragma once

#include "maffe/common.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace maffe::detail {

// Index a binary forest. A negative parent starts a component; callers decide
// whether cut edges are included before constructing the index.
class TreeIndex {
public:
    explicit TreeIndex(const Tree& tree) : TreeIndex([&] {
        auto parents = tree.parent;
        for (int u = 0; u < tree.vertices(); ++u) {
            if (tree.edge_state[u] == EdgeState::CUT)
                parents[u] = -1;
        }
        return parents;
    }()) {}

    explicit TreeIndex(std::vector<int> parents)
        : tin(parents.size()), tout(parents.size()), component(parents.size(), -1),
          parent_(std::move(parents)),
          up_(std::max(1, std::bit_width(static_cast<unsigned>(parent_.size()))),
              std::vector<int>(parent_.size())) {
        std::vector<std::array<int, 2>> children(parent_.size(), {-1, -1});
        for (int u = 0; u < static_cast<int>(parent_.size()); ++u) {
            if (parent_[u] < 0)
                continue;
            auto& [left, right] = children[parent_[u]];
            if (left < 0)
                left = u;
            else if (right < 0)
                right = u;
            else
                throw std::invalid_argument("tree is not binary");
        }

        int timer = 0;
        std::vector<std::pair<int, bool>> stack;
        for (int root = 0; root < static_cast<int>(parent_.size()); ++root) {
            if (parent_[root] >= 0)
                continue;
            stack.emplace_back(root, false);
            while (!stack.empty()) {
                const auto [u, exit] = stack.back();
                stack.pop_back();
                if (exit) {
                    tout[u] = timer;
                    continue;
                }
                component[u] = root;
                tin[u] = timer++;
                up_[0][u] = parent_[u] < 0 ? u : parent_[u];
                for (int level = 1; level < static_cast<int>(up_.size()); ++level)
                    up_[level][u] = up_[level - 1][up_[level - 1][u]];
                stack.emplace_back(u, true);
                const auto [left, right] = children[u];
                if (right >= 0)
                    stack.emplace_back(right, false);
                if (left >= 0)
                    stack.emplace_back(left, false);
            }
        }
    }

    [[nodiscard]] int parent(const int u) const { return parent_[u]; }

    [[nodiscard]] bool ancestor(const int a, const int b) const {
        return component[a] == component[b] && tin[a] <= tin[b] && tout[b] <= tout[a];
    }

    [[nodiscard]] int lca(int a, const int b) const {
        if (component[a] != component[b])
            return -1;
        if (ancestor(a, b))
            return a;
        if (ancestor(b, a))
            return b;
        for (int level = static_cast<int>(up_.size()) - 1; level >= 0; --level) {
            if (!ancestor(up_[level][a], b))
                a = up_[level][a];
        }
        return up_[0][a];
    }

    struct VirtualTree {
        std::vector<int> nodes;
        // Parent indices within nodes; the first node is the root.
        std::vector<int> parent;
    };

    [[nodiscard]] VirtualTree virtual_tree(std::span<const int> selected) const {
        VirtualTree tree{.nodes = {selected.begin(), selected.end()}, .parent = {}};
        if (selected.empty())
            return tree;
        const auto normalize = [&] {
            std::ranges::sort(tree.nodes, {}, [&](const int u) { return tin[u]; });
            tree.nodes.erase(std::unique(tree.nodes.begin(), tree.nodes.end()), tree.nodes.end());
        };
        normalize();
        const int leaves = static_cast<int>(tree.nodes.size());
        tree.nodes.reserve(2 * leaves);
        for (int i = 1; i < leaves; ++i) {
            const int join = lca(tree.nodes[i - 1], tree.nodes[i]);
            if (join < 0)
                throw std::invalid_argument("virtual tree nodes are disconnected by cuts");
            tree.nodes.push_back(join);
        }
        normalize();
        tree.parent.assign(tree.nodes.size(), -1);
        std::vector<int> stack{0};
        for (int i = 1; i < static_cast<int>(tree.nodes.size()); ++i) {
            while (!ancestor(tree.nodes[stack.back()], tree.nodes[i]))
                stack.pop_back();
            tree.parent[i] = stack.back();
            stack.push_back(i);
        }
        return tree;
    }

    std::vector<int> tin;
    std::vector<int> tout;
    std::vector<int> component;

private:
    std::vector<int> parent_;
    std::vector<std::vector<int>> up_;
};

} // namespace maffe::detail
