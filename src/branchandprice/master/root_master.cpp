#include "branchandprice/master/root_master.hpp"
#include "maffe/common.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace maffe {
namespace {

[[nodiscard]] std::vector<int> checked_leaves(const int leaf_count, std::span<const int> leaves) {
    if (leaves.empty())
        throw std::invalid_argument("column must contain at least one leaf");

    std::vector<int> checked(leaves.begin(), leaves.end());
    std::ranges::sort(checked);
    int previous = -1;
    for (const int leaf : checked) {
        if (leaf < 0 || leaf >= leaf_count)
            throw std::invalid_argument("column leaf out of range");
        if (leaf == previous)
            throw std::invalid_argument("column leaf set must be duplicate-free");
        previous = leaf;
    }
    return checked;
}

struct UsedSupport {
    std::vector<int> vertices;
    std::vector<int> edges;
};

[[nodiscard]] UsedSupport used_support(
    const Tree& tree,
    const detail::TreeIndex& index,
    std::span<const int> leaves
) {
    if (leaves.size() < 2)
        return {};

    for (const int leaf : leaves) {
        if (index.component[leaf] != index.component[leaves.front()])
            throw DisconnectedColumn{};
    }
    const auto virtual_tree = index.virtual_tree(leaves);

    UsedSupport support;
    for (int i = 1; i < static_cast<int>(virtual_tree.nodes.size()); ++i) {
        const int child = virtual_tree.nodes[i];
        const int parent = virtual_tree.nodes[virtual_tree.parent[i]];
        for (int node = child; node != parent; node = tree.parent[node]) {
            if (node < 0 || tree.parent[node] < 0 || tree.edge_state[node] == EdgeState::CUT)
                throw DisconnectedColumn{};

            support.edges.push_back(node);
            const int ancestor = tree.parent[node];
            if (ancestor >= tree.leaves())
                support.vertices.push_back(ancestor);
        }
    }

    std::ranges::sort(support.edges);
    std::ranges::sort(support.vertices);
    support.vertices.erase(std::unique(support.vertices.begin(), support.vertices.end()), support.vertices.end());
    return support;
}

} // namespace

RootMasterLayout build_root_master_layout(const AnnotatedInstance& instance) {
    if (instance.trees.empty())
        throw std::invalid_argument("root master requires at least one tree");

    RootMasterLayout layout{
        .leaf_count = instance.trees.front().leaves(),
        .row_count = 0,
        .row_of_vertex = std::vector<std::vector<int>>(instance.trees.size()),
        .tree_index = {},
    };
    layout.tree_index.reserve(instance.trees.size());

    for (int i = 0; i < static_cast<int>(instance.trees.size()); ++i) {
        const auto& tree = instance.trees[i];
        if (tree.leaves() != layout.leaf_count)
            throw std::invalid_argument("all trees must have the same number of leaves");

        layout.tree_index.emplace_back(tree);
        auto& rows = layout.row_of_vertex[i];
        rows.resize(tree.vertices(), -1);
        for (int u = tree.leaves(); u < tree.vertices(); ++u)
            rows[u] = layout.row_count++;
    }
    return layout;
}

RootMasterColumn build_root_master_column(
    const AnnotatedInstance& instance,
    const RootMasterLayout& layout,
    std::span<const int> leaves
) {
    if (static_cast<int>(instance.trees.size()) != static_cast<int>(layout.row_of_vertex.size()) ||
        static_cast<int>(instance.trees.size()) != static_cast<int>(layout.tree_index.size())) {
        throw std::invalid_argument("root master layout does not match instance");
    }

    RootMasterColumn column{
        .leaves = checked_leaves(layout.leaf_count, leaves),
        .row_indices = {},
        .used_vertices = std::vector<std::vector<int>>(instance.trees.size()),
        .used_edges = std::vector<std::vector<int>>(instance.trees.size()),
        .objective = 0.0,
    };

    column.objective = 1.0 - static_cast<double>(column.leaves.size());
    if (column.leaves.size() < 2)
        return column;

    std::size_t nnz = 0;
    for (int i = 0; i < static_cast<int>(instance.trees.size()); ++i) {
        auto support = used_support(instance.trees[i], layout.tree_index[i], column.leaves);
        column.used_vertices[i] = std::move(support.vertices);
        column.used_edges[i] = std::move(support.edges);
        nnz += column.used_vertices[i].size();
    }

    column.row_indices.reserve(nnz);
    for (int i = 0; i < static_cast<int>(instance.trees.size()); ++i) {
        for (const int u : column.used_vertices[i])
            column.row_indices.push_back(layout.row_of_vertex[i][u]);
    }
    return column;
}

double root_master_reduced_cost(
    const RootMasterColumn& column,
    std::span<const double> flat_row_duals
) {
    double reduced_cost = column.objective;
    for (const int row : column.row_indices)
        reduced_cost += flat_row_duals[row];
    return reduced_cost;
}

std::vector<std::vector<double>> zero_vertex_duals(const AnnotatedInstance& instance) {
    std::vector<std::vector<double>> duals(instance.trees.size());
    for (int i = 0; i < static_cast<int>(instance.trees.size()); ++i)
        duals[i].assign(instance.trees[i].vertices(), 0.0);
    return duals;
}

bool root_master_column_uses_tree_edge(const RootMasterColumn& column, const int tree, const int edge) {
    if (tree < 0 || tree >= static_cast<int>(column.used_edges.size()))
        return false;
    return std::binary_search(column.used_edges[tree].begin(), column.used_edges[tree].end(), edge);
}

} // namespace maffe
