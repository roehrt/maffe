#pragma once

#include "maffe/common.hpp"
#include "util/tree_index.hpp"

#include <span>
#include <stdexcept>
#include <vector>

namespace maffe {

struct DisconnectedColumn : std::invalid_argument {
    DisconnectedColumn() : std::invalid_argument("column leaf set is disconnected by cuts") {}
};

struct RootMasterLayout {
    int leaf_count = 0;
    int row_count = 0;
    std::vector<std::vector<int>> row_of_vertex;
    std::vector<detail::TreeIndex> tree_index;
};

struct RootMasterColumn {
    std::vector<int> leaves;
    std::vector<int> row_indices;
    std::vector<std::vector<int>> used_vertices;
    std::vector<std::vector<int>> used_edges;
    double objective = 0.0;
};

RootMasterLayout build_root_master_layout(const AnnotatedInstance& instance);
RootMasterColumn build_root_master_column(
    const AnnotatedInstance& instance,
    const RootMasterLayout& layout,
    std::span<const int> leaves
);
double root_master_reduced_cost(
    const RootMasterColumn& column,
    std::span<const double> flat_row_duals
);
std::vector<std::vector<double>> zero_vertex_duals(const AnnotatedInstance& instance);
bool root_master_column_uses_tree_edge(const RootMasterColumn& column, int tree, int edge);

} // namespace maffe
