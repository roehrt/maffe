#pragma once

#include "branchandprice/master/set_packing_heuristic.hpp"

#include <format>
#include <stdexcept>

namespace maffe::detail {

inline void validate_packing_column(
    const RootSetPackingColumnView& column,
    const int leaf_count,
    const int row_count,
    const int forced_row_count,
    const int column_count
) {
    if (column.column_id < 0 || column.column_id >= column_count)
        throw std::invalid_argument("set packing column id out of range");
    const auto check_indices = [](std::span<const int> indices, int limit, const char* kind) {
        int previous = -1;
        for (const int index : indices) {
            if (index < 0 || index >= limit)
                throw std::invalid_argument(std::format("set packing column {} out of range", kind));
            if (index <= previous)
                throw std::invalid_argument(std::format("set packing column {} indices must be sorted and unique", kind));
            previous = index;
        }
    };
    check_indices(column.leaves, leaf_count, "leaf");
    check_indices(column.row_indices, row_count, "row");
    check_indices(column.forced_rows, forced_row_count, "forced row");
}

} // namespace maffe::detail
