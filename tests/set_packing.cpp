#include "branchandprice/master/set_packing_heuristic.hpp"
#include "branchandprice/master/set_packing_validation.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition) {
    if (!condition)
        throw std::runtime_error("set packing regression");
}

void check_validation() {
    const std::vector valid{0, 2}, duplicate{1, 1}, unsorted{2, 0}, negative{-1}, outside{3};
    maffe::RootSetPackingColumnView column{.column_id = 0, .leaves = valid, .row_indices = valid, .forced_rows = valid};
    maffe::detail::validate_packing_column(column, 3, 3, 3, 1);
    const auto rejected = [](const auto& candidate) {
        try {
            maffe::detail::validate_packing_column(candidate, 3, 3, 3, 1);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    for (const auto& invalid : {duplicate, unsorted, negative, outside}) {
        auto candidate = column;
        candidate.leaves = invalid;
        require(rejected(candidate));
        candidate = column;
        candidate.row_indices = invalid;
        require(rejected(candidate));
        candidate = column;
        candidate.forced_rows = invalid;
        require(rejected(candidate));
    }
    for (int id : {-1, 1}) {
        auto candidate = column;
        candidate.column_id = id;
        require(rejected(candidate));
    }
}

void check_solution(const std::vector<maffe::RootSetPackingColumnView>& columns, int leaves, int rows, int forced) {
    const auto solution = maffe::solve_root_set_packing_heuristic(leaves, rows, forced, columns.size(), columns);
    std::vector<bool> leaf_used(leaves), row_used(rows), forced_used(forced);
    int saving = 0;
    for (int id : solution.columns) {
        require(id >= 0 && id < static_cast<int>(columns.size()));
        for (int leaf : columns[id].leaves) {
            require(!leaf_used[leaf]);
            leaf_used[leaf] = true;
        }
        for (int row : columns[id].row_indices) {
            require(!row_used[row]);
            row_used[row] = true;
        }
        for (int row : columns[id].forced_rows)
            forced_used[row] = true;
        saving += columns[id].leaves.size() - 1;
    }
    require(saving == solution.saving);
    require(std::ranges::all_of(forced_used, [](bool covered) { return covered; }));
}
} // namespace

int main() {
    check_validation();
    const std::vector a{0, 1}, b{2, 3}, overlap{1, 2}, row_a{0}, row_b{1};
    std::vector<maffe::RootSetPackingColumnView> fixture{
        {.column_id = 0, .leaves = a, .row_indices = row_a, .forced_rows = row_a},
        {.column_id = 1, .leaves = b, .row_indices = row_b, .forced_rows = row_b},
        {.column_id = 2, .leaves = overlap, .row_indices = row_a, .forced_rows = row_a},
    };
    check_solution(fixture, 4, 2, 2);
    require(maffe::solve_root_set_packing_heuristic(4, 2, 2, 3, fixture).saving == 2);

    std::mt19937 random(914);
    for (int trial = 0; trial < 80; ++trial) {
        const int count = trial == 79 ? 5100 : 20;
        std::vector<std::vector<int>> leaves(count), rows(count);
        std::vector<maffe::RootSetPackingColumnView> columns;
        for (int i = 0; i < count; ++i) {
            leaves[i] = {static_cast<int>(random() % 16), static_cast<int>(random() % 16)};
            if (leaves[i][0] == leaves[i][1])
                leaves[i][1] = (leaves[i][0] + 1) % 16;
            std::ranges::sort(leaves[i]);
            rows[i] = {static_cast<int>(random() % 8)};
            columns.push_back({.column_id = i, .leaves = leaves[i], .row_indices = rows[i], .forced_rows = {},
                .lp_value = static_cast<double>(random() % 100) / 100,
                .warm_start = i % 3 == 0, .tie_seed = static_cast<unsigned>(i)});
        }
        check_solution(columns, 16, 8, 0);
    }
}
