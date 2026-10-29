#include "branchandprice/highs/root_lp.hpp"

#include "branchandprice/pricer/lagrangian.hpp"
#include "branchandprice/master/root_master.hpp"
#include "branchandprice/master/set_packing_heuristic.hpp"
#include "util/partition_ops.hpp"
#include "util/constants.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "Highs.h"
#include "util/log.hpp"

namespace maffe {
namespace {

using Clock = std::chrono::steady_clock;

class RootLpTimeout : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct LeafSetHash {
    [[nodiscard]] std::size_t operator()(const std::vector<int>& leaves) const {
        std::size_t seed = leaves.size();
        for (const int leaf : leaves)
            seed = seed * 1315423911u + static_cast<std::size_t>(leaf + 1);
        return seed;
    }
};

[[nodiscard]] std::optional<Clock::time_point> deadline_after(std::optional<double> time_limit_seconds) {
    if (!time_limit_seconds.has_value())
        return std::nullopt;
    if (*time_limit_seconds < 0.0)
        throw std::invalid_argument("root LP time limit must be nonnegative");
    return Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(*time_limit_seconds));
}

void check_deadline(
    const std::optional<Clock::time_point> deadline,
    const char* const context
) {
    if (deadline.has_value() && Clock::now() >= *deadline)
        throw RootLpTimeout(std::format("{} timed out", context));
}

void check(const HighsStatus status) {
    if (status != HighsStatus::kOk)
        throw std::runtime_error("HiGHS call failed");
}

[[nodiscard]] double remaining_seconds(
    const std::optional<Clock::time_point> deadline,
    const char* const context
) {
    check_deadline(deadline, context);
    return std::chrono::duration<double>(*deadline - Clock::now()).count();
}

void apply_highs_time_limit(
    Highs& highs,
    const std::optional<Clock::time_point> deadline,
    const char* const context
) {
    if (!deadline.has_value())
        return;
    check(highs.setOptionValue("time_limit", std::max(
        constants::root_lp_min_remaining_seconds,
        remaining_seconds(deadline, context)
    )));
}

void add_column(
    Highs& highs,
    const RootMasterColumn& column
) {
    std::vector<HighsInt> index;
    std::vector<double> value;
    index.reserve(column.row_indices.size());
    value.reserve(column.row_indices.size());
    for (const int row : column.row_indices) {
        index.push_back(row);
        value.push_back(1.0);
    }

    check(highs.addCol(
        column.objective,
        0.0,
        highs.getInfinity(),
        static_cast<HighsInt>(index.size()),
        index.data(),
        value.data()
    ));
}

[[nodiscard]] BasisStatus column_basis_status(const HighsBasisStatus status, const double value) {
    switch (status) {
    case HighsBasisStatus::kLower:
        return BasisStatus::LOWER;
    case HighsBasisStatus::kBasic:
        return BasisStatus::BASIC;
    case HighsBasisStatus::kUpper:
        throw std::runtime_error("unexpected HiGHS upper-bounded column status");
    case HighsBasisStatus::kNonbasic:
        if (std::abs(value) <= constants::root_lp_basis_tol)
            return BasisStatus::LOWER;
        break;
    case HighsBasisStatus::kZero:
        break;
    }
    throw std::runtime_error("unsupported HiGHS column basis status");
}

[[nodiscard]] BasisStatus row_basis_status(const HighsBasisStatus status, const double row_value) {
    switch (status) {
    case HighsBasisStatus::kBasic:
        return BasisStatus::BASIC;
    case HighsBasisStatus::kUpper:
        return BasisStatus::UPPER;
    case HighsBasisStatus::kNonbasic:
        if (std::abs(row_value - 1.0) <= constants::root_lp_basis_tol)
            return BasisStatus::UPPER;
        break;
    case HighsBasisStatus::kLower:
    case HighsBasisStatus::kZero:
        break;
    }
    throw std::runtime_error("unsupported HiGHS row basis status");
}

[[nodiscard]] Basis extract_basis(const RootMasterLayout& layout, const HighsBasis& highs_basis, const HighsSolution& solution) {
    if (!highs_basis.valid)
        throw std::runtime_error("optimal root LP without valid HiGHS basis");
    if (!solution.value_valid)
        throw std::runtime_error("optimal root LP without valid primal values");
    if (static_cast<int>(highs_basis.col_status.size()) != static_cast<int>(solution.col_value.size()))
        throw std::runtime_error("HiGHS column basis size mismatch");
    if (static_cast<int>(highs_basis.row_status.size()) != static_cast<int>(solution.row_value.size()))
        throw std::runtime_error("HiGHS row basis size mismatch");
    if (static_cast<int>(highs_basis.row_status.size()) != layout.row_count)
        throw std::runtime_error("HiGHS row basis does not match root master layout");

    Basis basis;
    basis.column_status.reserve(highs_basis.col_status.size());
    for (int i = 0; i < static_cast<int>(highs_basis.col_status.size()); ++i)
        basis.column_status.push_back(column_basis_status(highs_basis.col_status[i], solution.col_value[i]));

    basis.row_status.reserve(highs_basis.row_status.size());
    for (int i = 0; i < static_cast<int>(highs_basis.row_status.size()); ++i)
        basis.row_status.push_back(row_basis_status(highs_basis.row_status[i], solution.row_value[i]));
    return basis;
}

void add_rows(Highs& highs, const RootMasterLayout& layout) {
    for (int row = 0; row < layout.row_count; ++row)
        check(highs.addRow(-highs.getInfinity(), 1.0, 0, nullptr, nullptr));
}

[[nodiscard]] std::vector<double> solve_lp(
    Highs& highs,
    const RootMasterLayout& layout,
    const AnnotatedInstance& instance,
    RootLpResult& result,
    const int objective_offset,
    const std::optional<Clock::time_point> deadline
) {
    apply_highs_time_limit(highs, deadline, "root LP");
    check(highs.run());
    result.total_simplex_iterations += highs.getInfo().simplex_iteration_count;
    ++result.rounds;

    const auto status = highs.getModelStatus();
    if (status == HighsModelStatus::kTimeLimit)
        throw RootLpTimeout("root LP timed out");
    if (status != HighsModelStatus::kOptimal && status != HighsModelStatus::kModelEmpty)
        throw std::runtime_error("HiGHS did not solve the root LP to optimality");

    result.objective =
        highs.getObjectiveValue() +
        static_cast<double>(instance.trees.front().leaves()) +
        static_cast<double>(objective_offset);

    const auto solution = highs.getSolution();
    if (status == HighsModelStatus::kModelEmpty) {
        result.vertex_duals = zero_vertex_duals(instance);
        result.warm_start.column_values.clear();
        result.warm_start.basis = Basis{
            .column_status = {},
            .row_status = std::vector<BasisStatus>(layout.row_count, BasisStatus::BASIC),
        };
        return std::vector<double>(layout.row_count, 0.0);
    } else if (!solution.dual_valid) {
        throw std::runtime_error("optimal root LP without valid row duals");
    }

    std::vector<double> flat_row_duals(layout.row_count, 0.0);
    for (int i = 0; i < layout.row_count; ++i)
        flat_row_duals[i] = -solution.row_dual[i];
    result.vertex_duals = zero_vertex_duals(instance);
    for (int tree = 0; tree < static_cast<int>(instance.trees.size()); ++tree) {
        for (int u = instance.trees[tree].leaves(); u < instance.trees[tree].vertices(); ++u)
            result.vertex_duals[tree][u] = flat_row_duals[layout.row_of_vertex[tree][u]];
    }
    result.warm_start.column_values = solution.col_value;
    result.warm_start.basis = extract_basis(layout, highs.getBasis(), solution);
    return flat_row_duals;
}

[[nodiscard]] std::optional<RootMasterColumn> try_build_root_master_column(
    const AnnotatedInstance& instance,
    const RootMasterLayout& layout,
    std::span<const int> leaves
) {
    try {
        return build_root_master_column(instance, layout, leaves);
    } catch (const DisconnectedColumn&) {
        return std::nullopt;
    }
}

[[nodiscard]] Result greedy_set_packing_solution(
    const AnnotatedInstance& instance,
    const RootMasterLayout& layout,
    std::span<const RootMasterColumn> columns,
    std::span<const double> column_values
) {
    const int leaves = instance.trees.front().leaves();
    std::vector<RootSetPackingColumnView> column_views;
    column_views.reserve(columns.size());
    for (int i = 0; i < static_cast<int>(columns.size()); ++i) {
        const auto& column = columns[i];
        if (column.leaves.size() < 2)
            continue;
        const double raw_value = i < static_cast<int>(column_values.size()) ? column_values[i] : 0.0;
        column_views.push_back(RootSetPackingColumnView{
            .column_id = i,
            .leaves = column.leaves,
            .row_indices = column.row_indices,
            .forced_rows = {},
            .lp_value = raw_value,
            .tie_seed = static_cast<std::uint64_t>(i),
        });
    }
    const RootSetPackingSolution packing = solve_root_set_packing_heuristic(
        leaves,
        layout.row_count,
        0,
        static_cast<int>(columns.size()),
        column_views);

    std::vector<char> covered(leaves, false);
    Result result;
    result.feasible = true;
    result.partition.reserve(leaves);
    for (const int column_id : packing.columns) {
        const auto& column = columns[column_id];
        for (const int leaf : column.leaves) {
            if (covered[leaf])
                throw std::runtime_error("set packing heuristic returned overlapping leaves");
        }
        for (const int leaf : column.leaves)
            covered[leaf] = true;
        result.partition.push_back(column.leaves);
    }

    for (int leaf = 0; leaf < leaves; ++leaf) {
        if (!covered[leaf])
            result.partition.push_back({leaf});
    }
    detail::sort_partition_blocks(result.partition);
    return result;
}

void update_heuristic_solution(
    const AnnotatedInstance& instance,
    const RootMasterLayout& layout,
    std::span<const RootMasterColumn> columns,
    RootLpResult& result
) {
    Result candidate = greedy_set_packing_solution(instance, layout, columns, result.warm_start.column_values);
    if (!candidate.feasible)
        return;
    if (!result.heuristic_solution.has_value() ||
        candidate.partition.size() < result.heuristic_solution->partition.size()) {
        result.heuristic_solution = std::move(candidate);
    }
}

} // namespace

RootLpResult solve_root_lp_with_highs(
    const AnnotatedInstance& instance,
    const LogLevel log_level,
    const std::span<const std::vector<int>> seed_columns,
    const int objective_offset,
    const std::optional<double> time_limit_seconds,
    const bool allow_abort_with_incumbent
) {
    if (instance.trees.size() < 2)
        throw std::invalid_argument("root LP requires at least two trees");

    Highs highs;
    check(highs.setOptionValue("output_flag", false));
    check(highs.setOptionValue("parallel", "off"));
    check(highs.setOptionValue("presolve", "off"));
    check(highs.setOptionValue("threads", 1));
    check(highs.changeObjectiveSense(ObjSense::kMinimize));

    RootLpResult result;
    const auto deadline = deadline_after(time_limit_seconds);
    result.layout = build_root_master_layout(instance);
    add_rows(highs, result.layout);
    Lagrangian lagrangian(instance);
    std::vector<std::vector<EdgeState>> edge_states(instance.trees.size());
    for (int i = 0; i < static_cast<int>(instance.trees.size()); ++i)
        edge_states[i] = instance.trees[i].edge_state;
    std::unordered_set<std::vector<int>, LeafSetHash> inserted_columns;
    std::vector<RootMasterColumn> columns;

    for (const auto& leaves : seed_columns) {
        if (leaves.size() < 2)
            continue;
        auto column = try_build_root_master_column(instance, result.layout, leaves);
        if (!column.has_value() || !inserted_columns.emplace(column->leaves).second)
            continue;
        add_column(highs, *column);
        columns.push_back(*column);
        result.warm_start.columns.push_back(column->leaves);
    }
    if (allow_abort_with_incumbent)
        update_heuristic_solution(instance, result.layout, columns, result);

    try {
        for (;;) {
            check_deadline(deadline, "root LP");
            const auto highs_start = std::chrono::steady_clock::now();
            const auto flat_row_duals = solve_lp(highs, result.layout, instance, result, objective_offset, deadline);
            const auto highs_end = std::chrono::steady_clock::now();
            result.highs_seconds += std::chrono::duration<double>(highs_end - highs_start).count();
            result.edge_duals = zero_vertex_duals(instance);
            if (allow_abort_with_incumbent)
                update_heuristic_solution(instance, result.layout, columns, result);

            const auto pricing_start = std::chrono::steady_clock::now();
            const auto lagrangian_result = lagrangian.solve(
                result.vertex_duals,
                result.edge_duals,
                edge_states
            );
            result.lagrangian_lower_bound =
                lagrangian_result.lower_bound + static_cast<double>(objective_offset);

            int added = 0;
            if (std::abs(result.lagrangian_lower_bound - result.objective) > constants::root_lp_bound_tol) {
                for (const auto& block : lagrangian_result.pricing_blocks()) {
                    if (block.size() < 2)
                        continue;
                    auto column = try_build_root_master_column(instance, result.layout, block);
                    if (!column.has_value() || inserted_columns.contains(column->leaves) ||
                        root_master_reduced_cost(*column, flat_row_duals) >= -constants::root_lp_reduced_cost_tol) {
                        continue;
                    }

                    inserted_columns.emplace(column->leaves);
                    add_column(highs, *column);
                    columns.push_back(*column);
                    result.warm_start.columns.push_back(column->leaves);
                    ++added;
                }
            }
            const auto pricing_end = std::chrono::steady_clock::now();
            result.pricing_seconds += std::chrono::duration<double>(pricing_end - pricing_start).count();
            if (allow_abort_with_incumbent && added > 0)
                update_heuristic_solution(instance, result.layout, columns, result);

            if (added == 0) {
                const double gap = std::abs(result.lagrangian_lower_bound - result.objective);
                if (gap > constants::root_lp_bound_tol) {
                    throw std::runtime_error(std::format(
                        "final root LP mismatch: objective={} lagrangian-lb={} gap={} round={} columns={}",
                        result.objective,
                        result.lagrangian_lower_bound,
                        gap,
                        result.rounds,
                        result.warm_start.columns.size()
                    ));
                }
                if (logging::enabled(log_level)) {
                    logging::line("rootlp: summary");
                    logging::line(std::format(
                        "rootlp:   objective={:.2f} lower-bound={:.2f} rounds={}",
                        result.objective,
                        result.lagrangian_lower_bound,
                        result.rounds
                    ));
                    logging::line(std::format(
                        "rootlp:   columns={}",
                        result.warm_start.columns.size()
                    ));
                    if (logging::enabled(log_level, LogLevel::VERBOSE)) {
                        const double total_seconds =
                            result.highs_seconds + result.pricing_seconds;
                        logging::line(std::format(
                            "rootlp:   time={:.3f}s highs={:.3f}s pricing={:.3f}s",
                            total_seconds,
                            result.highs_seconds,
                            result.pricing_seconds
                        ));
                    }
                }
                return result;
            }
        }
    } catch (const RootLpTimeout&) {
        if (allow_abort_with_incumbent) {
            if (!result.heuristic_solution.has_value())
                result.heuristic_solution = detail::singleton_result(instance.trees.front().leaves());
            result.interrupted = true;
            if (logging::enabled(log_level)) {
                logging::line(std::format(
                    "rootlp: aborted rounds={} columns={} heuristic-components={}",
                    result.rounds,
                    result.warm_start.columns.size(),
                    result.heuristic_solution->partition.size()
                ));
            }
            return result;
        }
        throw;
    }
}

} // namespace maffe
