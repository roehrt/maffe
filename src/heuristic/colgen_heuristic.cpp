#include "heuristic/colgen_heuristic.hpp"

#include "branchandprice/master/set_packing_heuristic.hpp"
#include "branchandprice/master/set_packing_scip.hpp"
#include "branchandprice/pricer/lagrangian.hpp"
#include "config.hpp"
#include "heuristic/node_coverage.hpp"
#include "util/constants.hpp"
#include "util/log.hpp"
#include "util/partition_ops.hpp"

#if MAFFE_HAVE_CONICBUNDLE
#include "CBSolver.hxx"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace maffe::heuristic {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double remaining_seconds(const Clock::time_point deadline) {
    return std::max(0.0, std::chrono::duration<double>(deadline - Clock::now()).count());
}

[[nodiscard]] std::string seconds_label(const double seconds) {
    return std::format("{:.1f}s", std::max(0.0, seconds));
}

[[nodiscard]] double phase_seconds(const double total_seconds, const double fraction) {
    return std::max(0.0, total_seconds * fraction);
}

[[nodiscard]] double bounded_phase_seconds(const Clock::time_point deadline, const double planned_seconds) {
    return std::min(std::max(0.0, planned_seconds), remaining_seconds(deadline));
}

[[nodiscard]] Result initial_or_singleton(const AnnotatedInstance& instance, const Result* initial_solution) {
    if (initial_solution != nullptr && initial_solution->feasible && !initial_solution->partition.empty()) {
        Result result = *initial_solution;
        detail::sort_partition_blocks(result.partition);
        return result;
    }
    return detail::singleton_result(instance.trees.front().leaves());
}

void add_column(std::map<std::vector<int>, int>& column_pool, std::vector<int> block, const int weight = 1) {
    if (block.size() < 2)
        return;
    std::ranges::sort(block);
    column_pool[std::move(block)] += weight;
}

void seed_column_pool(std::map<std::vector<int>, int>& column_pool, const Result& result) {
    constexpr int kInitialColumnWeight = 1'000'000;
    if (!result.feasible)
        return;
    for (auto block : result.partition)
        add_column(column_pool, std::move(block), kInitialColumnWeight);
}

[[nodiscard]] Result build_result_from_packing(
    const int leaves,
    const std::vector<std::vector<int>>& candidates,
    const RootSetPackingSolution& packing
) {
    Result result;
    result.feasible = true;
    std::vector<char> covered(leaves, 0);

    for (const int column : packing.columns) {
        if (column < 0 || column >= static_cast<int>(candidates.size()))
            throw std::runtime_error("set packing solver returned an invalid column index");
        std::vector<int> block = candidates[column];
        for (const int leaf : block) {
            if (leaf < 0 || leaf >= leaves)
                throw std::runtime_error("heuristic column contains an invalid leaf");
            if (covered[leaf] != 0)
                throw std::runtime_error("set packing solver returned overlapping columns");
            covered[leaf] = 1;
        }
        result.partition.push_back(std::move(block));
    }

    for (int leaf = 0; leaf < leaves; ++leaf) {
        if (covered[leaf] == 0)
            result.partition.push_back({leaf});
    }
    detail::sort_partition_blocks(result.partition);
    return result;
}

#if MAFFE_HAVE_CONICBUNDLE
struct ConicBundleStats {
    int oracle_evaluations = 0;
    int solve_calls = 0;
    double oracle_seconds = 0.0;
    double solve_seconds = 0.0;
};

class OracleEvaluationTimer {
public:
    explicit OracleEvaluationTimer(ConicBundleStats& stats) : stats_(stats) {}

    ~OracleEvaluationTimer() {
        stats_.oracle_seconds += std::chrono::duration<double>(Clock::now() - start_).count();
        ++stats_.oracle_evaluations;
    }

private:
    ConicBundleStats& stats_;
    Clock::time_point start_ = Clock::now();
};

class SubproblemOracle : public ConicBundle::FunctionOracle {
public:
    SubproblemOracle(
        const AnnotatedInstance& instance,
        std::map<std::vector<int>, int>& column_pool,
        Lagrangian& pricer,
        ConicBundleStats& stats
    )
        : instance_(instance),
          dual_count_(instance.trees[1].vertices() - instance.trees[1].leaves()),
          column_pool_(column_pool),
          pricer_(pricer),
          stats_(stats),
          coverage_(instance.trees[1]) {}

    int evaluate(
        const double* dual,
        double relprec,
        double& objective_value,
        std::vector<ConicBundle::Minorant*>& minorants,
        ConicBundle::PrimalExtender*&
    ) override {
        (void)relprec;
        OracleEvaluationTimer timer(stats_);

        std::vector<std::vector<double>> vertex_duals(2);
        vertex_duals[0].assign(instance_.trees[0].vertices(), 0.0);
        vertex_duals[1].assign(instance_.trees[1].vertices(), 0.0);

        const int leaves1 = instance_.trees[1].leaves();
        for (int i = 0; i < dual_count_; ++i)
            vertex_duals[1][leaves1 + i] = dual[i];

        std::vector<std::vector<double>> edge_duals(2);
        edge_duals[0].assign(instance_.trees[0].vertices(), 0.0);
        edge_duals[1].assign(instance_.trees[1].vertices(), 0.0);

        std::vector<std::vector<EdgeState>> edge_states(2);
        edge_states[0] = instance_.trees[0].edge_state;
        edge_states[1] = instance_.trees[1].edge_state;

        const auto priced = pricer_.solve_two_tree_orientation(0, 1, vertex_duals, edge_duals, edge_states);
        if (!std::isfinite(priced.lower_bound))
            return 1;

        for (auto block : priced.pricing_blocks())
            add_column(column_pool_, std::move(block));

        std::vector<int> cover(instance_.trees[1].vertices(), 0);
        coverage_.compute(priced.leaf_partition, cover);

        ConicBundle::DVector subgradient(dual_count_, 0.0);
        for (int i = 0; i < dual_count_; ++i)
            subgradient[i] = 1.0 - static_cast<double>(cover[leaves1 + i]);

        objective_value = -priced.lower_bound;
        minorants.push_back(new ConicBundle::Minorant(objective_value, subgradient, nullptr));
        return 0;
    }

private:
    const AnnotatedInstance& instance_;
    int dual_count_ = 0;
    std::map<std::vector<int>, int>& column_pool_;
    Lagrangian& pricer_;
    ConicBundleStats& stats_;
    NodeCoverage coverage_;
};
#endif

[[nodiscard]] std::vector<double> run_conicbundle_optimization(
    const AnnotatedInstance& instance,
    std::map<std::vector<int>, int>& column_pool,
    const std::string_view phase_name,
    const double time_limit_seconds,
    const LogLevel log_level
) {
    const int dual_count = instance.trees[1].vertices() - instance.trees[1].leaves();
    std::vector<double> lambda(dual_count, 0.0);
    if (time_limit_seconds <= 0.0)
        return lambda;

#if MAFFE_HAVE_CONICBUNDLE
    Lagrangian pricer(instance);
    std::ostream* output = logging::enabled(log_level, LogLevel::VERBOSE) ? &std::cerr : nullptr;
    ConicBundle::CBSolver solver(output, logging::enabled(log_level, LogLevel::VERBOSE) ? 1 : 0);
    solver.init_problem(dual_count);

    for (int i = 0; i < dual_count; ++i) {
        solver.set_lower_bound(i, 0.0);
        solver.set_upper_bound(i, constants::heuristic_colgen_lambda_bound);
    }

    ConicBundleStats stats;
    SubproblemOracle oracle(instance, column_pool, pricer, stats);
    solver.add_function(oracle);

    ConicBundle::BundleParameters params;
    params.set_max_model_size(constants::heuristic_colgen_conicbundle_max_model_size);
    params.set_max_bundle_size(constants::heuristic_colgen_conicbundle_max_bundle_size);
    solver.set_bundle_parameters(oracle, params);
    solver.set_term_relprec(1e-4);

    if (logging::enabled(log_level)) {
        logging::line(
            "heuristic-colgen: column-generation direction=", phase_name,
            " method=conic start time-limit=", seconds_label(time_limit_seconds),
            " model-size-limit=", constants::heuristic_colgen_conicbundle_max_model_size,
            " bundle-size-limit=", constants::heuristic_colgen_conicbundle_max_bundle_size,
            " iteration-limit=", constants::heuristic_colgen_conicbundle_max_iterations
        );
    }

    const auto start = Clock::now();
    int iterations = 0;
    while (!solver.termination_code() &&
           iterations < constants::heuristic_colgen_conicbundle_max_iterations &&
           std::chrono::duration<double>(Clock::now() - start).count() < time_limit_seconds) {
        const auto solve_start = Clock::now();
        const int oracle_evals_before = stats.oracle_evaluations;
        const std::size_t columns_before = column_pool.size();
        const int retval = solver.solve(0, true);
        const std::size_t columns_after = column_pool.size();
        const double step_seconds = std::chrono::duration<double>(Clock::now() - solve_start).count();
        stats.solve_seconds += step_seconds;
        ++stats.solve_calls;
        ++iterations;

        if (logging::enabled(log_level, LogLevel::VERBOSE)) {
            logging::line(
                "heuristic-colgen: column-generation direction=", phase_name,
                " method=conic step=", iterations,
                " status=", retval,
                " termination=", solver.termination_code(),
                " oracle-calls=", stats.oracle_evaluations - oracle_evals_before,
                " new-columns=", columns_after - columns_before,
                " pool=", columns_after,
                " time=", seconds_label(step_seconds)
            );
        }
        if (retval != 0 && solver.termination_code() == 0)
            break;
    }

    ConicBundle::DVector center;
    if (solver.get_center(center) == 0) {
        for (int i = 0; i < dual_count; ++i)
            lambda[i] = center[i];
    }

    if (logging::enabled(log_level)) {
        logging::line(
            "heuristic-colgen: column-generation direction=", phase_name,
            " method=conic done solver-calls=", stats.solve_calls,
            " iterations=", iterations,
            " oracle-calls=", stats.oracle_evaluations,
            " solver-time=", seconds_label(stats.solve_seconds),
            " oracle-time=", seconds_label(stats.oracle_seconds),
            " pool=", column_pool.size()
        );
    }
#else
    (void)instance;
    (void)column_pool;
    (void)phase_name;
    (void)log_level;
#endif

    return lambda;
}

void evaluate_perturbed_duals(
    const AnnotatedInstance& instance,
    std::map<std::vector<int>, int>& column_pool,
    const std::vector<double>& base_lambda,
    const std::string_view phase_name,
    const double time_limit_seconds,
    const LogLevel log_level
) {
    if (time_limit_seconds <= 0.0)
        return;

    Lagrangian pricer(instance);
    const int dual_count = instance.trees[1].vertices() - instance.trees[1].leaves();
    const int leaves1 = instance.trees[1].leaves();

    std::mt19937 rng(1337);
    std::uniform_real_distribution<double> multiplier(
        constants::heuristic_colgen_perturbation_multiplier_min,
        constants::heuristic_colgen_perturbation_multiplier_max
    );

    std::vector<std::vector<EdgeState>> edge_states(2);
    edge_states[0] = instance.trees[0].edge_state;
    edge_states[1] = instance.trees[1].edge_state;

    std::vector<std::vector<double>> edge_duals(2);
    edge_duals[0].assign(instance.trees[0].vertices(), 0.0);
    edge_duals[1].assign(instance.trees[1].vertices(), 0.0);

    const std::size_t initial_pool_size = column_pool.size();
    const auto start = Clock::now();
    int perturbations = 0;
    while (std::chrono::duration<double>(Clock::now() - start).count() < time_limit_seconds) {
        std::vector<double> perturbed = base_lambda;
        for (int i = 0; i < dual_count; ++i) {
            perturbed[i] = std::clamp(
                perturbed[i] * multiplier(rng),
                0.0,
                constants::heuristic_colgen_lambda_bound
            );
        }

        std::vector<std::vector<double>> vertex_duals(2);
        vertex_duals[0].assign(instance.trees[0].vertices(), 0.0);
        vertex_duals[1].assign(instance.trees[1].vertices(), 0.0);
        for (int i = 0; i < dual_count; ++i)
            vertex_duals[1][leaves1 + i] = perturbed[i];

        const auto priced = pricer.solve_two_tree_orientation(0, 1, vertex_duals, edge_duals, edge_states);
        if (!std::isfinite(priced.lower_bound))
            continue;

        for (auto block : priced.pricing_blocks())
            add_column(column_pool, std::move(block));
        ++perturbations;
    }

    if (logging::enabled(log_level)) {
        logging::line(
            "heuristic-colgen: column-generation direction=", phase_name,
            " method=perturb done time-limit=", seconds_label(time_limit_seconds),
            " trials=", perturbations,
            " new-columns=", column_pool.size() - initial_pool_size,
            " pool=", column_pool.size()
        );
    }
}

[[nodiscard]] std::vector<std::vector<int>> filter_columns(
    const std::map<std::vector<int>, int>& column_pool,
    const int max_columns,
    const LogLevel log_level
) {
    if (column_pool.empty() || max_columns <= 0)
        return {};

    struct ScoredColumn {
        std::vector<int> column;
        int count = 0;
        std::uint64_t tie_breaker = 0;
    };

    std::vector<ScoredColumn> scored;
    scored.reserve(column_pool.size());
    std::uint64_t tie = 0;
    for (const auto& [column, count] : column_pool)
        scored.push_back(ScoredColumn{.column = column, .count = count, .tie_breaker = tie++});

    std::ranges::sort(scored, [](const ScoredColumn& lhs, const ScoredColumn& rhs) {
        if (lhs.count != rhs.count)
            return lhs.count > rhs.count;
        if (lhs.column.size() != rhs.column.size())
            return lhs.column.size() < rhs.column.size();
        return lhs.tie_breaker < rhs.tie_breaker;
    });

    const int kept = std::min(max_columns, static_cast<int>(scored.size()));
    std::vector<std::vector<int>> result;
    result.reserve(kept);
    for (int i = 0; i < kept; ++i)
        result.push_back(std::move(scored[i].column));

    if (logging::enabled(log_level)) {
        logging::line(
            "heuristic-colgen: column-filter kept=", result.size(),
            " available=", column_pool.size(),
            " limit=", max_columns
        );
    }
    return result;
}

[[nodiscard]] std::vector<RootSetPackingColumnView> build_column_views(
    const AnnotatedInstance& instance,
    const std::vector<std::vector<int>>& candidates,
    std::vector<std::vector<int>>& column_rows
) {
    const int second_tree_offset = instance.trees[0].vertices() - instance.trees[0].leaves();
    const std::array coverage = {NodeCoverage(instance.trees[0]), NodeCoverage(instance.trees[1])};
    std::vector<int> cover;
    column_rows.clear();
    column_rows.resize(candidates.size());
    std::vector<RootSetPackingColumnView> views;
    views.reserve(candidates.size());
    for (int column = 0; column < static_cast<int>(candidates.size()); ++column) {
        const std::span single_component(&candidates[column], 1);
        for (int tree = 0; tree < 2; ++tree) {
            coverage[tree].compute(single_component, cover);
            for (int vertex = instance.trees[tree].leaves(); vertex < instance.trees[tree].vertices(); ++vertex) {
                if (cover[vertex] > 0)
                    column_rows[column].push_back((tree == 0 ? 0 : second_tree_offset) + vertex - instance.trees[tree].leaves());
            }
        }
        views.push_back(RootSetPackingColumnView{
            .column_id = column,
            .leaves = candidates[column],
            .row_indices = column_rows[column],
            .forced_rows = {},
            .lp_value = 0.0,
            .tie_seed = static_cast<std::uint64_t>(column),
        });
    }
    return views;
}

[[nodiscard]] std::vector<int> incumbent_column_ids(
    const std::vector<std::vector<int>>& candidates,
    const std::vector<std::vector<int>>& incumbent_columns
) {
    std::map<std::vector<int>, int> candidate_index;
    for (int column = 0; column < static_cast<int>(candidates.size()); ++column)
        candidate_index.emplace(candidates[column], column);

    std::vector<int> ids;
    ids.reserve(incumbent_columns.size());
    for (auto column : incumbent_columns) {
        std::ranges::sort(column);
        if (const auto it = candidate_index.find(column); it != candidate_index.end())
            ids.push_back(it->second);
    }
    return ids;
}

[[nodiscard]] std::vector<std::vector<int>> non_singleton_columns(const Result& result) {
    std::vector<std::vector<int>> columns;
    if (!result.feasible)
        return columns;
    for (auto block : result.partition) {
        if (block.size() < 2)
            continue;
        std::ranges::sort(block);
        columns.push_back(std::move(block));
    }
    detail::sort_partition_blocks(columns);
    return columns;
}

} // namespace

Result solve_colgen(
    const AnnotatedInstance& instance,
    const Clock::time_point deadline,
    const Result* const initial_solution,
    const int objective_offset,
    const LogLevel log_level
) {
    if (instance.trees.size() != 2)
        throw std::invalid_argument("heuristic column generation supports exactly two trees");
    if (instance.trees.front().leaves() < 1)
        throw std::invalid_argument("heuristic column generation requires at least one leaf");

    const int leaves = instance.trees.front().leaves();
    Result best_result = initial_or_singleton(instance, initial_solution);
    int best_components = static_cast<int>(best_result.partition.size());
    std::vector<std::vector<int>> best_packing_columns = non_singleton_columns(best_result);
    const auto objective = [objective_offset](const int components) {
        return components + objective_offset;
    };

    const double entry_seconds = remaining_seconds(deadline);
    if (entry_seconds <= 0.0)
        return best_result;

    std::map<std::vector<int>, int> column_pool;
    seed_column_pool(column_pool, best_result);

    const double conic_seconds = phase_seconds(entry_seconds, constants::heuristic_colgen_conicbundle_time_fraction);
    const double perturbation_seconds = phase_seconds(
        entry_seconds,
        constants::heuristic_colgen_perturbation_time_fraction
    );

    if (logging::enabled(log_level)) {
        logging::line(
            "heuristic-colgen: setup leaves=", leaves,
            " initial-components=", best_components,
            " initial-objective=", objective(best_components),
            " objective-offset=", objective_offset,
            " time-left=", seconds_label(entry_seconds),
            " conic-time-per-direction=", seconds_label(conic_seconds),
            " perturb-time-per-direction=", seconds_label(perturbation_seconds),
            " packing-column-limit=", constants::heuristic_colgen_max_set_packing_columns
        );
    }

    std::vector<double> lambda_a = run_conicbundle_optimization(
        instance,
        column_pool,
        "forward",
        bounded_phase_seconds(deadline, conic_seconds),
        log_level
    );
    evaluate_perturbed_duals(
        instance,
        column_pool,
        lambda_a,
        "forward",
        bounded_phase_seconds(deadline, perturbation_seconds),
        log_level
    );

    AnnotatedInstance swapped_instance{
        .trees = {instance.trees[1], instance.trees[0]},
    };
    std::vector<double> lambda_b = run_conicbundle_optimization(
        swapped_instance,
        column_pool,
        "reverse",
        bounded_phase_seconds(deadline, conic_seconds),
        log_level
    );
    evaluate_perturbed_duals(
        swapped_instance,
        column_pool,
        lambda_b,
        "reverse",
        bounded_phase_seconds(deadline, perturbation_seconds),
        log_level
    );

    if (logging::enabled(log_level))
        logging::line("heuristic-colgen: column-pool unique=", column_pool.size());
    if (column_pool.empty())
        return best_result;

    const int stage_count = static_cast<int>(constants::heuristic_colgen_set_packing_stage_columns.size());
    for (int stage = 0; stage < stage_count; ++stage) {
        const double remaining = remaining_seconds(deadline);
        if (remaining <= constants::heuristic_colgen_min_set_packing_seconds)
            break;

        const int max_columns = std::min(
            constants::heuristic_colgen_max_set_packing_columns,
            constants::heuristic_colgen_set_packing_stage_columns[stage]
        );
        std::vector<std::vector<int>> candidates = filter_columns(column_pool, max_columns, log_level);
        if (candidates.empty())
            break;

        const bool final_stage = stage + 1 == stage_count;
        const double stage_limit = constants::heuristic_colgen_set_packing_stage_seconds[stage];
        const double set_packing_seconds = final_stage || stage_limit <= 0.0
            ? remaining
            : std::min(remaining, stage_limit);
        const int root_cut_rounds = final_stage
            ? constants::heuristic_colgen_set_packing_max_cut_rounds_root
            : constants::heuristic_colgen_set_packing_early_max_cut_rounds_root;
        const bool root_only = !final_stage;

        std::vector<std::vector<int>> column_rows;
        std::vector<RootSetPackingColumnView> column_views = build_column_views(instance, candidates, column_rows);
        const int row_count = 2 * (leaves - 1);

        const RootSetPackingSolution greedy = solve_root_set_packing_heuristic(
            leaves,
            row_count,
            0,
            static_cast<int>(candidates.size()),
            column_views
        );
        const int greedy_components = leaves - greedy.saving;
        if (logging::enabled(log_level)) {
            logging::line(
                "heuristic-colgen: greedy-packing stage=", stage + 1,
                " selected-columns=", greedy.columns.size(),
                " saved=", greedy.saving,
                " components=", greedy_components,
                " objective=", objective(greedy_components),
                " improved=", greedy_components < best_components ? 1 : 0
            );
        }
        if (greedy_components < best_components) {
            best_components = greedy_components;
            best_result = build_result_from_packing(leaves, candidates, greedy);
            best_packing_columns = non_singleton_columns(best_result);
        }

        const std::vector<int> incumbent = incumbent_column_ids(candidates, best_packing_columns);

        if (logging::enabled(log_level)) {
            logging::line(
                "heuristic-colgen: packing stage=", stage + 1,
                "/", stage_count,
                " mode=", root_only ? "root-only" : "full",
                " candidates=", candidates.size(),
                " column-limit=", max_columns,
                " conflict-rows=", row_count,
                " warm-start-columns=", incumbent.size(),
                " root-cut-rounds=", root_cut_rounds,
                " time-limit=", seconds_label(set_packing_seconds)
            );
        }

        const RootSetPackingSolution packing = solve_root_set_packing_scip(
            leaves,
            row_count,
            0,
            static_cast<int>(candidates.size()),
            column_views,
            incumbent,
            objective_offset,
            set_packing_seconds,
            root_cut_rounds,
            root_only,
            log_level
        );
        const int candidate_components = leaves - packing.saving;
        if (logging::enabled(log_level)) {
            logging::line(
                "heuristic-colgen: packing stage=", stage + 1,
                " result selected-columns=", packing.columns.size(),
                " saved=", packing.saving,
                " components=", candidate_components,
                " objective=", objective(candidate_components),
                " improved=", candidate_components < best_components ? 1 : 0
            );
        }
        if (candidate_components < best_components) {
            best_components = candidate_components;
            best_result = build_result_from_packing(leaves, candidates, packing);
            best_packing_columns = non_singleton_columns(best_result);
        }
    }

    if (logging::enabled(log_level)) {
        logging::line(
            "heuristic-colgen: done best-components=", best_components,
            " best-objective=", objective(best_components),
            " selected-non-singletons=", best_packing_columns.size(),
            " time-left=", seconds_label(remaining_seconds(deadline))
        );
    }
    return best_result;
}

} // namespace maffe::heuristic
