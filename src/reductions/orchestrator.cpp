#include "config.hpp"

#include "branchandprice/scip/branch_and_price.hpp"
#include "compact/compact_solver.hpp"
#include "heuristic/colgen_heuristic.hpp"
#include "maffe/common.hpp"
#include "reductions/reductions.hpp"
#include "reductions/detail/binary_tree_view.hpp"
#include "util/partition_ops.hpp"
#include "util/log.hpp"
#include "util/constants.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <format>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace maffe {

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::string_view kPresolveHeader =
    "presolve:   time | round | leaves | dualbound | cherry | chain | 3-2 | 5-3 | cluster | #sub | maxsub";

struct ReductionCounters {
    int* reductions = nullptr;
    int* subinstances = nullptr;
    int* largest_subinstance = nullptr;
    double* subinstance_seconds = nullptr;
};

struct PostCutCherryCounts {
    int three_two = 0;
    int five_three = 0;
    int cherry = 0;
};

struct PresolveRoundStats {
    int cherry = 0;
    int chain = 0;
    int cluster = 0;
    int subinstances = 0;
    int largest_subinstance = 0;
    std::optional<int> three_two;
    std::optional<int> five_three;

    void add_post_cut_counts(const PostCutCherryCounts& counts, const bool record_zero) {
        cherry += counts.cherry;
        if (record_zero || counts.three_two > 0)
            three_two = three_two.value_or(0) + counts.three_two;
        if (record_zero || counts.five_three > 0)
            five_three = five_three.value_or(0) + counts.five_three;
    }

    [[nodiscard]] bool has_progress() const {
        return cherry > 0 || chain > 0 || cluster > 0 ||
            (three_two.has_value() && *three_two > 0) ||
            (five_three.has_value() && *five_three > 0);
    }
};

struct PresolveTimings {
    double cherry = 0.0;
    double chain = 0.0;
    double three_two = 0.0;
    double five_three = 0.0;
    double cluster = 0.0;
    double cut_aware_cluster = 0.0;
    double subinstances = 0.0;
};

template <class F>
auto timed(double& seconds, F&& f) {
    const auto start = Clock::now();
    auto result = std::forward<F>(f)();
    seconds += std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

void enumerate_partitions(
    const std::vector<detail::BinaryTreeView>& trees,
    const int next_leaf,
    const int leaf_count,
    std::vector<std::vector<int>>& blocks,
    int& best,
    std::vector<std::vector<int>>& best_partition
) {
    if (static_cast<int>(blocks.size()) >= best)
        return;
    if (next_leaf == leaf_count) {
        if (!detail::partition_feasible(trees, blocks))
            return;
        best = static_cast<int>(blocks.size());
        best_partition = blocks;
        return;
    }

    const int block_count = static_cast<int>(blocks.size());
    for (int i = 0; i < block_count; ++i) {
        blocks[i].push_back(next_leaf);
        enumerate_partitions(trees, next_leaf + 1, leaf_count, blocks, best, best_partition);
        blocks[i].pop_back();
    }

    blocks.push_back({next_leaf});
    enumerate_partitions(trees, next_leaf + 1, leaf_count, blocks, best, best_partition);
    blocks.pop_back();
}

[[nodiscard]] Result solve_small_exact(const AnnotatedInstance& instance) {
    const auto trees = detail::build_binary_tree_views(instance);

    const int leaf_count = instance.trees.front().leaves();
    int best = leaf_count;
    std::vector<std::vector<int>> best_partition;
    best_partition.reserve(leaf_count);
    for (int leaf = 0; leaf < leaf_count; ++leaf)
        best_partition.push_back({leaf});

    std::vector<std::vector<int>> blocks;
    enumerate_partitions(trees, 0, leaf_count, blocks, best, best_partition);
    detail::sort_partition_blocks(best_partition);
    return Result{
        .partition = std::move(best_partition),
        .feasible = true,
    };
}

[[nodiscard]] bool has_edge_state(const AnnotatedInstance& instance, const EdgeState target) {
    for (const auto& tree : instance.trees) {
        for (const auto state : tree.edge_state) {
            if (state == target)
                return true;
        }
    }
    return false;
}

[[nodiscard]] bool has_common_leaf_count(const AnnotatedInstance& instance) {
    if (instance.trees.empty())
        return true;
    const int leaves = instance.trees.front().leaves();
    for (const auto& tree : instance.trees) {
        if (tree.leaves() != leaves)
            return false;
    }
    return true;
}

[[nodiscard]] std::string leaf_count_list(const AnnotatedInstance& instance) {
    std::string result;
    for (int i = 0; i < static_cast<int>(instance.trees.size()); ++i) {
        if (i > 0)
            result += ",";
        result += std::format("{}", instance.trees[i].leaves());
    }
    return result;
}

void require_common_leaf_count(const AnnotatedInstance& instance, const std::string_view source) {
    if (!has_common_leaf_count(instance)) {
        throw std::runtime_error(std::format(
            "{} produced mismatched leaf counts [{}]",
            source,
            leaf_count_list(instance)
        ));
    }
}

[[nodiscard]] bool has_non_unknown_edge_states(const AnnotatedInstance& instance) {
    return has_edge_state(instance, EdgeState::CUT) || has_edge_state(instance, EdgeState::FORCED);
}

[[nodiscard]] bool usable_result(const Result& result) {
    return result.feasible && !result.partition.empty();
}

[[nodiscard]] Result singleton_result(const AnnotatedInstance& instance) {
    return detail::singleton_result(instance.trees.empty() ? 0 : instance.trees.front().leaves());
}

[[nodiscard]] bool should_use_compact_lp(const AnnotatedInstance& instance) {
    return !has_non_unknown_edge_states(instance) &&
        instance.trees.size() >= constants::compact_subsolver_min_trees &&
        instance.trees.front().leaves() < constants::compact_subsolver_max_leaves;
}

[[nodiscard]] constexpr std::string_view compact_backend_name() {
#if MAFFE_HAVE_GUROBI
    return "gurobi";
#else
    return "scip";
#endif
}

[[nodiscard]] Result solve_compact_lp_or_throw(
    const AnnotatedInstance& instance,
    const LogLevel log_level,
    const std::optional<double> time_limit_seconds,
    const bool allow_abort_with_incumbent,
    const int objective_offset
) {
    Result result;
#if MAFFE_HAVE_GUROBI
    result = solve_with_compact_gurobi(
        instance,
        log_level,
        time_limit_seconds,
        allow_abort_with_incumbent,
        objective_offset
    );
#else
    result = solve_with_compact_root_lp_seeded(
        instance,
        log_level,
        time_limit_seconds,
        allow_abort_with_incumbent,
        objective_offset
    );
#endif
    if (usable_result(result))
        return result;
    throw std::runtime_error(std::format(
        "compact LP subsolver ({}) failed to produce a valid solution",
        compact_backend_name()
    ));
}

[[nodiscard]] std::optional<Clock::time_point> deadline_after(std::optional<double> timeout_seconds) {
    if (!timeout_seconds.has_value())
        return std::nullopt;
    if (*timeout_seconds < 0.0)
        throw std::invalid_argument("timeout must be nonnegative");
    return Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(*timeout_seconds));
}

void validate_acceptance_rule(const double acceptable_factor) {
    if (!std::isfinite(acceptable_factor) || acceptable_factor < 0.0)
        throw std::invalid_argument("acceptable_factor must be finite and nonnegative");
}

[[nodiscard]] std::optional<double> remaining_seconds(const std::optional<Clock::time_point> deadline) {
    if (!deadline.has_value())
        return std::nullopt;
    return std::max(0.0, std::chrono::duration<double>(*deadline - Clock::now()).count());
}

template <class TryReduce>
bool apply_reduction(
    AnnotatedInstance& instance,
    std::vector<Lift>& lifts,
    int& objective_offset,
    const std::string_view name,
    const ReductionCounters& counters,
    TryReduce&& try_reduce
) {
    if (auto reduced = std::forward<TryReduce>(try_reduce)(instance)) {
        require_common_leaf_count(reduced->instance, name);
        lifts.push_back(std::move(reduced->lift));
        objective_offset += reduced->objective_offset;
        if (counters.reductions != nullptr)
            *counters.reductions = reduced->reduction_count;
        if (counters.subinstances != nullptr)
            *counters.subinstances = reduced->subinstance_count;
        if (counters.largest_subinstance != nullptr)
            *counters.largest_subinstance = reduced->largest_subinstance;
        if (counters.subinstance_seconds != nullptr)
            *counters.subinstance_seconds = reduced->subinstance_seconds;
        instance = std::move(reduced->instance);
        return true;
    }
    if (counters.reductions != nullptr)
        *counters.reductions = 0;
    if (counters.subinstances != nullptr)
        *counters.subinstances = 0;
    if (counters.largest_subinstance != nullptr)
        *counters.largest_subinstance = 0;
    if (counters.subinstance_seconds != nullptr)
        *counters.subinstance_seconds = 0.0;
    return false;
}

[[nodiscard]] Result solve_exact(
    const AnnotatedInstance& instance,
    const std::vector<std::vector<int>>* const seed_columns = nullptr,
    std::vector<std::vector<int>>* const generated_columns = nullptr,
    const Result* const initial_solution = nullptr,
    const bool allow_abort_with_incumbent = false,
    const int objective_offset = 0,
    const std::optional<Clock::time_point> deadline = std::nullopt,
    const double acceptable_factor = 1.0,
    const int acceptable_offset = 0,
    const LogLevel log_level = LogLevel::NORMAL
) {
    if (generated_columns != nullptr)
        generated_columns->clear();
    const auto time_limit_seconds = remaining_seconds(deadline);
    if (time_limit_seconds.has_value() && *time_limit_seconds <= 0.0) {
        if (allow_abort_with_incumbent)
            return singleton_result(instance);
        throw std::runtime_error("exact solve timed out before start");
    }
    if (instance_has_at_most_leaves(instance, constants::small_exact_leaves) && !has_edge_state(instance, EdgeState::FORCED))
        return solve_small_exact(instance);
    if (should_use_compact_lp(instance))
        return solve_compact_lp_or_throw(
            instance,
            log_level,
            time_limit_seconds,
            allow_abort_with_incumbent,
            objective_offset
        );
    return solve_with_scip_branch_and_price(
        instance,
        log_level,
        seed_columns ? std::span(*seed_columns) : std::span<const std::vector<int>>{},
        generated_columns,
        initial_solution,
        allow_abort_with_incumbent,
        objective_offset,
        time_limit_seconds,
        acceptable_factor,
        acceptable_offset
    );
}

[[nodiscard]] std::optional<Result> solve_trivial(const AnnotatedInstance& instance) {
    if (instance.trees.empty())
        throw std::invalid_argument("solve requires at least one tree");

    const int leaves = instance.trees.front().leaves();
    if (leaves < 1)
        throw std::invalid_argument("solve requires at least one leaf");
    if (leaves > 1)
        return std::nullopt;

    return Result{
        .partition = {{0}},
        .feasible = true,
    };
}

struct PreClusterReductions {
    std::vector<Lift> lifts;
    int cherry = 0;
    int chain = 0;
};

[[nodiscard]] PreClusterReductions apply_pre_cluster_reductions(
    AnnotatedInstance& instance,
    PresolveTimings& timings
) {
    PreClusterReductions summary;
    while (true) {
        if (auto next = timed(timings.cherry, [&] { return try_cherry_picking(instance); })) {
            require_common_leaf_count(next->instance, "cherry picking");
            const int before = instance.trees.front().leaves();
            summary.lifts.push_back(std::move(next->lift));
            instance = std::move(next->instance);
            summary.cherry += before - instance.trees.front().leaves();
            continue;
        }
        if (auto next = timed(timings.chain, [&] { return try_chain_rule(instance); })) {
            require_common_leaf_count(next->instance, "chain reduction");
            const int before = instance.trees.front().leaves();
            summary.lifts.push_back(std::move(next->lift));
            instance = std::move(next->instance);
            summary.chain += before - instance.trees.front().leaves();
            continue;
        }
        break;
    }
    return summary;
}

[[nodiscard]] PostCutCherryCounts apply_post_cut_cherry(
    AnnotatedInstance& instance,
    std::vector<Lift>& lifts,
    int& objective_offset,
    PresolveTimings& timings
) {
    PostCutCherryCounts counts;
    while (true) {
        int three_two = 0;
        int five_three = 0;
        apply_reduction(instance, lifts, objective_offset, "3-2 reduction", {.reductions = &three_two}, [&](const AnnotatedInstance& current) {
            return timed(timings.three_two, [&] { return try_three_two_reduction(current); });
        });
        apply_reduction(instance, lifts, objective_offset, "5-3 reduction", {.reductions = &five_three}, [&](const AnnotatedInstance& current) {
            return timed(timings.five_three, [&] { return try_five_three_reduction(current); });
        });
        if (three_two == 0 && five_three == 0)
            break;
        counts.three_two += three_two;
        counts.five_three += five_three;
        while (auto next = timed(timings.cherry, [&] { return try_cherry_picking(instance); })) {
            require_common_leaf_count(next->instance, "post-cut cherry picking");
            const int before = instance.trees.front().leaves();
            lifts.push_back(std::move(next->lift));
            instance = std::move(next->instance);
            counts.cherry += before - instance.trees.front().leaves();
        }
    }
    return counts;
}

class PresolveRoundRunner {
public:
    PresolveRoundRunner(
        AnnotatedInstance& instance,
        std::vector<Lift>& lifts,
        int& objective_offset,
        PresolveTimings& timings,
        PresolveRoundStats& round,
        SolveContext& context,
        bool& cut_aware_enabled
    ) : instance_(instance),
        lifts_(lifts),
        objective_offset_(objective_offset),
        timings_(timings),
        round_(round),
        context_(context),
        cut_aware_enabled_(cut_aware_enabled) {}

    void apply_pre_cluster() {
        auto pre = apply_pre_cluster_reductions(instance_, timings_);
        round_.cherry += pre.cherry;
        round_.chain += pre.chain;
        lifts_.insert(
            lifts_.end(),
            std::make_move_iterator(pre.lifts.begin()),
            std::make_move_iterator(pre.lifts.end())
        );
    }

    [[nodiscard]] int apply_post_cut(const bool record_zero) {
        const auto post = apply_post_cut_cherry(instance_, lifts_, objective_offset_, timings_);
        round_.add_post_cut_counts(post, record_zero);
        cut_aware_enabled_ = cut_aware_enabled_ || post.three_two > 0 || post.five_three > 0;
        return post.three_two + post.five_three;
    }

    [[nodiscard]] bool apply_normal_cluster() {
        return apply_cluster("cluster reduction", timings_.cluster, [&](const AnnotatedInstance& current) {
            return try_cluster_reduction(current, context_, objective_offset_);
        });
    }

    [[nodiscard]] bool apply_cut_aware_cluster() {
        return apply_cluster("cut-aware cluster reduction", timings_.cut_aware_cluster, [&](const AnnotatedInstance& current) {
            return try_cut_aware_cluster_reduction(current, context_, objective_offset_);
        });
    }

private:
    template <class Reducer>
    [[nodiscard]] bool apply_cluster(
        const std::string_view name,
        double& cluster_seconds,
        Reducer&& reducer
    ) {
        const int before = instance_.trees.front().leaves();
        double subinstance_seconds = 0.0;
        const bool used = timed(cluster_seconds, [&] {
            return apply_reduction(instance_, lifts_, objective_offset_, name, {
                .reductions = &round_.cluster,
                .subinstances = &round_.subinstances,
                .largest_subinstance = &round_.largest_subinstance,
                .subinstance_seconds = &subinstance_seconds,
            }, [&](const AnnotatedInstance& current) {
                context_.check_timeout();
                return std::forward<Reducer>(reducer)(current);
            });
        });
        timings_.subinstances += subinstance_seconds;
        if (used)
            round_.cluster = before - instance_.trees.front().leaves();
        return used;
    }

    AnnotatedInstance& instance_;
    std::vector<Lift>& lifts_;
    int& objective_offset_;
    PresolveTimings& timings_;
    PresolveRoundStats& round_;
    SolveContext& context_;
    bool& cut_aware_enabled_;
};

[[nodiscard]] Result apply_lifts(Result result, std::vector<Lift>& lifts) {
    for (int i = static_cast<int>(lifts.size()) - 1; i >= 0; --i)
        result = lifts[i](std::move(result));
    return result;
}

[[nodiscard]] std::string presolve_row(
    const double seconds,
    const std::string_view round,
    const int leaves,
    const int dualbound,
    const int cherry,
    const int chain,
    const std::optional<int> three_two,
    const std::optional<int> five_three,
    const int cluster,
    const int subinstances,
    const int largest_subinstance,
    const bool show_reductions = true
) {
    const auto reduction_field = [&](const int width, const int value) {
        return show_reductions ? std::format("{:>{}}", value, width) : std::format("{:>{}}", "n/a", width);
    };
    const std::string three_two_field = show_reductions && three_two.has_value()
        ? std::format("{:>3}", *three_two)
        : std::format("{:>3}", "n/a");
    const std::string five_three_field = show_reductions && five_three.has_value()
        ? std::format("{:>3}", *five_three)
        : std::format("{:>3}", "n/a");
    return std::format(
        "presolve: {:5.1f}s | {:>5} | {:>6} | {:>9} | {} | {} | {} | {} | {} | {} | {}",
        seconds,
        round,
        leaves,
        dualbound,
        reduction_field(6, cherry),
        reduction_field(5, chain),
        three_two_field,
        five_three_field,
        reduction_field(7, cluster),
        reduction_field(4, subinstances),
        reduction_field(6, largest_subinstance)
    );
}

void log_presolve_round(
    const LogLevel log_level,
    int& presolve_round,
    const Clock::time_point presolve_start,
    const AnnotatedInstance& instance,
    const int objective_offset,
    const PresolveRoundStats& round
) {
    if (!logging::enabled(log_level) || !round.has_progress())
        return;

    ++presolve_round;
    if (presolve_round % constants::presolve_header_repeat_rounds == 0)
        logging::line(kPresolveHeader);
    logging::line(presolve_row(
        std::chrono::duration<double>(Clock::now() - presolve_start).count(),
        std::format("{}", presolve_round),
        instance.trees.front().leaves(),
        objective_offset,
        round.cherry,
        round.chain,
        round.three_two,
        round.five_three,
        round.cluster,
        round.subinstances,
        round.largest_subinstance
    ));
}

} // namespace

SolveContext::SolveContext(const SolveOptions& options)
    : final_exact_deadline_(deadline_after(options.timeout_seconds)),
      acceptable_factor_(options.acceptable_factor),
      acceptable_offset_(options.acceptable_offset),
      log_level_(options.log_level),
      heuristic_mode_(options.heuristic_mode) {
    validate_acceptance_rule(acceptable_factor_);
}

void SolveContext::check_timeout() const {
    if (!final_exact_deadline_.has_value())
        return;
    if (timeout_check_skip_ > 0) {
        --timeout_check_skip_;
        return;
    }
    timeout_check_skip_ = constants::timeout_check_interval - 1;
    if (Clock::now() >= *final_exact_deadline_)
        throw SubsolveTimeout("solve timed out");
}

std::optional<Clock::time_point> SolveContext::residual_deadline(const AnnotatedInstance& instance) const {
    if (!final_exact_deadline_.has_value())
        return std::nullopt;

    check_timeout();
    const int subproblem_leaves = instance.trees.front().leaves();
    const int problem_leaves = std::max(active_problem_leaves_, subproblem_leaves);
    const int remaining_leaves = std::max(1, problem_leaves - subproblem_leaves);
    const double remaining_seconds = std::chrono::duration<double>(*final_exact_deadline_ - Clock::now()).count();
    const double subproblem_weight = static_cast<double>(std::max(1, subproblem_leaves));
    const double remaining_weight = static_cast<double>(remaining_leaves);
    const double budget_fraction = std::clamp(
        subproblem_weight /
            (subproblem_weight + constants::residual_deadline_remaining_weight_factor * remaining_weight),
        constants::residual_deadline_min_fraction,
        constants::residual_deadline_max_fraction
    );
    const double budget_seconds = std::clamp(
        remaining_seconds * budget_fraction,
        constants::residual_deadline_min_seconds,
        remaining_seconds
    );
    return Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(budget_seconds));
}

bool SolveContext::deadline_expired(const std::optional<Clock::time_point> deadline) const {
    return deadline.has_value() && Clock::now() + std::chrono::milliseconds(50) >= *deadline;
}

bool SolveContext::subsolve_deadline_expired(const std::optional<Clock::time_point> deadline) const {
    return deadline_expired(deadline);
}

std::optional<Result> SolveContext::try_heuristic_colgen(
    const AnnotatedInstance& instance,
    std::vector<Lift>& lifts,
    const int objective_offset
) {
#if MAFFE_HAVE_CONICBUNDLE
    if (!heuristic_mode_ ||
        !final_exact_deadline_.has_value() ||
        instance.trees.size() != 2 ||
        instance.trees.front().leaves() < constants::heuristic_colgen_min_leaves) {
        return std::nullopt;
    }

    if (logging::enabled(log_level_)) {
        logging::line(
            "heuristic-colgen: enabled leaves=", instance.trees.front().leaves(),
            " threshold=", constants::heuristic_colgen_min_leaves,
            " objective-offset=", objective_offset,
            " time-left=", remaining_seconds(final_exact_deadline_).value_or(0.0),
            "s"
        );
    }

    Result heuristic_result = heuristic::solve_colgen(
        instance,
        *final_exact_deadline_,
        nullptr,
        objective_offset,
        log_level_
    );
    const int heuristic_components = static_cast<int>(heuristic_result.partition.size());
    const int heuristic_objective = heuristic_components + objective_offset;
    const double remaining_after_heuristic = remaining_seconds(final_exact_deadline_).value_or(0.0);
    if (logging::enabled(log_level_)) {
        logging::line(
            "heuristic-colgen: handoff incumbent-components=", heuristic_components,
            " incumbent-objective=", heuristic_objective,
            " feasible=", heuristic_result.feasible,
            " normal-solver-time=", remaining_after_heuristic,
            "s"
        );
    }

    if (heuristic_result.feasible &&
        remaining_after_heuristic > constants::heuristic_colgen_min_set_packing_seconds) {
        try {
            Result exact_result = solve_exact(
                instance,
                nullptr,
                nullptr,
                &heuristic_result,
                true,
                objective_offset,
                final_exact_deadline_,
                acceptable_factor_,
                acceptable_offset_,
                log_level_
            );
            if (usable_result(exact_result) &&
                static_cast<int>(exact_result.partition.size()) <= heuristic_components) {
                if (logging::enabled(log_level_)) {
                    logging::line(
                        "heuristic-colgen: normal solver accepted components=",
                        exact_result.partition.size(),
                        " objective=",
                        static_cast<int>(exact_result.partition.size()) + objective_offset
                    );
                }
                return apply_lifts(std::move(exact_result), lifts);
            }
            if (logging::enabled(log_level_)) {
                logging::line(
                    "heuristic-colgen: keep incumbent source=column-generation components=",
                    heuristic_components,
                    " objective=",
                    heuristic_objective,
                    " reason=normal-solver-did-not-improve"
                );
            }
        } catch (const std::runtime_error&) {
            if (!deadline_expired(final_exact_deadline_))
                throw;
            if (logging::enabled(log_level_)) {
                logging::line(
                    "heuristic-colgen: keep incumbent source=column-generation components=",
                    heuristic_components,
                    " objective=",
                    heuristic_objective,
                    " reason=deadline"
                );
            }
        }
    } else if (logging::enabled(log_level_)) {
        logging::line(
            "heuristic-colgen: keep incumbent source=column-generation components=",
            heuristic_components,
            " objective=",
            heuristic_objective,
            " reason=no-time-for-normal-solver"
        );
    }
    return apply_lifts(std::move(heuristic_result), lifts);
#else
    (void)instance;
    (void)lifts;
    (void)objective_offset;
    return std::nullopt;
#endif
}

Result SolveContext::solve(AnnotatedInstance instance) {
    std::vector<Lift> lifts;
    int objective_offset = 0;
    int presolve_round = 0;
    bool cut_aware_enabled = has_edge_state(instance, EdgeState::CUT) && !has_edge_state(instance, EdgeState::FORCED);
    PresolveTimings timings;
    const int initial_leaves = instance.trees.front().leaves();
    const auto presolve_start = Clock::now();

    if (logging::enabled(log_level_)) {
        logging::line(kPresolveHeader);
        logging::line(presolve_row(0.0, "0", initial_leaves, objective_offset, 0, 0, std::nullopt, std::nullopt, 0, 0, 0, false));
    }

    while (true) {
        try {
            check_timeout();
            if (const auto trivial = solve_trivial(instance))
                return apply_lifts(*trivial, lifts);
            active_problem_leaves_ = instance.trees.front().leaves();

            PresolveRoundStats round;
            PresolveRoundRunner runner(instance, lifts, objective_offset, timings, round, *this, cut_aware_enabled);
            if (!has_non_unknown_edge_states(instance)) {
                runner.apply_pre_cluster();
                active_problem_leaves_ = instance.trees.front().leaves();
                const bool used_cluster = runner.apply_normal_cluster();
                if (!used_cluster && instance.trees.size() == 2) {
                    const int cut_reductions = runner.apply_post_cut(true);
                    if (cut_reductions > 0 &&
                        has_non_unknown_edge_states(instance) &&
                        !has_edge_state(instance, EdgeState::FORCED)) {
                        const bool used_cut_cluster = runner.apply_cut_aware_cluster();
                        if (used_cut_cluster)
                            static_cast<void>(runner.apply_post_cut(true));
                    }
                }
            } else if (cut_aware_enabled && instance.trees.size() == 2 && !has_edge_state(instance, EdgeState::FORCED)) {
                static_cast<void>(runner.apply_post_cut(true));
                const bool used_cluster = runner.apply_cut_aware_cluster();
                if (used_cluster)
                    static_cast<void>(runner.apply_post_cut(true));
            }
            log_presolve_round(log_level_, presolve_round, presolve_start, instance, objective_offset, round);
            if (round.cluster > 0)
                continue;
            break;
        } catch (const SubsolveTimeout&) {
            break;
        }
    }

    if (logging::enabled(log_level_)) {
        const int removed = initial_leaves - instance.trees.front().leaves();
        const double reduction_pct =
            initial_leaves > 0 ? 100.0 * static_cast<double>(removed) / static_cast<double>(initial_leaves) : 0.0;
        const double seconds = std::chrono::duration<double>(Clock::now() - presolve_start).count();
        logging::line("presolve: summary");
        logging::line(std::format("presolve:   total      {:.1f}s   rounds={}", seconds, presolve_round));
        logging::line(std::format("presolve:   dualbound  {}", objective_offset));
        logging::line(std::format(
            "presolve:   leaves     {} -> {} removed={} ({:.1f}%)",
            initial_leaves,
            instance.trees.front().leaves(),
            removed,
            reduction_pct
        ));
        logging::line(std::format(
            "presolve:   simple     cherry={:.3f}s chain={:.3f}s 3-2={:.3f}s 5-3={:.3f}s",
            timings.cherry,
            timings.chain,
            timings.three_two,
            timings.five_three
        ));
        logging::line(std::format(
            "presolve:   cluster    normal={:.3f}s cut-aware={:.3f}s",
            timings.cluster,
            timings.cut_aware_cluster
        ));
        logging::line(std::format(
            "presolve:   subsolves  {:.3f}s   included in cluster times",
            timings.subinstances
        ));
    }

    if (auto heuristic_result = try_heuristic_colgen(instance, lifts, objective_offset))
        return *heuristic_result;

    Result result;
    try {
        result = solve_exact(
            instance,
            nullptr,
            nullptr,
            nullptr,
            true,
            objective_offset,
            final_exact_deadline_,
            acceptable_factor_,
            acceptable_offset_,
            log_level_
        );
    } catch (const std::runtime_error&) {
        if (!deadline_expired(final_exact_deadline_))
            throw;
        result = singleton_result(instance);
    }
    result = apply_lifts(std::move(result), lifts);
    return result;
}

Result SolveContext::solve_residual(
    AnnotatedInstance instance,
    const std::vector<std::vector<int>>* const seed_columns,
    std::vector<std::vector<int>>* const generated_columns,
    const Result* const initial_solution,
    const int objective_offset
) {
    if (generated_columns != nullptr)
        generated_columns->clear();
    if (const auto trivial = solve_trivial(instance))
        return *trivial;

    std::vector<Lift> lifts;
    int local_objective_offset = objective_offset;
    if (!has_non_unknown_edge_states(instance)) {
        PresolveTimings timings;
        auto pre = apply_pre_cluster_reductions(instance, timings);
        lifts = std::move(pre.lifts);
        if (instance.trees.size() == 2) {
            static_cast<void>(apply_post_cut_cherry(instance, lifts, local_objective_offset, timings));
        }
    }
    const bool reductions_changed_instance = !lifts.empty() || local_objective_offset != objective_offset;
    const auto* exact_seed_columns = reductions_changed_instance ? nullptr : seed_columns;
    const auto* exact_initial_solution = reductions_changed_instance ? nullptr : initial_solution;
    auto* exact_generated_columns = reductions_changed_instance ? nullptr : generated_columns;
    const LogLevel residual_log_level =
        logging::enabled(log_level_, LogLevel::VERBOSE) &&
            instance.trees.front().leaves() > constants::verbose_residual_solve_leaves
        ? LogLevel::VERBOSE
        : LogLevel::QUIET;
    const auto local_deadline = residual_deadline(instance);
    try {
        return apply_lifts(solve_exact(
            instance,
            exact_seed_columns,
            exact_generated_columns,
            exact_initial_solution,
            local_deadline.has_value(),
            local_objective_offset,
            local_deadline,
            1.0,
            0,
            residual_log_level
        ), lifts);
    } catch (const std::runtime_error& ex) {
        if (subsolve_deadline_expired(local_deadline))
            return apply_lifts(singleton_result(instance), lifts);
        throw;
    }
}

Result solve_annotated(
    AnnotatedInstance instance,
    const std::optional<double> timeout_seconds,
    const double acceptable_factor,
    const int acceptable_offset,
    const LogLevel log_level,
    const bool heuristic_mode
) {
    SolveContext context(SolveOptions{
        .timeout_seconds = timeout_seconds,
        .acceptable_factor = acceptable_factor,
        .acceptable_offset = acceptable_offset,
        .log_level = log_level,
        .heuristic_mode = heuristic_mode,
    });
    return context.solve(std::move(instance));
}

} // namespace maffe
