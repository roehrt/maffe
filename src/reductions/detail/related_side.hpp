#pragma once

#include "reductions/reductions.hpp"
#include "reductions/detail/binary_tree_view.hpp"
#include "util/partition_ops.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <iterator>
#include <utility>
#include <optional>
#include <stdexcept>
#include <vector>

#ifndef MAFFE_COMPETITION_HEURISTIC_BUILD
#define MAFFE_COMPETITION_HEURISTIC_BUILD 0
#endif

namespace maffe::detail {

// Join the two sides of a cluster, including their possible bridge component.
template <class Summary>
[[nodiscard]] Result combine_side_summaries(Summary first, Summary second) {
    Result result{.partition = std::move(first.closed_blocks)};
    result.partition.insert(result.partition.end(),
        std::make_move_iterator(second.closed_blocks.begin()),
        std::make_move_iterator(second.closed_blocks.end()));
    first.bridge_block.insert(first.bridge_block.end(), second.bridge_block.begin(), second.bridge_block.end());
    if (!first.bridge_block.empty())
        result.partition.push_back(std::move(first.bridge_block));
    sort_partition_blocks(result.partition);
    return result;
}

struct RelatedSideResults {
    Result with_result;
    Result without_result;
};

struct SubinstanceStats {
    int solved = 0;
    int largest = 0;
    double seconds = 0.0;

    void record(const AnnotatedInstance& instance) {
        const int leaves = instance.trees.front().leaves();
        ++solved;
        largest = std::max(largest, leaves);
    }
};

template <class Solve>
[[nodiscard]] inline Result solve_subinstance(
    SubinstanceStats& stats,
    const AnnotatedInstance& instance,
    Solve solve
) {
    stats.record(instance);
    const auto start = std::chrono::steady_clock::now();
    Result result = solve();
    stats.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

[[nodiscard]] inline bool should_attach_placeholder(
    const Result& with_result,
    const Result& without_result,
    const char* reduction_name,
    const char* context
) {
    if (!with_result.feasible || !without_result.feasible)
        throw std::runtime_error(std::format("{} expected a feasible subsolution", reduction_name));
    const int with_objective = static_cast<int>(with_result.partition.size());
    const int without_objective = static_cast<int>(without_result.partition.size());
    if (with_objective != without_objective && with_objective != without_objective + 1) {
#if MAFFE_COMPETITION_HEURISTIC_BUILD
        static_cast<void>(context);
        // Heuristic subsolves may stop with nonoptimal incumbents. The exact
        // relation is then not guaranteed, but either feasible side can still
        // be lifted back, so keep the better incumbent instead of aborting.
        return with_objective <= without_objective;
#else
        throw std::runtime_error(std::format(
            "{} expected {} dummy objective to match or exceed by one: with={} without={}",
            reduction_name,
            context,
            with_objective,
            without_objective
        ));
#endif
    }
    return with_objective == without_objective;
}

[[nodiscard]] inline bool has_nonempty_placeholder_block(
    const Result& result,
    const int placeholder
) {
    return std::ranges::any_of(result.partition, [&](const auto& block) {
        return block.size() > 1 && std::ranges::find(block, placeholder) != block.end();
    });
}

[[nodiscard]] inline std::vector<std::vector<int>> deduplicate_non_singleton_columns(
    std::vector<std::vector<int>> columns
) {
    std::erase_if(columns, [](const auto& block) { return block.size() < 2; });
    sort_partition_blocks(columns);
    columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
    return columns;
}

[[nodiscard]] inline std::optional<Result> try_extend_with_placeholder(
    const AnnotatedInstance& with_instance,
    const Result& without_result
) {
    if (!without_result.feasible)
        return std::nullopt;

    const auto trees = build_binary_tree_views(with_instance);
    const int placeholder = with_instance.trees.front().leaves() - 1;
    Result candidate = without_result;
    for (auto& block : candidate.partition) {
        block.push_back(placeholder);
        if (partition_feasible(trees, candidate.partition)) {
            sort_partition_blocks(candidate.partition);
            return candidate;
        }
        block.pop_back();
    }
    return std::nullopt;
}

[[nodiscard]] inline RelatedSideResults solve_related_side(
    const AnnotatedInstance& with_instance,
    const AnnotatedInstance& without_instance,
    SolveContext& context,
    const int objective_offset,
    SubinstanceStats& stats
) {
    std::vector<std::vector<int>> without_columns;
    Result without_result = solve_subinstance(stats, without_instance, [&] {
        return context.solve_residual(
            without_instance,
            nullptr,
            &without_columns,
            nullptr,
            objective_offset
        );
    });
    if (auto extended = try_extend_with_placeholder(with_instance, without_result)) {
        return RelatedSideResults{
            .with_result = std::move(*extended),
            .without_result = std::move(without_result),
        };
    }

    without_columns.insert(
        without_columns.end(),
        without_result.partition.begin(),
        without_result.partition.end()
    );
    without_columns = deduplicate_non_singleton_columns(std::move(without_columns));

    return RelatedSideResults{
        .with_result = solve_subinstance(stats, with_instance, [&] {
            return context.solve_residual(
                with_instance,
                &without_columns,
                nullptr,
                nullptr,
                objective_offset
            );
        }),
        .without_result = std::move(without_result),
    };
}

} // namespace maffe::detail
