#include "maffe/common.hpp"
#include "reductions/reductions.hpp"
#include "reductions/detail/common_cluster_forest.hpp"
#include "reductions/detail/related_side.hpp"
#include "util/partition_ops.hpp"
#include "util/constants.hpp"

#include <array>
#include <algorithm>
#include <format>
#include <optional>
#include <queue>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>

namespace maffe {
namespace {

using detail::CommonClusterForest;
using detail::SubinstanceStats;

void assert_cluster_side_is_small_enough(
    const int whole_leaves,
    const int reduced_leaves,
    const char* context
) {
    if (2 * reduced_leaves > whole_leaves + 2) {
        throw std::runtime_error(std::format(
            "cluster reduction selected oversized {} side: reduced_leaves={} whole_leaves={}",
            context,
            reduced_leaves,
            whole_leaves
        ));
    }
}

struct ClusterSummary {
    std::vector<std::vector<int>> closed_blocks;
    std::vector<int> bridge_block;
    bool attached = false;
};

struct ClusterState {
    int current_size = 0;
    int pending_children = 0;
    bool solved = false;
    std::optional<ClusterSummary> summary;
};

struct ChildData {
    std::vector<const ClusterSummary*> summaries;
    std::vector<std::vector<int>> bridge_blocks;
    std::vector<std::vector<int>> closed_blocks;
};

using ChildSummarySpan = std::span<const ClusterSummary* const>;

struct LocalToken {
    int original_leaf = -1;
    int child_index = -1;
};

struct RegionTokens {
    std::vector<LocalToken> tokens;
    std::vector<int> local_leaf_of_original;
    std::vector<int> local_leaf_of_child;
};

[[nodiscard]] int child_lookup(const std::vector<std::pair<int, int>>& child_roots, const int node) {
    const auto it = std::ranges::lower_bound(
        child_roots,
        node,
        {},
        &std::pair<int, int>::first
    );
    if (it == child_roots.end() || it->first != node)
        return -1;
    return it->second;
}

[[nodiscard]] std::vector<std::pair<int, int>> sorted_child_roots(
    const CommonClusterForest& forest,
    const std::span<const int> child_ids,
    const int tree
) {
    std::vector<std::pair<int, int>> child_roots;
    child_roots.reserve(child_ids.size());
    for (int i = 0; i < static_cast<int>(child_ids.size()); ++i)
        child_roots.emplace_back(forest.clusters[child_ids[i]].roots[tree], i);
    std::ranges::sort(child_roots);
    return child_roots;
}

void append_closed_blocks(std::vector<std::vector<int>>& dst, const std::vector<std::vector<int>>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

[[nodiscard]] std::vector<int> expand_token(
    const LocalToken& token,
    const std::span<const std::vector<int>> child_bridge_blocks
) {
    if (token.child_index < 0)
        return {token.original_leaf};
    return child_bridge_blocks[token.child_index];
}

[[nodiscard]] ChildData collect_child_data(
    const std::vector<ClusterState>& states,
    const std::span<const int> cluster_ids
) {
    ChildData child_data;
    child_data.summaries.reserve(cluster_ids.size());
    child_data.bridge_blocks.reserve(cluster_ids.size());
    for (const int cluster_id : cluster_ids) {
        const auto& summary = states[cluster_id].summary;
        if (!summary.has_value())
            throw std::runtime_error("cluster reduction expected solved child summary");
        child_data.summaries.push_back(&*summary);
        child_data.bridge_blocks.push_back(summary->bridge_block);
        append_closed_blocks(child_data.closed_blocks, summary->closed_blocks);
    }
    return child_data;
}

[[nodiscard]] RegionTokens build_region_tokens(
    const AnnotatedInstance& instance,
    const CommonClusterForest& forest,
    const int tree0_root,
    const std::span<const int> child_ids,
    const ChildSummarySpan child_summaries
) {
    RegionTokens region{
        .tokens = {},
        .local_leaf_of_original = std::vector<int>(instance.trees.front().leaves(), -1),
        .local_leaf_of_child = std::vector<int>(child_ids.size(), -1),
    };

    const auto child_roots = sorted_child_roots(forest, child_ids, 0);

    const auto collect = [&](this auto&& self, const int node) -> void {
        const int child_index = child_lookup(child_roots, node);
        if (child_index >= 0) {
            if (child_summaries[child_index]->attached) {
                region.local_leaf_of_child[child_index] = static_cast<int>(region.tokens.size());
                region.tokens.push_back(LocalToken{
                    .original_leaf = -1,
                    .child_index = child_index,
                });
            }
            return;
        }
        if (node < instance.trees.front().leaves()) {
            region.local_leaf_of_original[node] = static_cast<int>(region.tokens.size());
            region.tokens.push_back(LocalToken{
                .original_leaf = node,
                .child_index = -1,
            });
            return;
        }
        const auto [left, right] = forest.intervals[0].children[node];
        self(left);
        self(right);
    };
    collect(tree0_root);
    return region;
}

[[nodiscard]] bool region_subinstances_fit_cap(
    const AnnotatedInstance& instance,
    const CommonClusterForest& forest,
    const int tree0_root,
    const std::span<const int> child_ids,
    const ChildSummarySpan child_summaries,
    const int max_leaves
) {
    const RegionTokens region = build_region_tokens(instance, forest, tree0_root, child_ids, child_summaries);
    if (region.tokens.empty())
        return true;
    return static_cast<int>(region.tokens.size()) + 1 <= max_leaves;
}

[[nodiscard]] Tree build_region_tree(
    const Tree& tree,
    const std::span<const std::array<int, 2>> children,
    const int subtree_root,
    const std::vector<std::pair<int, int>>& child_roots,
    const ChildSummarySpan child_summaries,
    const RegionTokens& region,
    const bool with_placeholder,
    const int placeholder_child_index
) {
    const int token_count = static_cast<int>(region.tokens.size());
    const int leaf_count = token_count + (with_placeholder ? 1 : 0);
    const int placeholder = with_placeholder ? token_count : -1;
    Tree reduced{
        .parent = std::vector<int>(leaf_count, -2),
        .edge_state = std::vector<EdgeState>(leaf_count, EdgeState::UNKNOWN),
    };

    int next = leaf_count;
    const auto copy = [&](this auto&& self, const int node) -> int {
        const int child_index = child_lookup(child_roots, node);
        if (child_index >= 0) {
            if (!child_summaries[child_index]->attached) {
                if (child_index == placeholder_child_index) {
                    if (placeholder < 0)
                        throw std::runtime_error("cluster reduction expected placeholder leaf for detached child");
                    return placeholder;
                }
                return -1;
            }
            return region.local_leaf_of_child[child_index];
        }
        if (node < tree.leaves())
            return region.local_leaf_of_original[node];

        const auto [left, right] = children[node];
        const int a = self(left);
        const int b = self(right);
        if (a < 0 || b < 0)
            return a < 0 ? b : a;

        const int current = next++;
        reduced.parent.resize(next, -2);
        reduced.edge_state.resize(next, EdgeState::UNKNOWN);
        reduced.parent[a] = current;
        reduced.parent[b] = current;
        return current;
    };

    const int local_root = copy(subtree_root);
    if (local_root < 0)
        throw std::runtime_error("cluster reduction removed all active leaves in a region");

    if (!with_placeholder || placeholder_child_index >= 0) {
        reduced.parent[local_root] = -1;
        reduced.edge_state[local_root] = EdgeState::UNKNOWN;
        return reduced;
    }

    const int root = next++;
    reduced.parent.resize(next, -2);
    reduced.edge_state.resize(next, EdgeState::UNKNOWN);
    reduced.parent[local_root] = root;
    reduced.parent[placeholder] = root;
    reduced.parent[root] = -1;
    reduced.edge_state[root] = EdgeState::UNKNOWN;
    return reduced;
}

[[nodiscard]] AnnotatedInstance build_region_instance(
    const AnnotatedInstance& instance,
    const CommonClusterForest& forest,
    const std::vector<int>& region_roots,
    const std::span<const int> child_ids,
    const ChildSummarySpan child_summaries,
    const RegionTokens& region,
    const bool with_placeholder,
    const int placeholder_child_index = -1
) {
    AnnotatedInstance reduced;
    reduced.trees.reserve(instance.trees.size());
    for (int tree = 0; tree < static_cast<int>(instance.trees.size()); ++tree) {
        reduced.trees.push_back(build_region_tree(
            instance.trees[tree],
            forest.intervals[tree].children,
            region_roots[tree],
            sorted_child_roots(forest, child_ids, tree),
            child_summaries,
            region,
            with_placeholder,
            placeholder_child_index
        ));
    }
    return reduced;
}

[[nodiscard]] ClusterSummary lift_region_summary(
    const Result& local_result,
    const RegionTokens& region,
    const std::span<const std::vector<int>> child_bridge_blocks,
    std::vector<std::vector<int>> closed_blocks,
    const int placeholder,
    const bool attached,
    const bool require_attached_bridge = true
) {
    ClusterSummary summary{
        .closed_blocks = std::move(closed_blocks),
        .bridge_block = {},
        .attached = attached,
    };

    for (const auto& block : local_result.partition) {
        const bool has_placeholder = placeholder >= 0 && std::ranges::find(block, placeholder) != block.end();
        std::vector<int> expanded;
        for (const int leaf : block) {
            if (leaf == placeholder)
                continue;
            const auto part = expand_token(region.tokens[leaf], child_bridge_blocks);
            expanded.insert(expanded.end(), part.begin(), part.end());
        }
        if (expanded.empty()) {
            if (has_placeholder)
                continue;
            throw std::runtime_error("cluster reduction produced an empty non-placeholder block");
        }
        std::ranges::sort(expanded);
        if (has_placeholder) {
            if (!attached)
                throw std::runtime_error("cluster reduction saw placeholder block without attachment");
            summary.bridge_block = std::move(expanded);
        } else {
            summary.closed_blocks.push_back(std::move(expanded));
        }
    }

    if (attached && require_attached_bridge && summary.bridge_block.empty())
        throw std::runtime_error("cluster reduction attachment lost the bridge block");
    detail::sort_partition_blocks(summary.closed_blocks);
    return summary;
}

[[nodiscard]] ClusterSummary solve_region_summary(
    const AnnotatedInstance& instance,
    const CommonClusterForest& forest,
    const std::vector<int>& region_roots,
    const int tree0_root,
    const std::span<const int> child_ids,
    const ChildSummarySpan child_summaries,
    SolveContext& context,
    const int objective_offset,
    const char* placeholder_context,
    SubinstanceStats& stats,
    const int placeholder_child_index = -1
) {
    const RegionTokens region = build_region_tokens(instance, forest, tree0_root, child_ids, child_summaries);
    ChildData child_data;
    child_data.bridge_blocks.reserve(child_summaries.size());
    for (const ClusterSummary* child_summary : child_summaries) {
        child_data.bridge_blocks.push_back(child_summary->bridge_block);
        append_closed_blocks(child_data.closed_blocks, child_summary->closed_blocks);
    }
    if (region.tokens.empty()) {
        detail::sort_partition_blocks(child_data.closed_blocks);
        return ClusterSummary{
            .closed_blocks = std::move(child_data.closed_blocks),
            .bridge_block = {},
            .attached = false,
        };
    }
    const AnnotatedInstance without_instance = build_region_instance(
        instance,
        forest,
        region_roots,
        child_ids,
        child_summaries,
        region,
        false
    );
    const AnnotatedInstance with_instance = build_region_instance(
        instance,
        forest,
        region_roots,
        child_ids,
        child_summaries,
        region,
        true,
        placeholder_child_index
    );
    const auto related = detail::solve_related_side(with_instance, without_instance, context, objective_offset, stats);
    bool attach = detail::should_attach_placeholder(
        related.with_result,
        related.without_result,
        "cluster reduction",
        placeholder_context
    );
#if MAFFE_COMPETITION_HEURISTIC_BUILD
    if (attach && !detail::has_nonempty_placeholder_block(related.with_result, region.tokens.size()))
        attach = false;
#endif
    return lift_region_summary(
        attach ? related.with_result : related.without_result,
        region,
        child_data.bridge_blocks,
        std::move(child_data.closed_blocks),
        attach ? static_cast<int>(region.tokens.size()) : -1,
        attach
    );
}

struct SelectedCluster {
    int cluster_id = -1;
    int reduced_size = 0;
    bool solve_top_first = false;
};

struct BottomEntry {
    int reduced_size = 0;
    int leaf_count = 0;
    int cluster_id = -1;
};

struct BottomEntryWorse {
    bool operator()(const BottomEntry& lhs, const BottomEntry& rhs) const {
        if (lhs.reduced_size != rhs.reduced_size)
            return lhs.reduced_size > rhs.reduced_size;
        if (lhs.leaf_count != rhs.leaf_count)
            return lhs.leaf_count < rhs.leaf_count;
        return lhs.cluster_id > rhs.cluster_id;
    }
};

struct TopEntry {
    int current_size = 0;
    int cluster_id = -1;
};

struct TopEntryWorse {
    bool operator()(const TopEntry& lhs, const TopEntry& rhs) const {
        if (lhs.current_size != rhs.current_size)
            return lhs.current_size < rhs.current_size;
        return lhs.cluster_id > rhs.cluster_id;
    }
};

[[nodiscard]] bool cluster_enabled(
    const int total_size,
    const int cluster_size
) {
    return total_size - cluster_size >= 2;
}

[[nodiscard]] std::vector<int> frontier_cluster_ids(
    const CommonClusterForest& forest,
    const std::vector<ClusterState>& states,
    const std::span<const int> roots,
    const int stop_cluster = -1
) {
    std::vector<int> frontier;
    const auto collect = [&](this auto&& self, const std::span<const int> ids) -> void {
        for (const int cluster_id : ids) {
            if (cluster_id == stop_cluster || states[cluster_id].solved) {
                frontier.push_back(cluster_id);
                continue;
            }
            self(forest.clusters[cluster_id].children);
        }
    };
    collect(roots);
    return frontier;
}

[[nodiscard]] std::optional<SelectedCluster> select_bottom_cluster(
    std::priority_queue<BottomEntry, std::vector<BottomEntry>, BottomEntryWorse>& ready_bottom,
    const std::vector<ClusterState>& states,
    const int total_size
) {
    while (!ready_bottom.empty()) {
        const BottomEntry entry = ready_bottom.top();
        ready_bottom.pop();
        if (entry.cluster_id < 0 || entry.cluster_id >= static_cast<int>(states.size()))
            continue;
        const auto& state = states[entry.cluster_id];
        if (state.solved || state.pending_children != 0 || state.current_size != entry.reduced_size)
            continue;
        if (!cluster_enabled(total_size, state.current_size))
            continue;
        return SelectedCluster{
            .cluster_id = entry.cluster_id,
            .reduced_size = state.current_size,
            .solve_top_first = false,
        };
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<TopEntry> peek_top_cluster(
    std::priority_queue<TopEntry, std::vector<TopEntry>, TopEntryWorse>& top_candidates,
    const std::vector<ClusterState>& states,
    const int total_size
) {
    while (!top_candidates.empty()) {
        const TopEntry entry = top_candidates.top();
        top_candidates.pop();
        if (entry.cluster_id < 0 || entry.cluster_id >= static_cast<int>(states.size()))
            continue;
        const auto& state = states[entry.cluster_id];
        if (state.solved || state.current_size != entry.current_size)
            continue;
        if (!cluster_enabled(total_size, state.current_size))
            continue;
        return entry;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<SelectedCluster> select_cluster(
    std::priority_queue<BottomEntry, std::vector<BottomEntry>, BottomEntryWorse>& ready_bottom,
    std::priority_queue<TopEntry, std::vector<TopEntry>, TopEntryWorse>& top_candidates,
    const std::vector<ClusterState>& states,
    const int total_size
) {
    const auto bottom = select_bottom_cluster(ready_bottom, states, total_size);
    const auto top = peek_top_cluster(top_candidates, states, total_size);
    if (!bottom.has_value()) {
        if (!top.has_value())
            return std::nullopt;
        top_candidates.push(*top);
        return SelectedCluster{
            .cluster_id = top->cluster_id,
            .reduced_size = total_size - top->current_size + 1,
            .solve_top_first = true,
        };
    }
    if (!top.has_value())
        return bottom;
    const int top_reduced_size = total_size - top->current_size + 1;
    if (top_reduced_size < bottom->reduced_size) {
        ready_bottom.push(BottomEntry{
            .reduced_size = bottom->reduced_size,
            .leaf_count = states[bottom->cluster_id].current_size,
            .cluster_id = bottom->cluster_id,
        });
        top_candidates.push(*top);
        return SelectedCluster{
            .cluster_id = top->cluster_id,
            .reduced_size = top_reduced_size,
            .solve_top_first = true,
        };
    }
    top_candidates.push(*top);
    return bottom;
}

[[nodiscard]] Reduced build_root_residual(
    const AnnotatedInstance& instance,
    const CommonClusterForest& forest,
    const std::vector<ClusterState>& states
) {
    const auto root_child_ids = frontier_cluster_ids(forest, states, forest.top_children);
    auto root_child_data = collect_child_data(states, root_child_ids);

    auto region = build_region_tokens(
        instance,
        forest,
        instance.trees.front().root(),
        root_child_ids,
        ChildSummarySpan(root_child_data.summaries)
    );
    if (region.tokens.size() <= 1) {
        if (!region.tokens.empty())
            root_child_data.closed_blocks.push_back(expand_token(region.tokens.front(), root_child_data.bridge_blocks));
        detail::sort_partition_blocks(root_child_data.closed_blocks);
        AnnotatedInstance reduced_instance;
        reduced_instance.trees.reserve(instance.trees.size());
        for (int tree = 0; tree < static_cast<int>(instance.trees.size()); ++tree) {
            reduced_instance.trees.push_back(Tree{
                .parent = {-1},
                .edge_state = {EdgeState::UNKNOWN},
            });
        }
        return Reduced{
            .instance = std::move(reduced_instance),
            .lift = [result = Result{
                .partition = std::move(root_child_data.closed_blocks),
                .feasible = true,
            }](Result) mutable { // NOLINT(performance-unnecessary-value-param)
                return std::move(result);
            },
            .objective_offset = 0,
        };
    }
    std::vector<int> roots(instance.trees.size());
    for (int tree = 0; tree < static_cast<int>(instance.trees.size()); ++tree)
        roots[tree] = instance.trees[tree].root();
    const int closed_count = static_cast<int>(root_child_data.closed_blocks.size());
    return Reduced{
        .instance = build_region_instance(
            instance,
            forest,
            roots,
            root_child_ids,
            ChildSummarySpan(root_child_data.summaries),
            region,
            false
        ),
        .lift = [
            region = std::move(region),
            bridge_blocks = std::move(root_child_data.bridge_blocks),
            closed_blocks = std::move(root_child_data.closed_blocks)
        ](Result result) mutable {
            if (!result.feasible)
                return result;
            ClusterSummary summary = lift_region_summary(
                result,
                region,
                bridge_blocks,
                std::move(closed_blocks),
                -1,
                false
            );
            return Result{
                .partition = std::move(summary.closed_blocks),
                .feasible = true,
            };
        },
        .objective_offset = closed_count,
    };
}

[[nodiscard]] std::optional<Reduced> reduce_top_first(
    const AnnotatedInstance& instance,
    const CommonClusterForest& forest,
    const SelectedCluster& selection,
    const std::vector<ClusterState>& states,
    SolveContext& context,
    const int objective_offset,
    SubinstanceStats& stats
) {
    const auto& cluster = forest.clusters[selection.cluster_id];
    const auto root_child_ids = frontier_cluster_ids(forest, states, forest.top_children, selection.cluster_id);
    const ClusterSummary unattached_summary{
        .closed_blocks = {},
        .bridge_block = {},
        .attached = false,
    };
    std::vector<const ClusterSummary*> root_child_summaries;
    root_child_summaries.reserve(root_child_ids.size());
    for (const int cluster_id : root_child_ids) {
        if (cluster_id == selection.cluster_id) {
            root_child_summaries.push_back(&unattached_summary);
            continue;
        }
        const auto& summary = states[cluster_id].summary;
        if (!summary.has_value())
            throw std::runtime_error("cluster reduction expected solved top child summary");
        root_child_summaries.push_back(&*summary);
    }
    assert_cluster_side_is_small_enough(
        instance.trees.front().leaves(),
        selection.reduced_size,
        "top"
    );
    std::vector<int> roots(instance.trees.size());
    for (int tree = 0; tree < static_cast<int>(instance.trees.size()); ++tree)
        roots[tree] = instance.trees[tree].root();
    if (!region_subinstances_fit_cap(
            instance,
            forest,
            instance.trees.front().root(),
            root_child_ids,
            ChildSummarySpan(root_child_summaries),
            constants::max_cluster_subinstance_leaves
        )) {
        return std::nullopt;
    }
    ClusterSummary top_summary = solve_region_summary(
        instance,
        forest,
        roots,
        instance.trees.front().root(),
        root_child_ids,
        ChildSummarySpan(root_child_summaries),
        context,
        objective_offset,
        "top-side",
        stats,
        static_cast<int>(std::ranges::find(root_child_ids, selection.cluster_id) - root_child_ids.begin())
    );
    const bool attach_cluster = top_summary.attached;
    const int top_closed_count = static_cast<int>(top_summary.closed_blocks.size());
    const auto bottom_child_ids = frontier_cluster_ids(forest, states, cluster.children);
    auto bottom_child_data = collect_child_data(states, bottom_child_ids);
    const RegionTokens bottom_region = build_region_tokens(
        instance,
        forest,
        cluster.tree0_node,
        bottom_child_ids,
        ChildSummarySpan(bottom_child_data.summaries)
    );
    const int bottom_closed_count = static_cast<int>(bottom_child_data.closed_blocks.size());
    const AnnotatedInstance bottom_instance = build_region_instance(
        instance,
        forest,
        cluster.roots,
        bottom_child_ids,
        ChildSummarySpan(bottom_child_data.summaries),
        bottom_region,
        attach_cluster
    );
    return Reduced{
        .instance = bottom_instance,
        .lift = [
            top_summary = std::move(top_summary),
            bottom_region = bottom_region,
            bottom_closed_blocks = std::move(bottom_child_data.closed_blocks),
            bottom_bridge_blocks = std::move(bottom_child_data.bridge_blocks),
            attach_cluster
        ](Result result) mutable {
            if (!result.feasible)
                return result;
            ClusterSummary bottom_summary = lift_region_summary(
                result,
                bottom_region,
                bottom_bridge_blocks,
                std::move(bottom_closed_blocks),
                attach_cluster ? static_cast<int>(bottom_region.tokens.size()) : -1,
                attach_cluster,
                false
            );
            return detail::combine_side_summaries(std::move(top_summary), std::move(bottom_summary));
        },
        .objective_offset = top_closed_count + bottom_closed_count,
    };
}

} // namespace

std::optional<Reduced> try_cluster_reduction(
    const AnnotatedInstance& instance,
    SolveContext& context,
    const int objective_offset
) {
    for (const auto& tree : instance.trees) {
        if (std::ranges::any_of(tree.edge_state, [](const EdgeState state) { return state != EdgeState::UNKNOWN; }))
            throw std::runtime_error("cluster reduction expects clean trees");
    }
    const CommonClusterForest forest = detail::build_common_cluster_forest(instance);
    if (forest.top_children.empty())
        return std::nullopt;
    std::vector<ClusterState> states(forest.clusters.size());
    std::priority_queue<BottomEntry, std::vector<BottomEntry>, BottomEntryWorse> ready_bottom;
    std::priority_queue<TopEntry, std::vector<TopEntry>, TopEntryWorse> top_candidates;
    for (int cluster_id = 0; cluster_id < static_cast<int>(forest.clusters.size()); ++cluster_id) {
        auto& state = states[cluster_id];
        state.current_size = forest.clusters[cluster_id].leaf_count;
        state.pending_children = static_cast<int>(forest.clusters[cluster_id].children.size());
        if (state.pending_children == 0)
            ready_bottom.push(BottomEntry{
                .reduced_size = state.current_size,
                .leaf_count = forest.clusters[cluster_id].leaf_count,
                .cluster_id = cluster_id,
            });
        top_candidates.push(TopEntry{.current_size = state.current_size, .cluster_id = cluster_id});
    }

    int total_size = instance.trees.front().leaves();
    bool reduced_any = false;
    int reduction_count = 0;
    SubinstanceStats stats;
    while (true) {
        try {
            context.check_timeout();
            const auto selected = select_cluster(ready_bottom, top_candidates, states, total_size);
            if (!selected.has_value())
                break;
            if (selected->reduced_size > constants::max_cluster_subinstance_leaves)
                break;
            if (reduced_any && selected->reduced_size > constants::small_cluster_leaf_threshold)
                break;
            if (selected->solve_top_first) {
                auto reduced = reduce_top_first(instance, forest, *selected, states, context, objective_offset, stats);
                if (!reduced.has_value())
                    break;
                reduced->reduction_count = reduction_count + 1;
                reduced->subinstance_count = stats.solved;
                reduced->largest_subinstance = stats.largest;
                reduced->subinstance_seconds = stats.seconds;
                return reduced;
            }
            assert_cluster_side_is_small_enough(
                instance.trees.front().leaves(),
                selected->reduced_size,
                "cluster"
            );
            const auto& cluster = forest.clusters[selected->cluster_id];
            const auto child_data = collect_child_data(states, cluster.children);
            if (!region_subinstances_fit_cap(
                    instance,
                    forest,
                    cluster.tree0_node,
                    cluster.children,
                    ChildSummarySpan(child_data.summaries),
                    constants::max_cluster_subinstance_leaves
                )) {
                break;
            }
            ClusterSummary summary = solve_region_summary(
                instance,
                forest,
                cluster.roots,
                cluster.tree0_node,
                cluster.children,
                ChildSummarySpan(child_data.summaries),
                context,
                objective_offset,
                "dummy",
                stats
            );
            reduced_any = true;
            ++reduction_count;

            auto& state = states[selected->cluster_id];
            const int delta = (summary.attached ? 1 : 0) - state.current_size;
            state.solved = true;
            state.summary = std::move(summary);
            total_size += delta;

            for (int cluster_id = forest.clusters[selected->cluster_id].parent_cluster;
                 cluster_id >= 0;
                 cluster_id = forest.clusters[cluster_id].parent_cluster) {
                states[cluster_id].current_size += delta;
                top_candidates.push(TopEntry{
                    .current_size = states[cluster_id].current_size,
                    .cluster_id = cluster_id,
                });
            }
            if (const int parent = forest.clusters[selected->cluster_id].parent_cluster; parent >= 0) {
                --states[parent].pending_children;
                if (states[parent].pending_children == 0) {
                    ready_bottom.push(BottomEntry{
                        .reduced_size = states[parent].current_size,
                        .leaf_count = forest.clusters[parent].leaf_count,
                        .cluster_id = parent,
                    });
                }
            }
        } catch (const SubsolveTimeout&) {
            break;
        }
    }
    if (!reduced_any)
        return std::nullopt;
    auto reduced = build_root_residual(instance, forest, states);
    reduced.reduction_count = reduction_count;
    reduced.subinstance_count = stats.solved;
    reduced.largest_subinstance = stats.largest;
    reduced.subinstance_seconds = stats.seconds;
    return reduced;
}

} // namespace maffe
