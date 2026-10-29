#include "reductions/detail/cut_aware_cluster.hpp"

#include "maffe/common.hpp"
#include "reductions/detail/related_side.hpp"
#include "reductions/reductions.hpp"
#include "util/partition_ops.hpp"
#include "util/constants.hpp"
#include "util/tree_ops.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace maffe::detail {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] int cut_aware_target_leaves(const int total_leaves) {
    return std::min(constants::small_cut_aware_cluster_leaves, total_leaves - 2);
}

struct Interval {
    int begin = 0;
    int end = 0;

    [[nodiscard]] bool operator==(const Interval &) const = default;
};

struct PartialChoice {
    CutAwarePartialKind kind = CutAwarePartialKind::NONE;
    int component = -1;
    int node = -1;
    int leaf_count = 0;
    std::array<Interval, 2> intervals{};
    int interval_count = 0;
};

struct ChoiceKey {
    CutAwarePartialKind kind = CutAwarePartialKind::NONE;
    int component = -1;
    int node = -1;
    std::array<Interval, 2> intervals{};
    int interval_count = 0;

    [[nodiscard]] bool operator==(const ChoiceKey &) const = default;
};

struct ChoiceKeyHash {
    [[nodiscard]] std::size_t operator()(const ChoiceKey &key) const {
        std::size_t hash = static_cast<std::size_t>(key.kind);
        const auto mix = [&](const int value) {
            hash ^= static_cast<std::size_t>(value) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        };
        mix(key.component);
        mix(key.node);
        mix(key.interval_count);
        for (const auto [begin, end] : key.intervals) {
            mix(begin);
            mix(end);
        }
        return hash;
    }
};

struct ComponentRange {
    int root = -1;
    int begin = 0;
    int end = 0;
};

struct ForestInfo {
    std::vector<std::array<int, 2>> children;
    std::vector<int> parent;
    std::vector<int> component_of_node;
    std::vector<int> component_of_leaf;
    std::vector<int> subtree_begin;
    std::vector<int> subtree_end;
    std::vector<int> component_order;
    std::vector<int> leaf_position;
    std::vector<int> leaf_order;
    std::vector<ComponentRange> components;
    std::vector<std::vector<int>> component_nodes;
    std::vector<std::vector<int>> component_leaves_by_label;
    std::vector<PartialChoice> choices;
};

struct CutAwarePlan {
    CutAwareCluster cluster;
    std::array<ForestInfo, 2> info;
};

void normalize_components(std::vector<int> &components) {
    std::ranges::sort(components);
    components.erase(std::unique(components.begin(), components.end()), components.end());
}

void clear_partial(CutAwareRepresentation &representation) {
    representation.partial_kind = CutAwarePartialKind::NONE;
    representation.partial_component = -1;
    representation.partial_node = -1;
}

[[nodiscard]] CutAwareRepresentation normalize_representation(
    CutAwareRepresentation representation,
    const ForestInfo &forest_info) {
    if (representation.partial_kind != CutAwarePartialKind::NONE &&
        representation.partial_component >= 0 &&
        representation.partial_node >= 0) {
        const auto &range = forest_info.components[representation.partial_component];
        const bool partial_spans_component =
            forest_info.subtree_begin[representation.partial_node] == range.begin &&
            forest_info.subtree_end[representation.partial_node] == range.end;
        if (!partial_spans_component) {
            normalize_components(representation.components);
            return representation;
        }
        if (representation.partial_kind == CutAwarePartialKind::BOTTOM)
            representation.components.push_back(representation.partial_component);
        clear_partial(representation);
    }
    normalize_components(representation.components);
    return representation;
}

[[nodiscard]] CutAwareRepresentation complement_representation(
    const CutAwareRepresentation &representation,
    const ForestInfo &forest_info) {
    CutAwareRepresentation normalized = normalize_representation(representation, forest_info);
    CutAwareRepresentation complement;
    if (normalized.partial_kind == CutAwarePartialKind::BOTTOM)
        complement.partial_kind = CutAwarePartialKind::TOP;
    else if (normalized.partial_kind == CutAwarePartialKind::TOP)
        complement.partial_kind = CutAwarePartialKind::BOTTOM;
    complement.partial_component = normalized.partial_component;
    complement.partial_node = normalized.partial_node;

    std::vector<char> used(forest_info.components.size(), 0);
    for (const int component : normalized.components)
        used[component] = 1;
    if (normalized.partial_component >= 0)
        used[normalized.partial_component] = 1;
    for (int component = 0; component < static_cast<int>(forest_info.components.size()); ++component) {
        if (!used[component])
            complement.components.push_back(component);
    }
    return normalize_representation(std::move(complement), forest_info);
}

[[nodiscard]] ForestInfo build_forest_info(const Tree &tree) {
    ForestInfo info{
        .children = tree_children(tree, CutHandling::EXCLUDE_CUT),
        .parent = tree.parent,
        .component_of_node = std::vector<int>(tree.vertices(), -1),
        .component_of_leaf = std::vector<int>(tree.leaves(), -1),
        .subtree_begin = std::vector<int>(tree.vertices(), -1),
        .subtree_end = std::vector<int>(tree.vertices(), -1),
        .component_order = std::vector<int>(tree.vertices(), -1),
        .leaf_position = std::vector<int>(tree.leaves(), -1),
        .leaf_order = {},
        .components = {},
        .component_nodes = {},
        .component_leaves_by_label = {},
        .choices = {},
    };

    const auto add_component = [&](this auto &&self, const int node, const int component, std::vector<int> &visited) -> void {
        info.component_order[node] = static_cast<int>(visited.size());
        visited.push_back(node);
        info.component_of_node[node] = component;
        info.subtree_begin[node] = static_cast<int>(info.leaf_order.size());
        if (node < tree.leaves()) {
            info.component_of_leaf[node] = component;
            info.leaf_position[node] = static_cast<int>(info.leaf_order.size());
            info.leaf_order.push_back(node);
            info.subtree_end[node] = static_cast<int>(info.leaf_order.size());
            return;
        }
        const auto [left, right] = info.children[node];
        if (left >= 0)
            self(left, component, visited);
        if (right >= 0)
            self(right, component, visited);
        info.subtree_end[node] = static_cast<int>(info.leaf_order.size());
    };

    for (int node = 0; node < tree.vertices(); ++node) {
        if (tree.parent[node] >= 0 && tree.edge_state[node] != EdgeState::CUT)
            continue;
        const int component = static_cast<int>(info.components.size());
        const int begin = static_cast<int>(info.leaf_order.size());
        std::vector<int> visited;
        add_component(node, component, visited);
        if (static_cast<int>(info.leaf_order.size()) == begin) {
            for (const int v : visited) {
                info.component_of_node[v] = -1;
                info.component_order[v] = -1;
            }
            continue;
        }
        info.components.push_back(ComponentRange{
            .root = node,
            .begin = begin,
            .end = static_cast<int>(info.leaf_order.size()),
        });
        info.component_nodes.emplace_back(std::move(visited));
        auto &leaves = info.component_leaves_by_label.emplace_back();
        leaves.reserve(info.components.back().end - info.components.back().begin);
        for (int pos = info.components.back().begin; pos < info.components.back().end; ++pos)
            leaves.push_back(info.leaf_order[pos]);
        std::ranges::sort(leaves);
    }

    if (static_cast<int>(info.leaf_order.size()) != tree.leaves())
        throw std::runtime_error("cut-aware cluster reduction lost leaves while building forest info");

    info.choices.push_back(PartialChoice{});
    std::unordered_set<ChoiceKey, ChoiceKeyHash> seen;
    seen.insert(ChoiceKey{});
    for (int node = 0; node < tree.vertices(); ++node) {
        const int component = info.component_of_node[node];
        if (component < 0)
            continue;
        const auto &range = info.components[component];
        const int begin = info.subtree_begin[node];
        const int end = info.subtree_end[node];
        const int subtree_size = end - begin;
        if (subtree_size <= 0)
            continue;

        if (node != range.root) {
            const ChoiceKey key{
                .kind = CutAwarePartialKind::BOTTOM,
                .component = component,
                .node = node,
                .intervals = {{{begin, end}, {0, 0}}},
                .interval_count = 1,
            };
            if (seen.insert(key).second) {
                info.choices.push_back(PartialChoice{
                    .kind = key.kind,
                    .component = component,
                    .node = node,
                    .leaf_count = subtree_size,
                    .intervals = key.intervals,
                    .interval_count = key.interval_count,
                });
            }
        }

        const int top_size = (range.end - range.begin) - subtree_size;
        if (top_size <= 0)
            continue;
        std::array<Interval, 2> intervals{};
        int interval_count = 0;
        if (range.begin < begin)
            intervals[interval_count++] = {range.begin, begin};
        if (end < range.end)
            intervals[interval_count++] = {end, range.end};
        const ChoiceKey key{
            .kind = CutAwarePartialKind::TOP,
            .component = component,
            .node = node,
            .intervals = intervals,
            .interval_count = interval_count,
        };
        if (seen.insert(key).second) {
            info.choices.push_back(PartialChoice{
                .kind = key.kind,
                .component = component,
                .node = node,
                .leaf_count = top_size,
                .intervals = key.intervals,
                .interval_count = key.interval_count,
            });
        }
    }

    std::ranges::stable_sort(
        info.choices.begin() + 1,
        info.choices.end(),
        [](const PartialChoice &lhs, const PartialChoice &rhs) {
            if (lhs.leaf_count != rhs.leaf_count)
                return lhs.leaf_count < rhs.leaf_count;
            if (lhs.kind != rhs.kind)
                return lhs.kind < rhs.kind;
            if (lhs.component != rhs.component)
                return lhs.component < rhs.component;
            return lhs.node < rhs.node;
        });
    return info;
}

void add_interval(std::vector<Interval> &intervals, const int begin, const int end) {
    if (begin < end)
        intervals.push_back({begin, end});
}

void normalize_intervals(std::vector<Interval> &intervals) {
    std::ranges::sort(intervals, [](const Interval lhs, const Interval rhs) {
        if (lhs.begin != rhs.begin)
            return lhs.begin < rhs.begin;
        return lhs.end < rhs.end;
    });
    int out = 0;
    for (const Interval interval : intervals) {
        if (out == 0 || intervals[out - 1].end < interval.begin) {
            intervals[out++] = interval;
        } else {
            intervals[out - 1].end = std::max(intervals[out - 1].end, interval.end);
        }
    }
    intervals.resize(out);
}

template <class F>
void for_each_representation_interval(
    const ForestInfo &info,
    const CutAwareRepresentation &representation,
    F &&visit) {
    for (const int component : representation.components) {
        const auto &range = info.components[component];
        visit(Interval{range.begin, range.end});
    }
    if (representation.partial_kind == CutAwarePartialKind::BOTTOM) {
        visit(Interval{
            info.subtree_begin[representation.partial_node],
            info.subtree_end[representation.partial_node]});
    } else if (representation.partial_kind == CutAwarePartialKind::TOP) {
        const auto &range = info.components[representation.partial_component];
        const int begin = info.subtree_begin[representation.partial_node];
        const int end = info.subtree_end[representation.partial_node];
        visit(Interval{range.begin, begin});
        visit(Interval{end, range.end});
    }
}

void append_representation_intervals(
    const ForestInfo &info,
    const CutAwareRepresentation &representation,
    std::vector<Interval> &intervals) {
    for_each_representation_interval(info, representation, [&](const Interval interval) {
        add_interval(intervals, interval.begin, interval.end);
    });
}

[[nodiscard]] int representation_leaf_count(
    const ForestInfo &info,
    const CutAwareRepresentation &representation) {
    int leaf_count = 0;
    for (const int component : representation.components)
        leaf_count += info.components[component].end - info.components[component].begin;
    if (representation.partial_kind == CutAwarePartialKind::NONE)
        return leaf_count;

    const int begin = info.subtree_begin[representation.partial_node];
    const int end = info.subtree_end[representation.partial_node];
    if (representation.partial_kind == CutAwarePartialKind::BOTTOM)
        return leaf_count + end - begin;
    const auto &range = info.components[representation.partial_component];
    return leaf_count + (range.end - range.begin) - (end - begin);
}

template <class F>
void for_each_representation_leaf(
    const ForestInfo &info,
    const CutAwareRepresentation &representation,
    F &&visit) {
    for_each_representation_interval(info, representation, [&](const Interval interval) {
        const auto [begin, end] = interval;
        for (int pos = begin; pos < end; ++pos)
            visit(info.leaf_order[pos]);
    });
}

void append_representation_leaves(
    const ForestInfo &info,
    const CutAwareRepresentation &representation,
    std::vector<int> &leaves) {
    for_each_representation_leaf(info, representation, [&](const int leaf) {
        leaves.push_back(leaf);
    });
}

[[nodiscard]] bool representation_matches_leaves(
    const ForestInfo &info,
    const CutAwareRepresentation &representation,
    const std::vector<int> &expected_leaves,
    std::vector<int> &mark,
    const int stamp) {
    if (representation_leaf_count(info, representation) != static_cast<int>(expected_leaves.size()))
        return false;
    if (static_cast<int>(mark.size()) != static_cast<int>(info.leaf_order.size()))
        mark.assign(info.leaf_order.size(), 0);

    int seen = 0;
    bool valid = true;
    for_each_representation_leaf(info, representation, [&](const int leaf) {
        if (!valid)
            return;
        if (leaf < 0 || leaf >= static_cast<int>(mark.size()) || mark[leaf] == stamp) {
            valid = false;
            return;
        }
        mark[leaf] = stamp;
        ++seen;
    });
    if (!valid || seen != static_cast<int>(expected_leaves.size()))
        return false;
    return std::ranges::all_of(expected_leaves, [&](const int leaf) {
        return leaf >= 0 && leaf < static_cast<int>(mark.size()) && mark[leaf] == stamp;
    });
}

struct SmallCandidate {
    std::vector<int> leaves;
    CutAwareRepresentation representation;
    int leaf_count = 0;
    std::uint64_t hash_a = 0;
    std::uint64_t hash_b = 0;
};

[[nodiscard]] std::uint64_t mix_leaf_hash(std::uint64_t hash, const int leaf) {
    const std::uint64_t value = static_cast<std::uint64_t>(leaf) + 0x9e3779b97f4a7c15ULL;
    hash ^= value + (hash << 6) + (hash >> 2);
    hash *= 0xbf58476d1ce4e5b9ULL;
    return hash ^ (hash >> 31);
}

void set_leaf_fingerprint(SmallCandidate &candidate) {
    candidate.leaf_count = static_cast<int>(candidate.leaves.size());
    std::uint64_t hash_a = 0x243f6a8885a308d3ULL;
    std::uint64_t hash_b = 0x13198a2e03707344ULL;
    for (const int leaf : candidate.leaves) {
        hash_a = mix_leaf_hash(hash_a, leaf);
        hash_b = mix_leaf_hash(hash_b, leaf ^ 0x5bd1e995);
    }
    candidate.hash_a = hash_a;
    candidate.hash_b = hash_b;
}

[[nodiscard]] int compare_leaf_set_key(const SmallCandidate &lhs, const SmallCandidate &rhs) {
    if (lhs.leaf_count != rhs.leaf_count)
        return lhs.leaf_count < rhs.leaf_count ? -1 : 1;
    if (lhs.hash_a != rhs.hash_a)
        return lhs.hash_a < rhs.hash_a ? -1 : 1;
    if (lhs.hash_b != rhs.hash_b)
        return lhs.hash_b < rhs.hash_b ? -1 : 1;
    if (lhs.leaves == rhs.leaves)
        return 0;
    return lhs.leaves < rhs.leaves ? -1 : 1;
}

[[nodiscard]] bool same_leaf_set(const SmallCandidate &lhs, const SmallCandidate &rhs) {
    return compare_leaf_set_key(lhs, rhs) == 0;
}

[[nodiscard]] int representation_rank(const CutAwareRepresentation &representation) {
    int rank = static_cast<int>(representation.partial_kind);
    rank *= 3;
    if (!representation.components.empty())
        rank = 0;
    return rank;
}

void add_small_candidate(
    const ForestInfo &info,
    CutAwareRepresentation representation,
    const int max_leaves,
    std::vector<SmallCandidate> &candidates) {
    std::vector<int> leaves;
    leaves.reserve(max_leaves);
    append_representation_leaves(info, representation, leaves);
    if (leaves.size() < 2 || static_cast<int>(leaves.size()) > max_leaves)
        return;
    std::ranges::sort(leaves);
    if (std::ranges::adjacent_find(leaves) != leaves.end())
        return;
    auto candidate = SmallCandidate{
        .leaves = std::move(leaves),
        .representation = std::move(representation),
    };
    set_leaf_fingerprint(candidate);
    candidates.push_back(std::move(candidate));
}

void add_small_candidate_variants(
    const ForestInfo &info,
    const CutAwareRepresentation &representation,
    const int leaf_count,
    const int max_leaves,
    std::vector<SmallCandidate> &candidates) {
    const int total_leaves = static_cast<int>(info.leaf_order.size());
    if (leaf_count >= 2 && leaf_count <= max_leaves)
        add_small_candidate(info, representation, max_leaves, candidates);
    const int complement_leaves = total_leaves - leaf_count;
    if (complement_leaves >= 2 && complement_leaves <= max_leaves)
        add_small_candidate(
            info,
            complement_representation(representation, info),
            max_leaves,
            candidates);
}

[[nodiscard]] std::vector<SmallCandidate> collect_small_candidates(
    const ForestInfo &info,
    const int max_leaves) {
    std::vector<SmallCandidate> candidates;
    for (int component = 0; component < static_cast<int>(info.components.size()); ++component) {
        const auto &range = info.components[component];
        const int leaf_count = range.end - range.begin;
        CutAwareRepresentation representation;
        representation.components.push_back(component);
        add_small_candidate_variants(info, representation, leaf_count, max_leaves, candidates);
    }
    for (int i = 1; i < static_cast<int>(info.choices.size()); ++i) {
        const auto &choice = info.choices[i];
        const CutAwareRepresentation representation{
            .partial_kind = choice.kind,
            .partial_component = choice.component,
            .partial_node = choice.node,
            .components = {},
        };
        add_small_candidate_variants(info, representation, choice.leaf_count, max_leaves, candidates);
    }

    std::ranges::sort(candidates, [](const SmallCandidate &lhs, const SmallCandidate &rhs) {
        const int key_cmp = compare_leaf_set_key(lhs, rhs);
        if (key_cmp != 0)
            return key_cmp < 0;
        return representation_rank(lhs.representation) < representation_rank(rhs.representation);
    });
    candidates.erase(
        std::unique(candidates.begin(), candidates.end(), [](const SmallCandidate &lhs, const SmallCandidate &rhs) {
            return same_leaf_set(lhs, rhs);
        }),
        candidates.end());
    return candidates;
}

[[nodiscard]] std::vector<SmallCandidate> collect_full_component_candidates(
    const ForestInfo &info,
    const int max_leaves) {
    std::vector<SmallCandidate> candidates;
    for (int component = 0; component < static_cast<int>(info.components.size()); ++component) {
        const auto &range = info.components[component];
        const int leaf_count = range.end - range.begin;
        if (leaf_count < 2 || leaf_count > max_leaves)
            continue;
        CutAwareRepresentation representation;
        representation.components.push_back(component);
        auto candidate = SmallCandidate{
            .leaves = info.component_leaves_by_label[component],
            .representation = std::move(representation),
        };
        set_leaf_fingerprint(candidate);
        candidates.push_back(std::move(candidate));
    }
    std::ranges::sort(candidates, [](const SmallCandidate &lhs, const SmallCandidate &rhs) {
        return compare_leaf_set_key(lhs, rhs) < 0;
    });
    return candidates;
}

[[nodiscard]] bool better_small_cluster_size(
    const int candidate,
    const int best,
    const int target) {
    const bool candidate_large_enough = candidate >= target;
    const bool best_large_enough = best >= target;
    if (candidate_large_enough != best_large_enough)
        return candidate_large_enough;
    return candidate_large_enough ? candidate < best : candidate > best;
}

[[nodiscard]] int node_with_leaf_interval(
    const ForestInfo &info,
    const int component,
    const int begin,
    const int end) {
    const auto &range = info.components[component];
    if (begin == range.begin && end == range.end)
        return range.root;
    int node = info.leaf_order[begin];
    while (node >= 0 && info.component_of_node[node] == component) {
        if (info.subtree_begin[node] == begin && info.subtree_end[node] == end)
            return node;
        if (info.subtree_begin[node] <= begin && info.subtree_end[node] >= end)
            return -1;
        node = info.parent[node];
    }
    return -1;
}

[[nodiscard]] std::optional<CutAwareRepresentation> representation_for_leaf_set(
    const ForestInfo &info,
    const std::vector<int> &leaves) {
    CutAwareRepresentation representation;
    CutAwareRepresentation partial;
    bool used_partial = false;

    std::vector<int> components;
    components.reserve(leaves.size());
    for (const int leaf : leaves) {
        const int component = info.component_of_leaf[leaf];
        if (component < 0)
            return std::nullopt;
        components.push_back(component);
    }
    std::ranges::sort(components);
    components.erase(std::unique(components.begin(), components.end()), components.end());

    std::vector<int> positions;
    positions.reserve(leaves.size());
    for (const int component : components) {
        positions.clear();
        for (const int leaf : leaves) {
            if (info.component_of_leaf[leaf] == component)
                positions.push_back(info.leaf_position[leaf]);
        }
        std::ranges::sort(positions);
        const int selected = static_cast<int>(positions.size());
        const auto &range = info.components[component];
        const int whole = range.end - range.begin;
        if (selected == whole) {
            representation.components.push_back(component);
            continue;
        }
        if (used_partial)
            return std::nullopt;

        if (positions.back() - positions.front() + 1 == selected) {
            const int node = node_with_leaf_interval(
                info,
                component,
                positions.front(),
                positions.back() + 1);
            if (node >= 0 && node != range.root) {
                partial = CutAwareRepresentation{
                    .partial_kind = CutAwarePartialKind::BOTTOM,
                    .partial_component = component,
                    .partial_node = node,
                    .components = {},
                };
                used_partial = true;
                continue;
            }
        }

        int prefix = 0;
        while (prefix < selected && positions[prefix] == range.begin + prefix)
            ++prefix;
        int suffix = 0;
        while (suffix < selected - prefix &&
               positions[selected - 1 - suffix] == range.end - 1 - suffix)
            ++suffix;
        if (prefix + suffix == selected) {
            const int omitted_begin = range.begin + prefix;
            const int omitted_end = range.end - suffix;
            const int node = node_with_leaf_interval(info, component, omitted_begin, omitted_end);
            if (node >= 0 && node != range.root) {
                partial = CutAwareRepresentation{
                    .partial_kind = CutAwarePartialKind::TOP,
                    .partial_component = component,
                    .partial_node = node,
                    .components = {},
                };
                used_partial = true;
                continue;
            }
        }
        return std::nullopt;
    }

    if (used_partial) {
        representation.partial_kind = partial.partial_kind;
        representation.partial_component = partial.partial_component;
        representation.partial_node = partial.partial_node;
    }
    return representation;
}

[[nodiscard]] std::optional<CutAwareCluster> find_small_direct_cluster(
    const std::array<ForestInfo, 2> &info,
    const int target_leaves) {
    const std::vector<SmallCandidate> first = collect_small_candidates(info[0], target_leaves);
    const std::vector<SmallCandidate> second = collect_small_candidates(info[1], target_leaves);
    std::optional<CutAwareCluster> best;
    const auto consider = [&](const SmallCandidate &candidate,
                              CutAwareRepresentation first_representation,
                              CutAwareRepresentation second_representation) {
        const int size = candidate.leaf_count;
        if (best.has_value() && !better_small_cluster_size(size, static_cast<int>(best->leaves.size()), target_leaves))
            return;
        best = CutAwareCluster{
            .leaves = candidate.leaves,
            .representation = {std::move(first_representation), std::move(second_representation)},
        };
    };
    int i = 0;
    int j = 0;
    while (i < static_cast<int>(first.size()) && j < static_cast<int>(second.size())) {
        const int key_cmp = compare_leaf_set_key(first[i], second[j]);
        if (key_cmp < 0) {
            ++i;
            continue;
        }
        if (key_cmp > 0) {
            ++j;
            continue;
        }
        const int size = first[i].leaf_count;
        if (!best.has_value() || better_small_cluster_size(size, static_cast<int>(best->leaves.size()), target_leaves)) {
            consider(first[i], first[i].representation, second[j].representation);
            if (size == target_leaves)
                return best;
        }
        ++i;
        ++j;
    }
    for (const SmallCandidate &candidate : first) {
        if (best.has_value() &&
            !better_small_cluster_size(candidate.leaf_count, static_cast<int>(best->leaves.size()), target_leaves))
            continue;
        if (auto representation = representation_for_leaf_set(info[1], candidate.leaves)) {
            consider(candidate, candidate.representation, std::move(*representation));
            if (candidate.leaf_count == target_leaves)
                return best;
        }
    }
    for (const SmallCandidate &candidate : second) {
        if (best.has_value() &&
            !better_small_cluster_size(candidate.leaf_count, static_cast<int>(best->leaves.size()), target_leaves))
            continue;
        if (auto representation = representation_for_leaf_set(info[0], candidate.leaves)) {
            consider(candidate, std::move(*representation), candidate.representation);
            if (candidate.leaf_count == target_leaves)
                return best;
        }
    }
    return best;
}

[[nodiscard]] bool full_component_cluster(const CutAwareCluster &cluster) {
    return std::ranges::all_of(cluster.representation, [](const CutAwareRepresentation &representation) {
        return representation.partial_kind == CutAwarePartialKind::NONE &&
               representation.components.size() == 1;
    });
}

struct FullComponentMatch {
    std::vector<int> leaves;
    std::array<int, 2> component{-1, -1};
};

[[nodiscard]] std::vector<FullComponentMatch> find_full_component_matches(
    const std::array<ForestInfo, 2> &info,
    const int max_leaves) {
    const std::vector<SmallCandidate> first = collect_full_component_candidates(info[0], max_leaves);
    const std::vector<SmallCandidate> second = collect_full_component_candidates(info[1], max_leaves);

    std::vector<FullComponentMatch> matches;
    int i = 0;
    int j = 0;
    while (i < static_cast<int>(first.size()) && j < static_cast<int>(second.size())) {
        const int key_cmp = compare_leaf_set_key(first[i], second[j]);
        if (key_cmp < 0) {
            ++i;
            continue;
        }
        if (key_cmp > 0) {
            ++j;
            continue;
        }
        matches.push_back(FullComponentMatch{
            .leaves = first[i].leaves,
            .component = {first[i].representation.components.front(), second[j].representation.components.front()},
        });
        ++i;
        ++j;
    }
    return matches;
}

class ClosureSearch {
  public:
    explicit ClosureSearch(
        const AnnotatedInstance &instance,
        const SolveContext *context = nullptr,
        const std::optional<int> preferred_seed_side_leaves = std::nullopt)
        : instance_(instance),
          context_(context),
          preferred_seed_side_leaves_(preferred_seed_side_leaves),
          search_start_(Clock::now()),
          total_leaves_(instance.trees.front().leaves()),
          info_{build_forest_info(instance.trees[0]), build_forest_info(instance.trees[1])},
          selected_{std::vector<char>(total_leaves_, 0), std::vector<char>(total_leaves_, 0)},
          component_touched_{
              std::vector<int>(static_cast<int>(info_[0].components.size()), 0),
              std::vector<int>(static_cast<int>(info_[1].components.size()), 0)},
          first_leaf_{
              std::vector<int>(static_cast<int>(info_[0].components.size()), -1),
              std::vector<int>(static_cast<int>(info_[1].components.size()), -1)},
          node_touched_{
              std::vector<int>(instance.trees[0].vertices(), 0),
              std::vector<int>(instance.trees[1].vertices(), 0)},
          best_zero_node_{
              std::vector<int>(static_cast<int>(info_[0].components.size()), -1),
              std::vector<int>(static_cast<int>(info_[1].components.size()), -1)},
          best_zero_size_{
              std::vector<int>(static_cast<int>(info_[0].components.size()), 0),
              std::vector<int>(static_cast<int>(info_[1].components.size()), 0)} {
        for (int side = 0; side < 2; ++side) {
            selected_leaves_[side].reserve(total_leaves_);
            touched_nodes_[side].reserve(instance_.trees[side].vertices());
            touched_components_[side].reserve(info_[side].components.size());
        }
    }

    [[nodiscard]] std::optional<CutAwareCluster> best_cluster() {
        check_timeout();
        consider_whole_component_clusters();
        if (has_optimal_target_cluster())
            return finalize_best();
        const int seed_side =
            static_cast<int>(info_[0].components.size() + info_[0].choices.size()) <=
                    static_cast<int>(info_[1].components.size() + info_[1].choices.size())
                ? 0
                : 1;
        const int preferred_seed_limit = cut_aware_seed_limit();
        const std::vector<Seed> seeds = ordered_seeds(seed_side, preferred_seed_limit);
        std::vector<char> skip_seed(seeds.size(), 0);
        const auto scan_seeds = [&](const bool preferred) {
            for (int i = 0; i < static_cast<int>(seeds.size()); ++i) {
                check_timeout();
                if (search_budget_exhausted())
                    return true;
                const Seed &seed = seeds[i];
                if (skip_seed[i])
                    continue;
                if (seed.lower_bound < 2 || seed.lower_bound > total_leaves_ - 2)
                    continue;
                if (pruned_by_best(seed.lower_bound))
                    continue;
                if (seed.preferred != preferred)
                    continue;
                clear_selected();
                set_selected_seed(seed_side, seed);
                const auto cluster = close_from_seed(seed_side);
                if (cluster.has_value()) {
                    mark_seeds_dominated_by_closed_cluster(seed_side, seeds, skip_seed, i, *cluster);
                    if (preferred && preferred_seed_side_leaves_.has_value())
                        return true;
                }
                if (has_optimal_target_cluster())
                    return true;
            }
            return false;
        };
        if (!scan_seeds(true) && !best_.has_value())
            scan_seeds(false);
        return finalize_best();
    }

    [[nodiscard]] std::optional<CutAwareCluster> best_small_direct_cluster() const {
        check_timeout();
        return find_small_direct_cluster(info_, target_cluster_leaves());
    }

    [[nodiscard]] std::array<ForestInfo, 2> take_info() && {
        return std::move(info_);
    }

  private:
    struct Completion {
        std::vector<int> leaves;
        int leaf_count = 0;
        bool inverted = false;
        CutAwareRepresentation representation;
        CutAwareRepresentation inverse_representation;
    };

    enum class SeedKind : std::uint8_t {
        COMPONENT,
        CHOICE,
    };

    struct Seed {
        SeedKind kind = SeedKind::COMPONENT;
        int index = -1;
        int lower_bound = 0;
        std::array<Interval, 2> intervals{};
        int interval_count = 0;
        bool preferred = false;
    };

    const AnnotatedInstance &instance_;
    const SolveContext *context_ = nullptr;
    std::optional<int> preferred_seed_side_leaves_;
    Clock::time_point search_start_;
    int total_leaves_ = 0;
    std::array<ForestInfo, 2> info_;
    std::optional<CutAwareCluster> best_;
    std::array<std::vector<char>, 2> selected_;
    std::array<std::vector<int>, 2> selected_leaves_;
    std::array<bool, 2> selected_inverted_{false, false};
    std::array<std::vector<int>, 2> component_touched_;
    std::array<std::vector<int>, 2> first_leaf_;
    std::array<std::vector<int>, 2> node_touched_;
    std::array<std::vector<int>, 2> touched_components_;
    std::array<std::vector<int>, 2> touched_nodes_;
    std::array<std::vector<int>, 2> best_zero_node_;
    std::array<std::vector<int>, 2> best_zero_size_;
    std::array<bool, 2> dense_work_dirty_{false, false};
    mutable std::vector<int> representation_mark_;
    mutable int representation_stamp_ = 0;
    [[maybe_unused]] mutable int budget_check_skip_ = 0;
    [[maybe_unused]] mutable bool budget_exhausted_ = false;

    [[nodiscard]] int target_cluster_leaves() const {
        return cut_aware_target_leaves(total_leaves_);
    }

    [[nodiscard]] int cut_aware_seed_limit() const {
        int limit = preferred_seed_side_leaves_.value_or(constants::cut_aware_preferred_seed_side_limit);
        return std::min(limit, total_leaves_ - 2);
    }

    [[nodiscard]] bool preferred_seed_size(const int leaf_count, const int preferred_seed_limit) const {
        return leaf_count <= preferred_seed_limit || total_leaves_ - leaf_count <= preferred_seed_limit;
    }

    void check_timeout() const {
        if (context_ != nullptr)
            context_->check_timeout();
    }

    [[nodiscard]] bool search_budget_exhausted() const {
#if MAFFE_COMPETITION_HEURISTIC_BUILD
        if (budget_exhausted_)
            return true;
        if (budget_check_skip_ > 0) {
            --budget_check_skip_;
            return false;
        }
        budget_check_skip_ = constants::competition_cut_aware_budget_check_interval - 1;
        budget_exhausted_ = std::chrono::duration<double>(Clock::now() - search_start_).count() >=
            constants::competition_cut_aware_search_seconds;
        return budget_exhausted_;
#else
        return false;
#endif
    }

    [[nodiscard]] bool better_size(const int candidate, const int best) const {
        const int target = target_cluster_leaves();
        const bool candidate_large_enough = candidate >= target;
        const bool best_large_enough = best >= target;
        if (candidate_large_enough != best_large_enough)
            return candidate_large_enough;
        return candidate_large_enough ? candidate < best : candidate > best;
    }

    [[nodiscard]] bool pruned_by_best(const int lower_bound) const {
        return best_.has_value() &&
               static_cast<int>(best_->leaves.size()) >= target_cluster_leaves() &&
               lower_bound >= static_cast<int>(best_->leaves.size());
    }

    [[nodiscard]] bool has_optimal_target_cluster() const {
        return best_.has_value() && static_cast<int>(best_->leaves.size()) == target_cluster_leaves();
    }

    [[nodiscard]] std::optional<CutAwareCluster> finalize_best() {
        if (!best_.has_value())
            return std::nullopt;
        std::ranges::sort(best_->leaves);
        return best_;
    }

    [[nodiscard]] std::vector<Seed> ordered_seeds(
        const int side,
        const int preferred_seed_limit) const {
        std::vector<Seed> seeds;
        seeds.reserve(info_[side].components.size() + info_[side].choices.size());

        std::vector<int> components(info_[side].components.size());
        std::iota(components.begin(), components.end(), 0);
        std::ranges::sort(components, [&](const int lhs, const int rhs) {
            const auto &a = info_[side].components[lhs];
            const auto &b = info_[side].components[rhs];
            return a.end - a.begin < b.end - b.begin;
        });
        for (const int component : components) {
            const auto &range = info_[side].components[component];
            const int leaf_count = range.end - range.begin;
            const bool preferred = preferred_seed_size(leaf_count, preferred_seed_limit);
            seeds.push_back(Seed{
                .kind = SeedKind::COMPONENT,
                .index = component,
                .lower_bound = leaf_count,
                .intervals = {{{range.begin, range.end}, {0, 0}}},
                .interval_count = 1,
                .preferred = preferred,
            });
        }

        for (int i = 1; i < static_cast<int>(info_[side].choices.size()); ++i) {
            const auto &choice = info_[side].choices[i];
            const bool preferred = preferred_seed_size(choice.leaf_count, preferred_seed_limit);
            seeds.push_back(Seed{
                .kind = SeedKind::CHOICE,
                .index = i,
                .lower_bound = choice.leaf_count,
                .intervals = choice.intervals,
                .interval_count = choice.interval_count,
                .preferred = preferred,
            });
        }
        if (preferred_seed_side_leaves_.has_value()) {
            std::ranges::stable_sort(seeds, [&](const Seed &lhs, const Seed &rhs) {
                const int lhs_side = std::min(lhs.lower_bound, total_leaves_ - lhs.lower_bound);
                const int rhs_side = std::min(rhs.lower_bound, total_leaves_ - rhs.lower_bound);
                if (lhs_side != rhs_side)
                    return lhs_side < rhs_side;
                return lhs.lower_bound < rhs.lower_bound;
            });
        }
        return seeds;
    }

    void set_selected_seed(const int side, const Seed &seed) {
        if (seed.kind == SeedKind::COMPONENT) {
            set_selected_component(side, seed.index);
        } else {
            set_selected_choice(side, info_[side].choices[seed.index]);
        }
    }

    [[nodiscard]] CutAwareCluster complement_cluster(const CutAwareCluster &cluster) const {
        std::vector<char> in_cluster(total_leaves_, 0);
        for (const int leaf : cluster.leaves)
            in_cluster[leaf] = 1;

        CutAwareCluster complement;
        complement.leaves.reserve(total_leaves_ - static_cast<int>(cluster.leaves.size()));
        for (int leaf = 0; leaf < total_leaves_; ++leaf) {
            if (!in_cluster[leaf])
                complement.leaves.push_back(leaf);
        }
        complement.representation = {
            complement_representation(cluster.representation[0], info_[0]),
            complement_representation(cluster.representation[1], info_[1]),
        };
        return complement;
    }

    [[nodiscard]] bool representation_matches_leaves(
        const int side,
        const CutAwareRepresentation &representation,
        const std::vector<int> &leaves) const {
        if (++representation_stamp_ == std::numeric_limits<int>::max()) {
            std::ranges::fill(representation_mark_, 0);
            representation_stamp_ = 1;
        }
        return maffe::detail::representation_matches_leaves(
            info_[side],
            representation,
            leaves,
            representation_mark_,
            representation_stamp_);
    }

    [[nodiscard]] bool cluster_representation_matches_leaves(const CutAwareCluster &cluster) const {
        return representation_matches_leaves(0, cluster.representation[0], cluster.leaves) &&
            representation_matches_leaves(1, cluster.representation[1], cluster.leaves);
    }

    [[nodiscard]] static bool intervals_cover(
        const std::array<Interval, 2> &cover,
        const int cover_count,
        const Interval target) {
        int cursor = target.begin;
        for (int i = 0; i < cover_count && cursor < target.end; ++i) {
            if (cover[i].end <= cursor)
                continue;
            if (cover[i].begin > cursor)
                return false;
            cursor = std::max(cursor, cover[i].end);
        }
        return cursor >= target.end;
    }

    [[nodiscard]] static bool intervals_cover(
        const std::vector<Interval> &cover,
        const Interval target) {
        int cursor = target.begin;
        for (const auto interval : cover) {
            if (interval.end <= cursor)
                continue;
            if (interval.begin > cursor)
                return false;
            cursor = std::max(cursor, interval.end);
            if (cursor >= target.end)
                return true;
        }
        return cursor >= target.end;
    }

    [[nodiscard]] static bool seed_is_covered_by_intervals(
        const Seed &seed,
        const std::vector<Interval> &cover) {
        for (int i = 0; i < seed.interval_count; ++i) {
            if (!intervals_cover(cover, seed.intervals[i]))
                return false;
        }
        return true;
    }

    [[nodiscard]] static bool seed_contains_seed(const Seed &container, const Seed &contained) {
        for (int i = 0; i < contained.interval_count; ++i) {
            if (!intervals_cover(container.intervals, container.interval_count, contained.intervals[i]))
                return false;
        }
        return true;
    }

    void mark_seeds_dominated_by_closed_cluster(
        const int seed_side,
        const std::vector<Seed> &seeds,
        std::vector<char> &skip_seed,
        const int current_seed,
        const CutAwareCluster &cluster) const {
        const int cluster_size = static_cast<int>(cluster.leaves.size());
        if (cluster_size == target_cluster_leaves())
            return;

        std::vector<Interval> coverage;
        coverage.reserve(cluster.representation[seed_side].components.size() + 2);
        append_representation_intervals(info_[seed_side], cluster.representation[seed_side], coverage);
        normalize_intervals(coverage);
        const bool current_seed_is_in_cluster = seed_is_covered_by_intervals(seeds[current_seed], coverage);
        for (int i = 0; i < static_cast<int>(seeds.size()); ++i) {
            if (skip_seed[i])
                continue;
            if (!seed_is_covered_by_intervals(seeds[i], coverage))
                continue;

            // Closed clusters induce seed implications. Below target, every seed inside the cluster is dominated.
            // Above target, a seed between the current seed and its closure is in the same closure SCC.
            if (cluster_size < target_cluster_leaves() ||
                (current_seed_is_in_cluster && seed_contains_seed(seeds[i], seeds[current_seed]))) {
                skip_seed[i] = 1;
            }
        }
    }

    [[nodiscard]] bool consider_cluster(CutAwareCluster cluster) {
        const int size = static_cast<int>(cluster.leaves.size());
        if (size < 2 || size > total_leaves_ - 2)
            return false;
        if (!cluster_representation_matches_leaves(cluster))
            return false;
        if (!best_.has_value() || better_size(size, static_cast<int>(best_->leaves.size())))
            best_ = std::move(cluster);
        return true;
    }

    void clear_selected() {
        clear_selected_side(0);
        clear_selected_side(1);
    }

    void clear_selected_side(const int side) {
        for (const int leaf : selected_leaves_[side])
            selected_[side][leaf] = 0;
        selected_leaves_[side].clear();
        selected_inverted_[side] = false;
    }

    void add_marked_leaf(const int side, const int leaf) {
        if (selected_[side][leaf])
            return;
        selected_[side][leaf] = 1;
        selected_leaves_[side].push_back(leaf);
    }

    [[nodiscard]] bool is_selected(const int side, const int leaf) const {
        return selected_inverted_[side] ? !selected_[side][leaf] : selected_[side][leaf];
    }

    [[nodiscard]] int selected_leaf_count(const int side) const {
        const int marked = static_cast<int>(selected_leaves_[side].size());
        return selected_inverted_[side] ? total_leaves_ - marked : marked;
    }

    void add_leaf_order_interval_marks(
        const int side,
        const std::vector<int> &leaf_order,
        const int begin,
        const int end) {
        for (int pos = begin; pos < end; ++pos)
            add_marked_leaf(side, leaf_order[pos]);
    }

    void set_selected_intervals(
        const int side,
        const std::vector<int> &leaf_order,
        const std::array<Interval, 2> &intervals,
        const int interval_count,
        const int leaf_count) {
        clear_selected_side(side);
        if (leaf_count <= total_leaves_ / 2) {
            selected_inverted_[side] = false;
            selected_leaves_[side].reserve(leaf_count);
            for (int i = 0; i < interval_count; ++i)
                add_leaf_order_interval_marks(side, leaf_order, intervals[i].begin, intervals[i].end);
            return;
        }

        selected_inverted_[side] = true;
        selected_leaves_[side].reserve(total_leaves_ - leaf_count);
        int cursor = 0;
        for (int i = 0; i < interval_count; ++i) {
            add_leaf_order_interval_marks(side, leaf_order, cursor, intervals[i].begin);
            cursor = intervals[i].end;
        }
        add_leaf_order_interval_marks(side, leaf_order, cursor, total_leaves_);
    }

    void set_selected_component(const int side, const int component) {
        const auto &range = info_[side].components[component];
        set_selected_intervals(
            side,
            info_[side].leaf_order,
            {{{range.begin, range.end}, {0, 0}}},
            1,
            range.end - range.begin);
    }

    void set_selected_choice(const int side, const PartialChoice &choice) {
        set_selected_intervals(
            side,
            info_[side].leaf_order,
            choice.intervals,
            choice.interval_count,
            choice.leaf_count);
    }

    void set_selected_completion(const int side, const Completion &completion) {
        clear_selected_side(side);
        selected_inverted_[side] = completion.inverted;
        selected_leaves_[side].reserve(completion.leaves.size());
        for (const int leaf : completion.leaves)
            add_marked_leaf(side, leaf);
    }

    void consider_whole_component_clusters() {
        const int comp0 = static_cast<int>(info_[0].components.size());
        const int comp1 = static_cast<int>(info_[1].components.size());
        std::vector<std::vector<int>> adj0(comp0);
        std::vector<std::vector<int>> adj1(comp1);
        std::vector<std::pair<int, int>> edges;
        edges.reserve(total_leaves_);
        for (int leaf = 0; leaf < total_leaves_; ++leaf) {
            edges.emplace_back(info_[0].component_of_leaf[leaf], info_[1].component_of_leaf[leaf]);
        }
        std::ranges::sort(edges);
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        for (const auto [a, b] : edges) {
            adj0[a].push_back(b);
            adj1[b].push_back(a);
        }

        std::vector<char> seen0(comp0, 0);
        std::vector<char> seen1(comp1, 0);
        for (int start = 0; start < comp0; ++start) {
            if (seen0[start])
                continue;
            std::vector<int> comps0;
            std::vector<int> comps1;
            std::vector<std::pair<int, int>> queue = {{0, start}};
            seen0[start] = 1;
            for (std::size_t head = 0; head < queue.size(); ++head) {
                const auto [side, id] = queue[head];
                if (side == 0) {
                    comps0.push_back(id);
                    for (const int next : adj0[id]) {
                        if (!seen1[next]) {
                            seen1[next] = 1;
                            queue.emplace_back(1, next);
                        }
                    }
                } else {
                    comps1.push_back(id);
                    for (const int next : adj1[id]) {
                        if (!seen0[next]) {
                            seen0[next] = 1;
                            queue.emplace_back(0, next);
                        }
                    }
                }
            }
            int leaf_count = 0;
            for (const int component : comps0)
                leaf_count += info_[0].components[component].end - info_[0].components[component].begin;
            CutAwareCluster cluster;
            cluster.representation[0].components = comps0;
            cluster.representation[1].components = comps1;
            cluster.leaves.reserve(leaf_count);
            for (const int component : comps0) {
                const auto &range = info_[0].components[component];
                for (int pos = range.begin; pos < range.end; ++pos)
                    cluster.leaves.push_back(info_[0].leaf_order[pos]);
            }
            static_cast<void>(consider_cluster(std::move(cluster)));
            if (has_optimal_target_cluster())
                break;
        }
    }

    void reset_sparse_work(const int side) {
        if (dense_work_dirty_[side]) {
            std::fill(node_touched_[side].begin(), node_touched_[side].end(), 0);
            std::fill(component_touched_[side].begin(), component_touched_[side].end(), 0);
            std::fill(first_leaf_[side].begin(), first_leaf_[side].end(), -1);
            std::fill(best_zero_node_[side].begin(), best_zero_node_[side].end(), -1);
            std::fill(best_zero_size_[side].begin(), best_zero_size_[side].end(), 0);
            dense_work_dirty_[side] = false;
        } else {
            for (const int node : touched_nodes_[side])
                node_touched_[side][node] = 0;
            for (const int component : touched_components_[side]) {
                component_touched_[side][component] = 0;
                first_leaf_[side][component] = -1;
                best_zero_node_[side][component] = -1;
                best_zero_size_[side][component] = 0;
            }
        }
        touched_nodes_[side].clear();
        touched_components_[side].clear();
    }

    [[nodiscard]] int representation_leaf_count(
        const int side,
        const CutAwareRepresentation &representation) const {
        return maffe::detail::representation_leaf_count(info_[side], representation);
    }

    void append_representation_leaves(
        const int side,
        const CutAwareRepresentation &representation,
        std::vector<int> &leaves) const {
        maffe::detail::append_representation_leaves(info_[side], representation, leaves);
    }

    [[nodiscard]] Completion finish_completion_from_counts(
        const int side,
        const std::vector<int> &touched_components,
        const bool use_cached_zero_subtree,
        const bool node_counts_are_unselected = false) const {
        const auto &node_touched = node_touched_[side];
        const auto &component_touched = component_touched_[side];
        const auto &first_leaf = first_leaf_[side];
        const auto selected_count_at = [&](const int node) {
            if (!node_counts_are_unselected)
                return node_touched[node];
            return info_[side].subtree_end[node] - info_[side].subtree_begin[node] - node_touched[node];
        };

        Completion completion;
        int best_component = -1;
        int best_saving = 0;
        CutAwareRepresentation best_partial;
        for (const int component : touched_components) {
            const int whole = info_[side].components[component].end - info_[side].components[component].begin;
            if (component_touched[component] == whole) {
                completion.representation.components.push_back(component);
                continue;
            }

            int bottom_node = first_leaf[component];
            while (selected_count_at(bottom_node) < component_touched[component]) {
                const int parent = instance_.trees[side].parent[bottom_node];
                if (parent < 0 || info_[side].component_of_node[parent] != component)
                    break;
                bottom_node = parent;
            }
            CutAwareRepresentation partial{
                .partial_kind = CutAwarePartialKind::BOTTOM,
                .partial_component = component,
                .partial_node = bottom_node,
                .components = {},
            };
            int partial_size = info_[side].subtree_end[bottom_node] - info_[side].subtree_begin[bottom_node];

            int best_zero_node = -1;
            int best_zero_size = 0;
            if (use_cached_zero_subtree) {
                best_zero_node = best_zero_node_[side][component];
                best_zero_size = best_zero_size_[side][component];
            } else {
                for (const int node : info_[side].component_nodes[component]) {
                    if (selected_count_at(node) != 0)
                        continue;
                    const int size = info_[side].subtree_end[node] - info_[side].subtree_begin[node];
                    if (size > best_zero_size) {
                        best_zero_size = size;
                        best_zero_node = node;
                    }
                }
            }
            if (best_zero_size > 0) {
                const int top_size = whole - best_zero_size;
                if (top_size < partial_size) {
                    partial = CutAwareRepresentation{
                        .partial_kind = CutAwarePartialKind::TOP,
                        .partial_component = component,
                        .partial_node = best_zero_node,
                        .components = {},
                    };
                    partial_size = top_size;
                }
            }

            const int saving = whole - partial_size;
            if (saving > best_saving) {
                best_saving = saving;
                best_component = component;
                best_partial = partial;
            }
            completion.representation.components.push_back(component);
        }

        if (best_component >= 0) {
            completion.representation.components.erase(
                std::ranges::find(completion.representation.components, best_component));
            completion.representation.partial_kind = best_partial.partial_kind;
            completion.representation.partial_component = best_partial.partial_component;
            completion.representation.partial_node = best_partial.partial_node;
        }

        completion.leaf_count = representation_leaf_count(side, completion.representation);
        if (2 * completion.leaf_count <= total_leaves_) {
            append_representation_leaves(side, completion.representation, completion.leaves);
        } else {
            completion.inverted = true;
            completion.inverse_representation =
                complement_representation(completion.representation, info_[side]);
            append_representation_leaves(side, completion.inverse_representation, completion.leaves);
        }
        return completion;
    }

    [[nodiscard]] Completion finish_inverted_completion_from_counts(
        const int side,
        const std::vector<int> &touched_components,
        const bool use_cached_zero_subtree) const {
        const auto &unselected_node_count = node_touched_[side];
        const auto &unselected_component_count = component_touched_[side];
        const auto &first_leaf = first_leaf_[side];
        const auto selected_count_at = [&](const int node) {
            return info_[side].subtree_end[node] - info_[side].subtree_begin[node] - unselected_node_count[node];
        };

        Completion completion;
        completion.inverted = true;
        int best_component = -1;
        int best_saving = 0;
        int best_first_leaf = total_leaves_;
        CutAwareRepresentation best_partial;
        for (const int component : touched_components) {
            const int whole = info_[side].components[component].end - info_[side].components[component].begin;
            const int selected = whole - unselected_component_count[component];
            if (selected <= 0) {
                completion.inverse_representation.components.push_back(component);
                continue;
            }

            int bottom_node = first_leaf[component];
            while (selected_count_at(bottom_node) < selected) {
                const int parent = instance_.trees[side].parent[bottom_node];
                if (parent < 0 || info_[side].component_of_node[parent] != component)
                    break;
                bottom_node = parent;
            }
            CutAwareRepresentation partial{
                .partial_kind = CutAwarePartialKind::BOTTOM,
                .partial_component = component,
                .partial_node = bottom_node,
                .components = {},
            };
            int partial_size = info_[side].subtree_end[bottom_node] - info_[side].subtree_begin[bottom_node];

            int best_zero_node = -1;
            int best_zero_size = 0;
            if (use_cached_zero_subtree) {
                best_zero_node = best_zero_node_[side][component];
                best_zero_size = best_zero_size_[side][component];
            } else {
                for (const int node : info_[side].component_nodes[component]) {
                    if (selected_count_at(node) != 0)
                        continue;
                    const int size = info_[side].subtree_end[node] - info_[side].subtree_begin[node];
                    if (size > best_zero_size) {
                        best_zero_size = size;
                        best_zero_node = node;
                    }
                }
            }
            if (best_zero_size > 0) {
                const int top_size = whole - best_zero_size;
                if (top_size < partial_size) {
                    partial = CutAwareRepresentation{
                        .partial_kind = CutAwarePartialKind::TOP,
                        .partial_component = component,
                        .partial_node = best_zero_node,
                        .components = {},
                    };
                    partial_size = top_size;
                }
            }

            const int saving = whole - partial_size;
            if (saving > best_saving ||
                (saving == best_saving && first_leaf[component] < best_first_leaf)) {
                best_saving = saving;
                best_component = component;
                best_first_leaf = first_leaf[component];
                best_partial = partial;
            }
        }

        if (best_component >= 0) {
            completion.inverse_representation.partial_kind =
                best_partial.partial_kind == CutAwarePartialKind::BOTTOM
                    ? CutAwarePartialKind::TOP
                    : CutAwarePartialKind::BOTTOM;
            completion.inverse_representation.partial_component = best_partial.partial_component;
            completion.inverse_representation.partial_node = best_partial.partial_node;
        }
        completion.leaf_count =
            total_leaves_ - representation_leaf_count(side, completion.inverse_representation);
        append_representation_leaves(side, completion.inverse_representation, completion.leaves);
        return completion;
    }

    [[nodiscard]] Completion complete_side_dense(const int side) {
        auto &node_touched = node_touched_[side];
        std::fill(node_touched.begin(), node_touched.end(), 0);
        auto &component_touched = component_touched_[side];
        std::fill(component_touched.begin(), component_touched.end(), 0);
        auto &first_leaf = first_leaf_[side];
        std::fill(first_leaf.begin(), first_leaf.end(), -1);
        std::vector<int> touched_components;
        touched_components.reserve(info_[side].components.size());

        for (int leaf = 0; leaf < total_leaves_; ++leaf) {
            if (!is_selected(side, leaf))
                continue;
            node_touched[leaf] = 1;
            const int component = info_[side].component_of_leaf[leaf];
            if (component_touched[component]++ == 0) {
                touched_components.push_back(component);
                first_leaf[component] = leaf;
            } else if (leaf < first_leaf[component]) {
                first_leaf[component] = leaf;
            }
        }
        for (int node = instance_.trees[side].leaves(); node < instance_.trees[side].vertices(); ++node) {
            const auto [left, right] = info_[side].children[node];
            node_touched[node] = (left >= 0 ? node_touched[left] : 0) + (right >= 0 ? node_touched[right] : 0);
        }

        touched_nodes_[side].clear();
        touched_components_[side].clear();
        dense_work_dirty_[side] = true;
        return finish_completion_from_counts(side, touched_components, false);
    }

    void update_best_zero_subtree(const int side, const int component, const int node) {
        const int size = info_[side].subtree_end[node] - info_[side].subtree_begin[node];
        const int current = best_zero_node_[side][component];
        if (size > best_zero_size_[side][component] ||
            (size == best_zero_size_[side][component] &&
             (current < 0 || info_[side].component_order[node] < info_[side].component_order[current]))) {
            best_zero_size_[side][component] = size;
            best_zero_node_[side][component] = node;
        }
    }

    [[nodiscard]] Completion complete_side_sparse(const int side) {
        if (selected_inverted_[side])
            throw std::logic_error("cut-aware sparse completion expected explicit selected leaves");
        reset_sparse_work(side);
        auto &node_touched = node_touched_[side];
        auto &component_touched = component_touched_[side];
        auto &first_leaf = first_leaf_[side];

        for (const int leaf : selected_leaves_[side]) {
            const int component = info_[side].component_of_leaf[leaf];
            if (component < 0)
                throw std::runtime_error("cut-aware cluster reduction selected a missing leaf");
            if (component_touched[component]++ == 0) {
                touched_components_[side].push_back(component);
                first_leaf[component] = leaf;
            } else if (leaf < first_leaf[component]) {
                first_leaf[component] = leaf;
            }
            for (int node = leaf; node >= 0 && info_[side].component_of_node[node] == component;
                 node = instance_.trees[side].parent[node]) {
                if (node_touched[node]++ == 0)
                    touched_nodes_[side].push_back(node);
            }
        }

        std::ranges::sort(touched_components_[side], [&](const int lhs, const int rhs) {
            return first_leaf[lhs] < first_leaf[rhs];
        });

        for (const int node : touched_nodes_[side]) {
            const int component = info_[side].component_of_node[node];
            if (component < 0)
                continue;
            const auto [left, right] = info_[side].children[node];
            if (left >= 0 && node_touched[left] == 0)
                update_best_zero_subtree(side, component, left);
            if (right >= 0 && node_touched[right] == 0)
                update_best_zero_subtree(side, component, right);
        }

        return finish_completion_from_counts(side, touched_components_[side], true);
    }

    [[nodiscard]] int first_selected_leaf_in_component_from_unselected_counts(
        const int side,
        const int component) const {
        int node = info_[side].components[component].root;
        while (node >= instance_.trees[side].leaves()) {
            const auto [left, right] = info_[side].children[node];
            if (left >= 0) {
                const int left_size = info_[side].subtree_end[left] - info_[side].subtree_begin[left];
                if (node_touched_[side][left] < left_size) {
                    node = left;
                    continue;
                }
            }
            if (right >= 0) {
                const int right_size = info_[side].subtree_end[right] - info_[side].subtree_begin[right];
                if (node_touched_[side][right] < right_size) {
                    node = right;
                    continue;
                }
            }
            return -1;
        }
        return node;
    }

    [[nodiscard]] Completion complete_side_complement_sparse(const int side) {
        if (!selected_inverted_[side])
            throw std::logic_error("cut-aware complement completion expected inverted selected leaves");
        reset_sparse_work(side);
        auto &unselected_node_count = node_touched_[side];
        auto &component_count = component_touched_[side];
        auto &first_leaf = first_leaf_[side];

        for (const int leaf : selected_leaves_[side]) {
            const int component = info_[side].component_of_leaf[leaf];
            if (component < 0)
                throw std::runtime_error("cut-aware cluster reduction selected a missing leaf");
            if (component_count[component]++ == 0)
                touched_components_[side].push_back(component);
            for (int node = leaf; node >= 0 && info_[side].component_of_node[node] == component;
                 node = instance_.trees[side].parent[node]) {
                if (unselected_node_count[node]++ == 0)
                    touched_nodes_[side].push_back(node);
            }
        }

        for (const int component : touched_components_[side]) {
            const int whole = info_[side].components[component].end - info_[side].components[component].begin;
            const int unselected = component_count[component];
            const int selected = whole - unselected;
            if (selected > 0 && selected != whole) {
                first_leaf[component] = first_selected_leaf_in_component_from_unselected_counts(side, component);
                if (first_leaf[component] < 0)
                    throw std::runtime_error("cut-aware cluster reduction lost selected leaves");
            }
        }

        for (const int node : touched_nodes_[side]) {
            const int component = info_[side].component_of_node[node];
            if (component < 0)
                continue;
            const int size = info_[side].subtree_end[node] - info_[side].subtree_begin[node];
            if (unselected_node_count[node] == size)
                update_best_zero_subtree(side, component, node);
        }

        Completion completion = finish_inverted_completion_from_counts(side, touched_components_[side], true);
        for (const int component : touched_components_[side]) {
            component_count[component] = 0;
            first_leaf[component] = -1;
            best_zero_node_[side][component] = -1;
            best_zero_size_[side][component] = 0;
        }
        for (const int node : touched_nodes_[side])
            unselected_node_count[node] = 0;
        touched_nodes_[side].clear();
        touched_components_[side].clear();
        return completion;
    }

    [[nodiscard]] Completion complete_side(const int side) {
        if (!selected_inverted_[side] &&
            static_cast<int>(selected_leaves_[side].size()) <= constants::sparse_completion_leaf_limit)
            return complete_side_sparse(side);

        if (selected_inverted_[side] &&
            static_cast<int>(selected_leaves_[side].size()) <= constants::sparse_completion_leaf_limit)
            return complete_side_complement_sparse(side);

        return complete_side_dense(side);
    }

    [[nodiscard]] bool selected_equals_completion(const int side, const Completion &completion) const {
        if (selected_leaf_count(side) != completion.leaf_count)
            return false;
        if (completion.inverted) {
            return std::ranges::none_of(completion.leaves, [&](const int leaf) {
                return is_selected(side, leaf);
            });
        }
        return std::ranges::all_of(completion.leaves, [&](const int leaf) {
            return is_selected(side, leaf);
        });
    }

    [[nodiscard]] std::optional<CutAwareCluster> close_from_seed(const int seed_side) {
        const int other = seed_side ^ 1;
        while (true) {
            check_timeout();
            if (search_budget_exhausted())
                return std::nullopt;
            const Completion on_seed = complete_side(seed_side);
            if (pruned_by_best(on_seed.leaf_count))
                return std::nullopt;

            set_selected_completion(other, on_seed);

            const Completion on_other = complete_side(other);
            if (pruned_by_best(on_other.leaf_count))
                return std::nullopt;
            if (on_other.leaf_count > total_leaves_ - 2)
                return std::nullopt;

            if (selected_equals_completion(seed_side, on_other)) {
                CutAwareCluster cluster;
                cluster.leaves = on_other.leaves;
                cluster.representation[seed_side] = on_seed.representation;
                cluster.representation[other] = on_other.representation;
                if (on_other.inverted) {
                    cluster.representation[seed_side] =
                        on_seed.inverted
                            ? on_seed.inverse_representation
                            : complement_representation(on_seed.representation, info_[seed_side]);
                    cluster.representation[other] = on_other.inverse_representation;
                    cluster = complement_cluster(cluster);
                }
                if (!cluster_representation_matches_leaves(cluster))
                    return std::nullopt;
                CutAwareCluster result = cluster;
                static_cast<void>(consider_cluster(std::move(cluster)));
                return result;
            }
            set_selected_completion(seed_side, on_other);
        }
    }
};

void validate_cut_aware_instance(const AnnotatedInstance &instance) {
    if (instance.trees.size() != 2)
        throw std::invalid_argument("cut-aware cluster reduction currently supports exactly two trees");
    if (instance.trees[0].leaves() != instance.trees[1].leaves())
        throw std::invalid_argument("cut-aware cluster reduction expects a common leaf set");
}

struct SideSummary {
    std::vector<std::vector<int>> closed_blocks;
    std::vector<int> bridge_block;
};

struct SideBuild {
    AnnotatedInstance instance;
    std::vector<int> original_leaf_by_local;
    int placeholder = -1;
};

struct CutAwareReduction {
    Reduced reduced;
    int reduction_side_leaves = 0;
};

class TreeBuilder {
  public:
    TreeBuilder(
        const Tree &tree,
        const ForestInfo &info,
        const std::vector<int> &local_leaf_of_original,
        const bool with_placeholder)
        : tree_(tree),
          info_(info),
          local_leaf_of_original_(local_leaf_of_original),
          placeholder_(with_placeholder ? static_cast<int>(std::ranges::count_if(local_leaf_of_original, [](const int leaf) {
              return leaf >= 0;
          }))
                                        : -1) {
        const int leaf_count = placeholder_ >= 0 ? placeholder_ + 1 : static_cast<int>(std::ranges::count_if(local_leaf_of_original_, [](const int leaf) { return leaf >= 0; }));
        tree_out_.parent.assign(leaf_count, -2);
        tree_out_.edge_state.assign(leaf_count, EdgeState::UNKNOWN);
        next_ = leaf_count;
    }

    [[nodiscard]] int placeholder() const {
        return placeholder_;
    }

    [[nodiscard]] Tree build(const CutAwareRepresentation &representation) {
        std::vector<int> roots;
        roots.reserve(representation.components.size() + 1);
        for (const int component : representation.components)
            roots.push_back(copy_subtree(info_.components[component].root));
        if (representation.partial_kind != CutAwarePartialKind::NONE) {
            const int partial_root = build_partial(representation);
            if (partial_root < 0) {
                throw std::runtime_error(std::format(
                    "cut-aware cluster reduction lost partial component kind={} component={} node={} root={} placeholder={}",
                    static_cast<int>(representation.partial_kind),
                    representation.partial_component,
                    representation.partial_node,
                    representation.partial_component >= 0 ? info_.components[representation.partial_component].root : -1,
                    placeholder_));
            }
            roots.push_back(partial_root);
        }
        if (roots.empty())
            throw std::runtime_error("cut-aware cluster reduction produced an empty side");
        int current = roots.front();
        for (int i = 1; i < static_cast<int>(roots.size()); ++i)
            current = join_roots(current, roots[i]);
        tree_out_.parent[current] = -1;
        tree_out_.edge_state[current] = EdgeState::UNKNOWN;
        return std::move(tree_out_);
    }

  private:
    [[nodiscard]] int alloc_internal(const int left, const int right, const EdgeState right_state = EdgeState::UNKNOWN) {
        const int current = next_++;
        tree_out_.parent.resize(next_, -2);
        tree_out_.edge_state.resize(next_, EdgeState::UNKNOWN);
        tree_out_.parent[left] = current;
        tree_out_.edge_state[left] = EdgeState::UNKNOWN;
        tree_out_.parent[right] = current;
        tree_out_.edge_state[right] = right_state;
        return current;
    }

    [[nodiscard]] int join_roots(const int current_root, const int cut_root) {
        return alloc_internal(current_root, cut_root, EdgeState::CUT);
    }

    [[nodiscard]] int copy_subtree(const int node) {
        if (node < tree_.leaves())
            return local_leaf_of_original_[node];
        const auto [left, right] = info_.children[node];
        const int a = left >= 0 ? copy_subtree(left) : -1;
        const int b = right >= 0 ? copy_subtree(right) : -1;
        if (a < 0 && b < 0)
            throw std::runtime_error(std::format("cut-aware cluster reduction lost a selected subtree at node={}", node));
        if (a < 0 || b < 0)
            return a < 0 ? b : a;
        return alloc_internal(a, b);
    }

    [[nodiscard]] int copy_without_subtree(
        const int node,
        const int omitted_node,
        const int replacement) {
        if (node == omitted_node)
            return replacement;
        if (node < tree_.leaves())
            return local_leaf_of_original_[node];
        const auto [left, right] = info_.children[node];
        const int a = left >= 0 ? copy_without_subtree(left, omitted_node, replacement) : -1;
        const int b = right >= 0 ? copy_without_subtree(right, omitted_node, replacement) : -1;
        if (a < 0 || b < 0)
            return a < 0 ? b : a;
        return alloc_internal(a, b);
    }

    [[nodiscard]] int build_partial(const CutAwareRepresentation &representation) {
        if (representation.partial_kind == CutAwarePartialKind::BOTTOM) {
            int root = copy_subtree(representation.partial_node);
            if (placeholder_ >= 0)
                root = alloc_internal(root, placeholder_);
            return root;
        }
        return copy_without_subtree(
            info_.components[representation.partial_component].root,
            representation.partial_node,
            placeholder_);
    }

    const Tree &tree_;
    const ForestInfo &info_;
    const std::vector<int> &local_leaf_of_original_;
    int placeholder_ = -1;
    int next_ = 0;
    Tree tree_out_;
};

[[nodiscard]] SideBuild build_side(
    const AnnotatedInstance &instance,
    const std::array<ForestInfo, 2> &info,
    const std::array<CutAwareRepresentation, 2> &representation,
    const std::vector<int> &leaves,
    const bool with_placeholder) {
    SideBuild build{
        .instance = {},
        .original_leaf_by_local = leaves,
        .placeholder = with_placeholder ? static_cast<int>(leaves.size()) : -1,
    };
    std::vector<int> local_leaf_of_original(instance.trees.front().leaves(), -1);
    for (int i = 0; i < static_cast<int>(leaves.size()); ++i)
        local_leaf_of_original[leaves[i]] = i;
    build.instance.trees.reserve(instance.trees.size());
    for (int tree = 0; tree < static_cast<int>(instance.trees.size()); ++tree) {
        TreeBuilder builder(instance.trees[tree], info[tree], local_leaf_of_original, with_placeholder);
        if (builder.placeholder() != build.placeholder)
            throw std::runtime_error("cut-aware cluster reduction placeholder mismatch");
        build.instance.trees.push_back(builder.build(representation[tree]));
    }
    return build;
}

[[nodiscard]] SideSummary summarize_side_result(const Result &result, const SideBuild &build) {
    SideSummary summary;
    for (const auto &block : result.partition) {
        const bool has_placeholder = build.placeholder >= 0 &&
                                     std::ranges::find(block, build.placeholder) != block.end();
        std::vector<int> expanded;
        expanded.reserve(block.size());
        for (const int leaf : block) {
            if (leaf == build.placeholder)
                continue;
            if (leaf < 0 || leaf >= static_cast<int>(build.original_leaf_by_local.size()))
                throw std::runtime_error("cut-aware cluster reduction saw invalid lifted leaf");
            expanded.push_back(build.original_leaf_by_local[leaf]);
        }
        std::ranges::sort(expanded);
        if (has_placeholder)
            summary.bridge_block = std::move(expanded);
        else
            summary.closed_blocks.push_back(std::move(expanded));
    }
    sort_partition_blocks(summary.closed_blocks);
    return summary;
}

} // namespace

std::optional<CutAwareReduction> apply_cut_aware_plan(
    const AnnotatedInstance &instance,
    SolveContext &context,
    const int objective_offset,
    CutAwarePlan plan) {
    const auto &cluster = plan.cluster;
    const auto &info = plan.info;
    std::array<CutAwareRepresentation, 2> cluster_representation = {
        normalize_representation(cluster.representation[0], info[0]),
        normalize_representation(cluster.representation[1], info[1]),
    };
    const int reduction_side_leaves = std::min(
        static_cast<int>(cluster.leaves.size()),
        instance.trees.front().leaves() - static_cast<int>(cluster.leaves.size())
    );
    const std::array residual_representation = {
        complement_representation(cluster_representation[0], info[0]),
        complement_representation(cluster_representation[1], info[1]),
    };
    std::vector<char> in_cluster(instance.trees.front().leaves(), 0);
    for (const int leaf : cluster.leaves)
        in_cluster[leaf] = 1;
    std::vector<int> residual_leaves;
    residual_leaves.reserve(instance.trees.front().leaves() - static_cast<int>(cluster.leaves.size()));
    for (int leaf = 0; leaf < instance.trees.front().leaves(); ++leaf) {
        if (!in_cluster[leaf])
            residual_leaves.push_back(leaf);
    }

    const bool solve_cluster_side =
        static_cast<int>(cluster.leaves.size()) <= static_cast<int>(residual_leaves.size());
    const auto &solved_leaves = solve_cluster_side ? cluster.leaves : residual_leaves;
    const auto &remaining_leaves = solve_cluster_side ? residual_leaves : cluster.leaves;
    const auto &solved_representation = solve_cluster_side ? cluster_representation : residual_representation;
    const auto &remaining_representation = solve_cluster_side ? residual_representation : cluster_representation;

    std::vector<int> representation_mark(instance.trees.front().leaves(), 0);
    int representation_stamp = 0;
    const auto matches = [&](const ForestInfo &side_info,
                             const CutAwareRepresentation &representation,
                             const std::vector<int> &leaves) {
        if (++representation_stamp == std::numeric_limits<int>::max()) {
            std::ranges::fill(representation_mark, 0);
            representation_stamp = 1;
        }
        return representation_matches_leaves(
            side_info,
            representation,
            leaves,
            representation_mark,
            representation_stamp);
    };
    const bool representation_consistent =
        matches(info[0], solved_representation[0], solved_leaves) &&
        matches(info[1], solved_representation[1], solved_leaves) &&
        matches(info[0], remaining_representation[0], remaining_leaves) &&
        matches(info[1], remaining_representation[1], remaining_leaves);
    if (!representation_consistent)
        throw std::runtime_error("cut-aware cluster reduction produced an inconsistent representation");

    if (2 * static_cast<int>(solved_leaves.size()) > instance.trees.front().leaves() + 2)
        return std::nullopt;

    const bool attach_capable =
        solved_representation[0].partial_kind != CutAwarePartialKind::NONE &&
        solved_representation[1].partial_kind != CutAwarePartialKind::NONE;

    SubinstanceStats stats;
    const SideBuild solved_without = build_side(
        instance,
        info,
        solved_representation,
        solved_leaves,
        false);
    bool attach = false;
    Result solved_result;
    const SideBuild *solved_build = &solved_without;
    std::optional<SideBuild> solved_with;
    if (attach_capable) {
        solved_with = build_side(
            instance,
            info,
            solved_representation,
            solved_leaves,
            true);
        const auto related = solve_related_side(
            solved_with->instance,
            solved_without.instance,
            context,
            objective_offset,
            stats);
        attach = should_attach_placeholder(
            related.with_result,
            related.without_result,
            "cut-aware cluster reduction",
            "cut-aware dummy");
#if MAFFE_COMPETITION_HEURISTIC_BUILD
        if (attach && !has_nonempty_placeholder_block(related.with_result, solved_with->placeholder))
            attach = false;
#endif
        solved_result = attach ? related.with_result : related.without_result;
        if (attach)
            solved_build = &*solved_with;
    } else {
        solved_result = solve_subinstance(stats, solved_without.instance, [&] {
            return context.solve_residual(
                solved_without.instance,
                nullptr,
                nullptr,
                nullptr,
                objective_offset);
        });
    }
    SideSummary solved_summary = summarize_side_result(solved_result, *solved_build);

    SideBuild remaining_build = build_side(
        instance,
        info,
        remaining_representation,
        remaining_leaves,
        attach);
    const int closed_count = static_cast<int>(solved_summary.closed_blocks.size());
    return CutAwareReduction{
        .reduced = Reduced{
            .instance = std::move(remaining_build.instance),
            .lift = [solved_summary = std::move(solved_summary),
                     remaining_build = std::move(remaining_build)](Result result) mutable {
                if (!result.feasible)
                    return result;
                return combine_side_summaries(
                    std::move(solved_summary),
                    summarize_side_result(result, remaining_build));
            },
            .objective_offset = closed_count,
            .reduction_count = 1,
            .subinstance_count = stats.solved,
            .largest_subinstance = stats.largest,
            .subinstance_seconds = stats.seconds,
        },
        .reduction_side_leaves = reduction_side_leaves,
    };
}

std::optional<CutAwareReduction> apply_batched_full_component_plan(
    const AnnotatedInstance &instance,
    SolveContext &context,
    const int objective_offset,
    const std::array<ForestInfo, 2> &info) {
    const int total_leaves = instance.trees.front().leaves();
    std::vector<FullComponentMatch> matches = find_full_component_matches(
        info,
        cut_aware_target_leaves(total_leaves));
    if (matches.empty())
        return std::nullopt;

    std::array<std::vector<char>, 2> removed_component{
        std::vector<char>(info[0].components.size(), 0),
        std::vector<char>(info[1].components.size(), 0),
    };
    std::vector<char> removed_leaf(total_leaves, 0);
    std::vector<SideSummary> solved_summaries;
    SubinstanceStats stats;
    int removed_leaf_count = 0;
    int total_objective_offset = 0;
    int max_reduction_side_leaves = 0;

    for (const FullComponentMatch &match : matches) {
        const int side_leaves = static_cast<int>(match.leaves.size());
        if (total_leaves - removed_leaf_count - side_leaves < 2)
            continue;
        if (std::ranges::any_of(match.leaves, [&](const int leaf) { return removed_leaf[leaf] != 0; }))
            continue;

        std::array<CutAwareRepresentation, 2> representation;
        for (int side = 0; side < 2; ++side)
            representation[side].components.push_back(match.component[side]);

        const SideBuild solved_build = build_side(
            instance,
            info,
            representation,
            match.leaves,
            false);
        const Result solved_result = solve_subinstance(stats, solved_build.instance, [&] {
            return context.solve_residual(
                solved_build.instance,
                nullptr,
                nullptr,
                nullptr,
                objective_offset + total_objective_offset);
        });
        SideSummary solved_summary = summarize_side_result(solved_result, solved_build);
        total_objective_offset += static_cast<int>(solved_summary.closed_blocks.size());
        solved_summaries.push_back(std::move(solved_summary));
        max_reduction_side_leaves = std::max(max_reduction_side_leaves, side_leaves);

        for (int side = 0; side < 2; ++side)
            removed_component[side][match.component[side]] = 1;
        for (const int leaf : match.leaves)
            removed_leaf[leaf] = 1;
        removed_leaf_count += side_leaves;
    }

    if (solved_summaries.empty())
        return std::nullopt;

    std::vector<int> remaining_leaves;
    remaining_leaves.reserve(total_leaves - removed_leaf_count);
    for (int leaf = 0; leaf < total_leaves; ++leaf) {
        if (removed_leaf[leaf] == 0)
            remaining_leaves.push_back(leaf);
    }

    std::array<CutAwareRepresentation, 2> remaining_representation;
    for (int side = 0; side < 2; ++side) {
        for (int component = 0; component < static_cast<int>(info[side].components.size()); ++component) {
            if (removed_component[side][component] == 0)
                remaining_representation[side].components.push_back(component);
        }
    }
    SideBuild remaining_build = build_side(
        instance,
        info,
        remaining_representation,
        remaining_leaves,
        false);
    const int solved_count = static_cast<int>(solved_summaries.size());

    return CutAwareReduction{
        .reduced = Reduced{
            .instance = std::move(remaining_build.instance),
            .lift = [solved_summaries = std::move(solved_summaries),
                     remaining_build = std::move(remaining_build)](Result result) mutable {
                if (!result.feasible)
                    return result;
                Result combined{
                    .partition = {},
                    .feasible = true,
                };
                for (auto &summary : solved_summaries) {
                    combined.partition.insert(
                        combined.partition.end(),
                        std::make_move_iterator(summary.closed_blocks.begin()),
                        std::make_move_iterator(summary.closed_blocks.end()));
                }
                SideSummary remaining_summary = summarize_side_result(result, remaining_build);
                combined.partition.insert(
                    combined.partition.end(),
                    std::make_move_iterator(remaining_summary.closed_blocks.begin()),
                    std::make_move_iterator(remaining_summary.closed_blocks.end()));
                sort_partition_blocks(combined.partition);
                return combined;
            },
            .objective_offset = total_objective_offset,
            .reduction_count = solved_count,
            .subinstance_count = stats.solved,
            .largest_subinstance = stats.largest,
            .subinstance_seconds = stats.seconds,
        },
        .reduction_side_leaves = max_reduction_side_leaves,
    };
}

std::optional<CutAwareReduction> try_cut_aware_cluster_reduction_impl(
    const AnnotatedInstance &instance,
    SolveContext &context,
    const int objective_offset,
    const std::optional<int> preferred_seed_side_leaves) {
    context.check_timeout();
    validate_cut_aware_instance(instance);
    if (instance.trees[0].leaves() < 4)
        return std::nullopt;

    ClosureSearch search(instance, &context, preferred_seed_side_leaves);
    auto cluster = search.best_small_direct_cluster();
    if (!cluster.has_value())
        cluster = search.best_cluster();
    std::array<ForestInfo, 2> info = std::move(search).take_info();
    if (!cluster.has_value())
        return std::nullopt;
    if (full_component_cluster(*cluster)) {
        if (auto batched = apply_batched_full_component_plan(instance, context, objective_offset, info))
            return batched;
    }
    return apply_cut_aware_plan(instance, context, objective_offset, CutAwarePlan{
        .cluster = std::move(*cluster),
        .info = std::move(info),
    });
}

} // namespace maffe::detail

namespace maffe {

std::optional<Reduced> try_cut_aware_cluster_reduction(
    const AnnotatedInstance &instance,
    SolveContext &context,
    const int objective_offset) {
    AnnotatedInstance current = instance;
    std::vector<Lift> lifts;
    int total_objective_offset = 0;
    int total_reduction_count = 0;
    int total_subinstance_count = 0;
    int largest_subinstance = 0;
    double total_subinstance_seconds = 0.0;
    std::optional<int> preferred_seed_side_leaves;
    while (true) {
        std::optional<detail::CutAwareReduction> reduction;
        try {
            context.check_timeout();
            reduction = detail::try_cut_aware_cluster_reduction_impl(
                current,
                context,
                objective_offset + total_objective_offset,
                preferred_seed_side_leaves
            );
        } catch (const SubsolveTimeout&) {
            break;
        }
        if (!reduction.has_value())
            break;
        Reduced reduced = std::move(reduction->reduced);
        total_objective_offset += reduced.objective_offset;
        total_reduction_count += reduced.reduction_count;
        total_subinstance_count += reduced.subinstance_count;
        largest_subinstance = std::max(largest_subinstance, reduced.largest_subinstance);
        total_subinstance_seconds += reduced.subinstance_seconds;
        lifts.push_back(std::move(reduced.lift));
        current = std::move(reduced.instance);
        if (reduction->reduction_side_leaves > constants::small_cut_aware_cluster_leaves)
            break;
        preferred_seed_side_leaves = constants::small_cut_aware_cluster_leaves;
    }
    if (lifts.empty())
        return std::nullopt;
    return Reduced{
        .instance = std::move(current),
        .lift = [lifts = std::move(lifts)](Result result) mutable {
            for (int i = static_cast<int>(lifts.size()) - 1; i >= 0; --i)
                result = lifts[i](std::move(result));
            return result;
        },
        .objective_offset = total_objective_offset,
        .reduction_count = total_reduction_count,
        .subinstance_count = total_subinstance_count,
        .largest_subinstance = largest_subinstance,
        .subinstance_seconds = total_subinstance_seconds,
    };
}

} // namespace maffe
