#include "branchandprice/master/set_packing_validation.hpp"
#include "branchandprice/master/set_packing_heuristic.hpp"

#include "util/constants.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace maffe {
namespace {

struct Candidate {
    int view_index = -1;
    int column_id = -1;
    double lp_value = 0.0;
    double dual_cost = 0.0;
    double history_value = 0.0;
    bool warm_start = false;
    int saving = 0;
    int forced_count = 0;
    double base_score = 0.0;
    double score = 0.0;
    std::uint64_t tie_break = 0;
};

struct State {
    std::vector<int> leaf_owner;
    std::vector<int> row_owner;
    std::vector<int> forced_cover_count;
    std::vector<char> selected;
    int saving = 0;

    State(const int leaf_count, const int row_count, const int forced_row_count, const int candidate_count)
        : leaf_owner(leaf_count, -1),
          row_owner(row_count, -1),
          forced_cover_count(forced_row_count, 0),
          selected(candidate_count, 0) {}
};

struct PairBucket {
    std::array<int, constants::scip_root_set_packing_heuristic_pair_bucket_size> candidates{};
    int count = 0;
};

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

[[nodiscard]] double deterministic_unit_interval(const std::uint64_t seed) {
    return std::max(0x1.0p-53, static_cast<double>(splitmix64(seed) >> 11) * 0x1.0p-53);
}

[[nodiscard]] bool better_candidate(const Candidate& lhs, const Candidate& rhs) {
    if (lhs.score != rhs.score)
        return lhs.score > rhs.score;
    if (lhs.lp_value != rhs.lp_value)
        return lhs.lp_value > rhs.lp_value;
    if (lhs.saving != rhs.saving)
        return lhs.saving > rhs.saving;
    return lhs.tie_break < rhs.tie_break;
}

[[nodiscard]] bool better_pair_candidate(
    const std::vector<Candidate>& candidates,
    const int lhs,
    const int rhs
) {
    const auto& a = candidates[lhs];
    const auto& b = candidates[rhs];
    if (a.saving != b.saving)
        return a.saving > b.saving;
    if (a.forced_count != b.forced_count)
        return a.forced_count > b.forced_count;
    if (a.score != b.score)
        return a.score > b.score;
    if (a.lp_value != b.lp_value)
        return a.lp_value > b.lp_value;
    return a.tie_break < b.tie_break;
}

[[nodiscard]] bool sorted_ranges_intersect(
    const std::span<const int> lhs,
    const std::span<const int> rhs
) {
    int i = 0;
    int j = 0;
    while (i < static_cast<int>(lhs.size()) && j < static_cast<int>(rhs.size())) {
        if (lhs[i] == rhs[j])
            return true;
        if (lhs[i] < rhs[j])
            ++i;
        else
            ++j;
    }
    return false;
}

[[nodiscard]] bool columns_compatible(
    const std::span<const RootSetPackingColumnView> columns,
    const Candidate& lhs,
    const Candidate& rhs
) {
    const RootSetPackingColumnView& a = columns[lhs.view_index];
    const RootSetPackingColumnView& b = columns[rhs.view_index];
    return !sorted_ranges_intersect(a.leaves, b.leaves) &&
           !sorted_ranges_intersect(a.row_indices, b.row_indices);
}

[[nodiscard]] std::vector<Candidate> build_candidates(
    const int leaf_count,
    const int row_count,
    const int forced_row_count,
    const int column_count,
    const std::span<const RootSetPackingColumnView> columns
) {
    std::vector<Candidate> candidates;
    candidates.reserve(columns.size());

    for (int i = 0; i < static_cast<int>(columns.size()); ++i) {
        const RootSetPackingColumnView& column = columns[i];
        detail::validate_packing_column(column, leaf_count, row_count, forced_row_count, column_count);
        if (column.leaves.size() < 2)
            continue;
        const double lp_value = std::clamp(column.lp_value, 0.0, 1.0);
        const double dual_cost = std::isfinite(column.dual_cost) ? std::max(0.0, column.dual_cost) : 0.0;
        const double history_value = std::clamp(column.history_value, 0.0, 1.0);
        const int saving = static_cast<int>(column.leaves.size()) - 1;
        const double base_score =
            std::max(constants::scip_root_set_packing_heuristic_min_probability, lp_value) *
            static_cast<double>(saving);
        candidates.push_back(Candidate{
            .view_index = i,
            .column_id = column.column_id,
            .lp_value = lp_value,
            .dual_cost = dual_cost,
            .history_value = history_value,
            .warm_start = column.warm_start,
            .saving = saving,
            .forced_count = static_cast<int>(column.forced_rows.size()),
            .base_score = base_score,
            .score = base_score,
            .tie_break = splitmix64(column.tie_seed),
        });
    }

    std::vector<double> leaf_pressure(leaf_count, 0.0);
    std::vector<double> row_pressure(row_count, 0.0);
    for (const Candidate& candidate : candidates) {
        const auto& column = columns[candidate.view_index];
        for (const int leaf : column.leaves)
            leaf_pressure[leaf] += candidate.base_score;
        for (const int row : column.row_indices)
            row_pressure[row] += candidate.base_score;
    }

    for (Candidate& candidate : candidates) {
        const auto& column = columns[candidate.view_index];
        double pressure_sum = 0.0;
        for (const int leaf : column.leaves)
            pressure_sum += std::max(0.0, leaf_pressure[leaf] - candidate.base_score);
        for (const int row : column.row_indices)
            pressure_sum += std::max(0.0, row_pressure[row] - candidate.base_score);
        const double support_size = static_cast<double>(column.leaves.size() + column.row_indices.size());
        const double average_pressure = support_size > 0.0 ? pressure_sum / support_size : 0.0;
        candidate.score = candidate.base_score / (1.0 + std::sqrt(average_pressure));
    }

    return candidates;
}

[[nodiscard]] bool can_add(
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    const State& state,
    const int candidate_index
) {
    const RootSetPackingColumnView& column = columns[candidates[candidate_index].view_index];
    for (const int leaf : column.leaves) {
        if (state.leaf_owner[leaf] >= 0)
            return false;
    }
    for (const int row : column.row_indices) {
        if (state.row_owner[row] >= 0)
            return false;
    }
    return true;
}

[[nodiscard]] bool can_add_after_removing(
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    const State& state,
    const int candidate_index,
    const int removed_candidate_index
) {
    const RootSetPackingColumnView& column = columns[candidates[candidate_index].view_index];
    for (const int leaf : column.leaves) {
        const int owner = state.leaf_owner[leaf];
        if (owner >= 0 && owner != removed_candidate_index)
            return false;
    }
    for (const int row : column.row_indices) {
        const int owner = state.row_owner[row];
        if (owner >= 0 && owner != removed_candidate_index)
            return false;
    }
    return true;
}

[[nodiscard]] bool can_add_after_removing(
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    const State& state,
    const int candidate_index,
    const std::vector<char>& removed
) {
    const RootSetPackingColumnView& column = columns[candidates[candidate_index].view_index];
    for (const int leaf : column.leaves) {
        const int owner = state.leaf_owner[leaf];
        if (owner >= 0 && removed[owner] == 0)
            return false;
    }
    for (const int row : column.row_indices) {
        const int owner = state.row_owner[row];
        if (owner >= 0 && removed[owner] == 0)
            return false;
    }
    return true;
}

void add_column(
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    State& state,
    const int candidate_index
) {
    const RootSetPackingColumnView& column = columns[candidates[candidate_index].view_index];
    state.selected[candidate_index] = 1;
    state.saving += candidates[candidate_index].saving;
    for (const int leaf : column.leaves) {
        state.leaf_owner[leaf] = candidate_index;
    }
    for (const int row : column.row_indices)
        state.row_owner[row] = candidate_index;
    for (const int row : column.forced_rows)
        ++state.forced_cover_count[row];
}

void remove_column(
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    State& state,
    const int candidate_index
) {
    const RootSetPackingColumnView& column = columns[candidates[candidate_index].view_index];
    state.selected[candidate_index] = 0;
    state.saving -= candidates[candidate_index].saving;
    for (const int leaf : column.leaves) {
        if (state.leaf_owner[leaf] == candidate_index)
            state.leaf_owner[leaf] = -1;
    }
    for (const int row : column.row_indices) {
        if (state.row_owner[row] == candidate_index)
            state.row_owner[row] = -1;
    }
    for (const int row : column.forced_rows)
        --state.forced_cover_count[row];
}

[[nodiscard]] bool forced_rows_covered(const State& state) {
    return std::ranges::all_of(state.forced_cover_count, [](const int count) {
        return count > 0;
    });
}

void collect_conflicts(
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    const State& state,
    const int candidate_index,
    std::vector<int>& seen,
    std::vector<int>& conflicts,
    int& stamp
) {
    conflicts.clear();
    ++stamp;
    const auto mark_conflict = [&](const int owner) {
        if (owner < 0 || seen[owner] == stamp)
            return;
        seen[owner] = stamp;
        conflicts.push_back(owner);
    };

    const RootSetPackingColumnView& column = columns[candidates[candidate_index].view_index];
    for (const int leaf : column.leaves)
        mark_conflict(state.leaf_owner[leaf]);
    for (const int row : column.row_indices)
        mark_conflict(state.row_owner[row]);
}

void insert_pair_candidate(
    PairBucket& bucket,
    const std::vector<Candidate>& candidates,
    const int candidate_index
) {
    int pos = bucket.count;
    if (pos < constants::scip_root_set_packing_heuristic_pair_bucket_size) {
        ++bucket.count;
    } else {
        pos = constants::scip_root_set_packing_heuristic_pair_bucket_size - 1;
        if (!better_pair_candidate(candidates, candidate_index, bucket.candidates[pos]))
            return;
    }

    bucket.candidates[pos] = candidate_index;
    while (pos > 0 && better_pair_candidate(candidates, bucket.candidates[pos], bucket.candidates[pos - 1])) {
        std::swap(bucket.candidates[pos], bucket.candidates[pos - 1]);
        --pos;
    }
}

[[nodiscard]] bool improve_pairs(
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    State& state
) {
    std::vector<PairBucket> buckets(candidates.size());
    std::vector<int> seen(candidates.size(), 0);
    std::vector<int> conflicts;
    int stamp = 0;

    for (int candidate_index = 0; candidate_index < static_cast<int>(candidates.size()); ++candidate_index) {
        if (state.selected[candidate_index] != 0)
            continue;

        collect_conflicts(columns, candidates, state, candidate_index, seen, conflicts, stamp);
        if (conflicts.size() == 1 && state.selected[conflicts.front()] != 0)
            insert_pair_candidate(buckets[conflicts.front()], candidates, candidate_index);
    }

    bool changed = false;
    for (int removed = 0; removed < static_cast<int>(buckets.size()); ++removed) {
        const PairBucket& bucket = buckets[removed];
        if (bucket.count < 2 || state.selected[removed] == 0)
            continue;

        int best_first = -1;
        int best_second = -1;
        int best_gain = 0;
        for (int i = 0; i < bucket.count; ++i) {
            const int first = bucket.candidates[i];
            if (state.selected[first] != 0 ||
                !can_add_after_removing(columns, candidates, state, first, removed)) {
                continue;
            }
            for (int j = i + 1; j < bucket.count; ++j) {
                const int second = bucket.candidates[j];
                if (state.selected[second] != 0 ||
                    !can_add_after_removing(columns, candidates, state, second, removed) ||
                    !columns_compatible(columns, candidates[first], candidates[second])) {
                    continue;
                }
                const int gain = candidates[first].saving + candidates[second].saving - candidates[removed].saving;
                if (gain > best_gain) {
                    best_first = first;
                    best_second = second;
                    best_gain = gain;
                }
            }
        }
        if (best_gain <= 0)
            continue;
        State trial = state;
        remove_column(columns, candidates, trial, removed);
        add_column(columns, candidates, trial, best_first);
        add_column(columns, candidates, trial, best_second);
        if (!forced_rows_covered(trial))
            continue;
        state = std::move(trial);
        changed = true;
    }
    return changed;
}

void greedy_refill(
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    State& state
) {
    const int limit = std::min(
        static_cast<int>(candidates.size()),
        constants::scip_root_set_packing_heuristic_ejection_refill_limit);
    for (int candidate_index = 0; candidate_index < limit; ++candidate_index) {
        if (state.selected[candidate_index] == 0 && can_add(columns, candidates, state, candidate_index))
            add_column(columns, candidates, state, candidate_index);
    }
}

[[nodiscard]] bool cover_forced_rows(
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    State& state
) {
    if (state.forced_cover_count.empty())
        return true;

    std::vector<std::vector<int>> covering(state.forced_cover_count.size());
    for (int candidate_index = 0; candidate_index < static_cast<int>(candidates.size()); ++candidate_index) {
        const RootSetPackingColumnView& column = columns[candidates[candidate_index].view_index];
        for (const int row : column.forced_rows)
            covering[static_cast<std::size_t>(row)].push_back(candidate_index);
    }

    std::vector<int> order(state.forced_cover_count.size());
    for (int row = 0; row < static_cast<int>(order.size()); ++row)
        order[static_cast<std::size_t>(row)] = row;
    std::ranges::sort(order, [&](const int lhs, const int rhs) {
        const auto& a = covering[static_cast<std::size_t>(lhs)];
        const auto& b = covering[static_cast<std::size_t>(rhs)];
        if (a.size() != b.size())
            return a.size() < b.size();
        return lhs < rhs;
    });

    for (const int row : order) {
        if (state.forced_cover_count[static_cast<std::size_t>(row)] > 0)
            continue;
        int best = -1;
        for (const int candidate_index : covering[static_cast<std::size_t>(row)]) {
            if (state.selected[candidate_index] != 0)
                continue;
            if (!can_add(columns, candidates, state, candidate_index))
                continue;
            if (best < 0 || better_candidate(candidates[candidate_index], candidates[best]))
                best = candidate_index;
        }
        if (best < 0)
            return false;
        add_column(columns, candidates, state, best);
    }
    return true;
}

[[nodiscard]] bool improve_ejection(
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    State& state
) {
    const int limit = std::min(
        static_cast<int>(candidates.size()),
        constants::scip_root_set_packing_heuristic_ejection_candidate_limit);
    std::vector<int> seen(candidates.size(), 0);
    std::vector<int> conflicts;
    int stamp = 0;

    State best = state;
    for (int candidate_index = 0; candidate_index < limit; ++candidate_index) {
        if (state.selected[candidate_index] != 0)
            continue;
        collect_conflicts(columns, candidates, state, candidate_index, seen, conflicts, stamp);
        if (conflicts.size() < 2 ||
            static_cast<int>(conflicts.size()) > constants::scip_root_set_packing_heuristic_max_ejected) {
            continue;
        }

        State trial = state;
        for (const int conflict : conflicts)
            remove_column(columns, candidates, trial, conflict);
        if (!can_add(columns, candidates, trial, candidate_index))
            continue;
        add_column(columns, candidates, trial, candidate_index);
        greedy_refill(columns, candidates, trial);
        if (forced_rows_covered(trial) && trial.saving > best.saving)
            best = std::move(trial);
    }

    if (best.saving <= state.saving)
        return false;
    state = std::move(best);
    return true;
}

[[nodiscard]] std::uint64_t exact_neighborhood_mask(
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    const std::vector<int>& local
) {
    const int count = static_cast<int>(local.size());
    std::vector<std::uint64_t> conflicts(count, 0);
    for (int i = 0; i < count; ++i) {
        conflicts[i] |= std::uint64_t{1} << i;
        for (int j = i + 1; j < count; ++j) {
            if (!columns_compatible(columns, candidates[local[i]], candidates[local[j]])) {
                conflicts[i] |= std::uint64_t{1} << j;
                conflicts[j] |= std::uint64_t{1} << i;
            }
        }
    }

    std::vector<int> saving(count, 0);
    for (int i = 0; i < count; ++i)
        saving[i] = candidates[local[i]].saving;

    int best_saving = 0;
    std::uint64_t best_mask = 0;
    const auto sum_available = [&](std::uint64_t mask) {
        int total = 0;
        while (mask != 0) {
            const int bit = std::countr_zero(mask);
            total += saving[bit];
            mask &= mask - 1;
        }
        return total;
    };

    auto search = [&](auto&& self, const std::uint64_t available, const int current_saving, const std::uint64_t current_mask) -> void {
        if (available == 0) {
            if (current_saving > best_saving) {
                best_saving = current_saving;
                best_mask = current_mask;
            }
            return;
        }
        if (current_saving + sum_available(available) <= best_saving)
            return;

        int chosen = -1;
        double best_score = -1.0;
        std::uint64_t scan = available;
        while (scan != 0) {
            const int bit = std::countr_zero(scan);
            const double score = static_cast<double>(saving[bit]) /
                                 static_cast<double>(std::popcount(conflicts[bit] & available));
            if (score > best_score) {
                best_score = score;
                chosen = bit;
            }
            scan &= scan - 1;
        }
        if (chosen < 0)
            return;

        const std::uint64_t chosen_bit = std::uint64_t{1} << chosen;
        self(
            self,
            available & ~conflicts[chosen],
            current_saving + saving[chosen],
            current_mask | chosen_bit
        );
        self(self, available & ~chosen_bit, current_saving, current_mask);
    };

    search(
        search,
        local.size() == 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << local.size()) - 1),
        0,
        0
    );
    return best_mask;
}

[[nodiscard]] bool improve_exact_neighborhood(
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    State& state
) {
    const int seed_limit = std::min(
        static_cast<int>(candidates.size()),
        constants::scip_root_set_packing_heuristic_exact_seed_limit);
    std::vector<int> seen(candidates.size(), 0);
    std::vector<int> conflicts;
    int stamp = 0;

    State best = state;
    for (int seed = 0; seed < seed_limit; ++seed) {
        if (state.selected[seed] != 0)
            continue;
        collect_conflicts(columns, candidates, state, seed, seen, conflicts, stamp);
        if (conflicts.empty() || static_cast<int>(conflicts.size()) > constants::scip_root_set_packing_heuristic_max_ejected)
            continue;

        std::vector<char> removed(candidates.size(), 0);
        int removed_saving = 0;
        for (const int conflict : conflicts) {
            removed[conflict] = 1;
            removed_saving += candidates[conflict].saving;
        }

        std::vector<char> in_local(candidates.size(), 0);
        std::vector<int> local;
        const auto add_local = [&](const int candidate_index) {
            if (candidate_index < 0 || in_local[candidate_index] != 0)
                return;
            if (!can_add_after_removing(columns, candidates, state, candidate_index, removed))
                return;
            in_local[candidate_index] = 1;
            local.push_back(candidate_index);
        };

        for (const int conflict : conflicts)
            add_local(conflict);
        add_local(seed);
        for (int candidate_index = 0;
             candidate_index < static_cast<int>(candidates.size()) &&
             static_cast<int>(local.size()) < constants::scip_root_set_packing_heuristic_exact_neighborhood_limit;
             ++candidate_index) {
            add_local(candidate_index);
        }
        if (local.size() <= conflicts.size() || local.size() > 63)
            continue;

        std::ranges::sort(local, [&](const int lhs, const int rhs) {
            const auto& a = candidates[lhs];
            const auto& b = candidates[rhs];
            if (a.saving != b.saving)
                return a.saving > b.saving;
            if (a.score != b.score)
                return a.score > b.score;
            return a.tie_break < b.tie_break;
        });

        const std::uint64_t selected_mask = exact_neighborhood_mask(columns, candidates, local);
        int local_saving = 0;
        for (int i = 0; i < static_cast<int>(local.size()); ++i) {
            if ((selected_mask & (std::uint64_t{1} << i)) != 0)
                local_saving += candidates[local[i]].saving;
        }
        if (local_saving <= removed_saving)
            continue;

        State trial = state;
        for (const int conflict : conflicts)
            remove_column(columns, candidates, trial, conflict);
        for (int i = 0; i < static_cast<int>(local.size()); ++i) {
            if ((selected_mask & (std::uint64_t{1} << i)) == 0)
                continue;
            if (trial.selected[local[i]] == 0 && can_add(columns, candidates, trial, local[i]))
                add_column(columns, candidates, trial, local[i]);
        }
        greedy_refill(columns, candidates, trial);
        if (forced_rows_covered(trial) && trial.saving > best.saving)
            best = std::move(trial);
    }

    if (best.saving <= state.saving)
        return false;
    state = std::move(best);
    return true;
}

void improve(
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    State& state
) {
    std::vector<int> seen(candidates.size(), 0);
    std::vector<int> conflicts;
    int stamp = 0;

    for (int pass = 0; pass < constants::scip_root_set_packing_heuristic_improvement_passes; ++pass) {
        bool changed = false;
        for (int candidate_index = 0; candidate_index < static_cast<int>(candidates.size()); ++candidate_index) {
            if (state.selected[candidate_index] != 0)
                continue;

            collect_conflicts(columns, candidates, state, candidate_index, seen, conflicts, stamp);
            int lost_saving = 0;
            for (const int conflict : conflicts)
                lost_saving += candidates[conflict].saving;
            if (candidates[candidate_index].saving <= lost_saving)
                continue;

            State trial = state;
            for (const int conflict : conflicts)
                remove_column(columns, candidates, trial, conflict);
            add_column(columns, candidates, trial, candidate_index);
            if (!forced_rows_covered(trial))
                continue;
            state = std::move(trial);
            changed = true;
        }
        changed = improve_pairs(columns, candidates, state) || changed;
        changed = improve_ejection(columns, candidates, state) || changed;
        if (!changed)
            break;
    }
}

[[nodiscard]] State build_state(
    const int leaf_count,
    const int row_count,
    const int forced_row_count,
    const std::span<const RootSetPackingColumnView> columns,
    const std::vector<Candidate>& candidates,
    const std::vector<int>& order
) {
    State state(leaf_count, row_count, forced_row_count, static_cast<int>(candidates.size()));
    if (!cover_forced_rows(columns, candidates, state))
        return state;
    for (const int candidate_index : order) {
        if (can_add(columns, candidates, state, candidate_index))
            add_column(columns, candidates, state, candidate_index);
    }
    improve(columns, candidates, state);
    return state;
}

[[nodiscard]] RootSetPackingSolution solution_from_state(
    const std::vector<Candidate>& candidates,
    const State& state
) {
    RootSetPackingSolution solution;
    solution.saving = state.saving;
    for (int candidate_index = 0; candidate_index < static_cast<int>(candidates.size()); ++candidate_index) {
        if (state.selected[candidate_index] != 0)
            solution.columns.push_back(candidates[candidate_index].column_id);
    }
    std::ranges::sort(solution.columns);
    return solution;
}

} // namespace

RootSetPackingSolution solve_root_set_packing_heuristic(
    const int leaf_count,
    const int row_count,
    const int forced_row_count,
    const int column_count,
    const std::span<const RootSetPackingColumnView> columns
) {
    if (leaf_count < 0 || row_count < 0 || forced_row_count < 0 || column_count < 0)
        throw std::invalid_argument("set packing dimensions must be nonnegative");

    std::vector<Candidate> candidates = build_candidates(leaf_count, row_count, forced_row_count, column_count, columns);
    if (static_cast<int>(candidates.size()) > constants::scip_root_set_packing_heuristic_candidate_limit) {
        std::ranges::nth_element(
            candidates,
            candidates.begin() + constants::scip_root_set_packing_heuristic_candidate_limit,
            better_candidate
        );
        candidates.resize(constants::scip_root_set_packing_heuristic_candidate_limit);
    }
    std::ranges::sort(candidates, better_candidate);
    if (candidates.empty())
        return {};

    std::vector<int> order(candidates.size());
    for (int i = 0; i < static_cast<int>(order.size()); ++i)
        order[i] = i;

    State best(leaf_count, row_count, forced_row_count, static_cast<int>(candidates.size()));
    const auto evaluate_order = [&](const std::vector<int>& current_order) {
        State state = build_state(leaf_count, row_count, forced_row_count, columns, candidates, current_order);
        if (forced_rows_covered(state) && state.saving > best.saving)
            best = std::move(state);
    };

    evaluate_order(order);

    auto lp_order = order;
    std::ranges::sort(lp_order, [&](const int lhs, const int rhs) {
        const auto& a = candidates[lhs];
        const auto& b = candidates[rhs];
        if (a.lp_value != b.lp_value)
            return a.lp_value > b.lp_value;
        if (a.saving != b.saving)
            return a.saving > b.saving;
        return a.tie_break < b.tie_break;
    });
    evaluate_order(lp_order);

    auto saving_order = order;
    std::ranges::sort(saving_order, [&](const int lhs, const int rhs) {
        const auto& a = candidates[lhs];
        const auto& b = candidates[rhs];
        if (a.saving != b.saving)
            return a.saving > b.saving;
        if (a.score != b.score)
            return a.score > b.score;
        return a.tie_break < b.tie_break;
    });
    evaluate_order(saving_order);

    auto warm_order = order;
    std::ranges::stable_sort(warm_order, [&](const int lhs, const int rhs) {
        const auto& a = candidates[lhs];
        const auto& b = candidates[rhs];
        if (a.warm_start != b.warm_start)
            return a.warm_start;
        return false;
    });
    if (std::ranges::any_of(candidates, [](const Candidate& candidate) { return candidate.warm_start; }))
        evaluate_order(warm_order);

    auto history_order = order;
    std::ranges::sort(history_order, [&](const int lhs, const int rhs) {
        const auto& a = candidates[lhs];
        const auto& b = candidates[rhs];
        if (a.history_value != b.history_value)
            return a.history_value > b.history_value;
        if (a.score != b.score)
            return a.score > b.score;
        return a.tie_break < b.tie_break;
    });
    evaluate_order(history_order);

    auto dual_order = order;
    std::ranges::sort(dual_order, [&](const int lhs, const int rhs) {
        const auto& a = candidates[lhs];
        const auto& b = candidates[rhs];
        const double a_margin = static_cast<double>(a.saving) -
                                constants::scip_root_set_packing_heuristic_dual_gain_weight * a.dual_cost;
        const double b_margin = static_cast<double>(b.saving) -
                                constants::scip_root_set_packing_heuristic_dual_gain_weight * b.dual_cost;
        if (a_margin != b_margin)
            return a_margin > b_margin;
        if (a.score != b.score)
            return a.score > b.score;
        return a.tie_break < b.tie_break;
    });
    evaluate_order(dual_order);

    for (int pass = 0; pass < constants::scip_root_set_packing_heuristic_randomized_passes; ++pass) {
        std::vector<double> priority(candidates.size(), 0.0);
        for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
            const double weight = std::max(1e-9, candidates[i].score);
            const double sample = deterministic_unit_interval(candidates[i].tie_break + static_cast<std::uint64_t>(pass));
            priority[i] = -std::log(sample) / weight;
        }
        auto randomized_order = order;
        std::ranges::sort(randomized_order, [&](const int lhs, const int rhs) {
            if (priority[lhs] != priority[rhs])
                return priority[lhs] < priority[rhs];
            return candidates[lhs].tie_break < candidates[rhs].tie_break;
        });
        evaluate_order(randomized_order);
    }

    if (forced_rows_covered(best))
        (void)improve_exact_neighborhood(columns, candidates, best);
    if (!forced_rows_covered(best))
        return {};
    return solution_from_state(candidates, best);
}

} // namespace maffe
