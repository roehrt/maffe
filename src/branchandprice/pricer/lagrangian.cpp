#include "branchandprice/pricer/lagrangian.hpp"
#include "maffe/common.hpp"
#include "util/partition_ops.hpp"
#include "util/constants.hpp"
#include "util/union_find.hpp"

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace maffe {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

struct BinaryTree {
    int leaves = 0;
    int root = -1;
    std::vector<int> parent;
    std::vector<EdgeState> edge_state;
    std::vector<std::array<int, 2>> child;
    std::vector<int> leaf_count;
    std::vector<std::vector<int>> ancestors;

    [[nodiscard]] int vertices() const {
        return static_cast<int>(parent.size());
    }
};

struct PairState {
    int u = -1;
    int v = -1;
    int up = -1;
    int keep_left = -1;
    int keep_right = -1;
    std::array<int, 2> both_left{{-1, -1}};
    std::array<int, 2> both_right{{-1, -1}};
};

enum class OpenChoiceKind : std::uint8_t {
    NONE,
    LEAF,
    BOTH,
    LEFT,
    RIGHT,
    UP,
};

struct OpenChoice {
    OpenChoiceKind kind = OpenChoiceKind::NONE;
    int next0 = -1;
    int next1 = -1;
    bool next0_multi = false;
    bool next1_multi = false;
};

enum class ClosedChoiceKind : std::uint8_t {
    LEAF,
    SPLIT,
    COMPONENT,
};

struct ClosedChoice {
    ClosedChoiceKind kind = ClosedChoiceKind::LEAF;
    int open_state = -1;
};

struct ReconstructedComponent {
    std::vector<int> leaves;
};

struct ForestResult {
    std::vector<std::vector<int>> partition;
    double primary_value = kInf;
};

[[nodiscard]] bool is_zero(const double value) {
    return std::abs(value) <= constants::lagrangian_dp_abs_tol;
}

[[nodiscard]] double compare_tolerance(const double lhs, const double rhs) {
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::max(constants::lagrangian_dp_abs_tol, constants::lagrangian_dp_rel_tol * scale);
}

[[nodiscard]] bool better_min(const double candidate, const double current) {
    if (!std::isfinite(candidate))
        return false;
    if (!std::isfinite(current))
        return true;
    return candidate < current - compare_tolerance(candidate, current);
}

[[nodiscard]] bool better_max(const double candidate, const double current) {
    if (!std::isfinite(candidate))
        return false;
    if (!std::isfinite(current))
        return true;
    return candidate > current + compare_tolerance(candidate, current);
}

[[nodiscard]] bool tied_score(const double candidate, const double current) {
    return std::isfinite(candidate) && std::isfinite(current) &&
        std::abs(candidate - current) <= compare_tolerance(candidate, current);
}

[[nodiscard]] int open_choice_rank(const OpenChoiceKind kind) {
    switch (kind) {
    case OpenChoiceKind::LEAF:
        return 0;
    case OpenChoiceKind::BOTH:
        return 1;
    case OpenChoiceKind::LEFT:
        return 2;
    case OpenChoiceKind::RIGHT:
        return 3;
    case OpenChoiceKind::UP:
        return 4;
    case OpenChoiceKind::NONE:
        return 5;
    }
    throw std::runtime_error("unknown open choice kind");
}

[[nodiscard]] bool prefer_open_choice(const OpenChoice& candidate, const OpenChoice& current) {
    return std::tuple{
        open_choice_rank(candidate.kind),
        candidate.next0,
        candidate.next1,
        candidate.next0_multi,
        candidate.next1_multi,
    } < std::tuple{
        open_choice_rank(current.kind),
        current.next0,
        current.next1,
        current.next0_multi,
        current.next1_multi,
    };
}

[[nodiscard]] int cover_choice_rank(const std::pair<int, int> choice) {
    return choice.first >= 0 ? 0 : 1;
}

[[nodiscard]] bool prefer_cover_choice(
    const std::pair<int, int> candidate,
    const std::pair<int, int> current
) {
    return std::tuple{cover_choice_rank(candidate), candidate.first, candidate.second} <
        std::tuple{cover_choice_rank(current), current.first, current.second};
}

void relax_open(
    const double candidate,
    const OpenChoice& next_choice,
    double& best,
    OpenChoice& choice
) {
    if (!better_min(candidate, best) && !(tied_score(candidate, best) && prefer_open_choice(next_choice, choice)))
        return;
    best = candidate;
    choice = next_choice;
}

[[nodiscard]] BinaryTree build_binary_tree(const Tree& tree) {
    if (tree.parent.empty() || tree.parent.size() != tree.edge_state.size())
        throw std::invalid_argument("invalid tree");

    BinaryTree out{
        .leaves = tree.leaves(),
        .root = tree.root(),
        .parent = tree.parent,
        .edge_state = tree.edge_state,
        .child = std::vector<std::array<int, 2>>(tree.vertices(), {-1, -1}),
        .leaf_count = std::vector<int>(tree.vertices()),
        .ancestors = std::vector<std::vector<int>>(tree.leaves()),
    };

    for (int u = 0; u < tree.vertices(); ++u) {
        const int p = tree.parent[u];
        if (p < 0)
            continue;

        auto& [left, right] = out.child[p];
        if (left < 0)
            left = u;
        else if (right < 0)
            right = u;
        else
            throw std::invalid_argument("tree is not binary");
    }

    for (int u = out.leaves; u < tree.vertices(); ++u) {
        auto& [left, right] = out.child[u];
        if (left < 0 || right < 0)
            throw std::invalid_argument("tree is not binary");
        if (right < left)
            std::swap(left, right);
    }

    for (int leaf = 0; leaf < out.leaves; ++leaf) {
        out.leaf_count[leaf] = 1;
        for (int u = leaf; u >= 0; u = out.parent[u])
            out.ancestors[leaf].push_back(u);
    }
    for (int u = out.leaves; u < tree.vertices(); ++u) {
        const auto [left, right] = out.child[u];
        out.leaf_count[u] = out.leaf_count[left] + out.leaf_count[right];
    }

    return out;
}

struct PairStateIndex {
    std::vector<PairState> states;
    std::vector<int> id;
    int second_vertices = 0;

    [[nodiscard]] int lookup(const int u, const int v) const {
        return id[static_cast<std::size_t>(u) * static_cast<std::size_t>(second_vertices) +
            static_cast<std::size_t>(v)];
    }
};

[[nodiscard]] PairStateIndex build_pair_state_index(const BinaryTree& first, const BinaryTree& second) {
    if (first.leaves != second.leaves)
        throw std::invalid_argument("trees do not share the same leaves");

    PairStateIndex out{
        .states = {},
        .id = std::vector<int>(
            static_cast<std::size_t>(first.vertices()) * static_cast<std::size_t>(second.vertices()),
            -1
        ),
        .second_vertices = second.vertices(),
    };
    out.states.reserve(static_cast<std::size_t>(first.vertices()) * static_cast<std::size_t>(second.vertices()));

    const auto has_state = [&](const int u, const int v) {
        return out.lookup(u, v) >= 0;
    };

    for (int u = 0; u < first.vertices(); ++u) {
        for (int v = 0; v < second.vertices(); ++v) {
            bool intersects = false;
            if (u < first.leaves && v < second.leaves) {
                intersects = u == v;
            } else if (u < first.leaves) {
                const auto [left, right] = second.child[v];
                intersects = has_state(u, left) || has_state(u, right);
            } else if (v < second.leaves) {
                const auto [left, right] = first.child[u];
                intersects = has_state(left, v) || has_state(right, v);
            } else {
                const auto [first_left, first_right] = first.child[u];
                const auto [second_left, second_right] = second.child[v];
                intersects =
                    has_state(first_left, second_left) ||
                    has_state(first_left, second_right) ||
                    has_state(first_right, second_left) ||
                    has_state(first_right, second_right);
            }
            if (!intersects)
                continue;

            const int state_id = static_cast<int>(out.states.size());
            out.id[static_cast<std::size_t>(u) * static_cast<std::size_t>(second.vertices()) +
                static_cast<std::size_t>(v)] = state_id;
            out.states.push_back(PairState{.u = u, .v = v});
        }
    }

    return out;
}

void validate_inputs(
    const AnnotatedInstance& instance,
    const std::vector<std::vector<double>>& vertex_duals,
    const std::vector<std::vector<double>>& edge_duals,
    const std::vector<std::vector<EdgeState>>& edge_states
) {
    if (instance.trees.size() < 2)
        throw std::invalid_argument("Lagrangian requires at least two trees");
    if (vertex_duals.size() != instance.trees.size() ||
        edge_duals.size() != instance.trees.size() ||
        edge_states.size() != instance.trees.size()) {
        throw std::invalid_argument("invalid Lagrangian dimensions");
    }

    const int leaves = instance.trees.front().leaves();
    for (int i = 0; i < static_cast<int>(instance.trees.size()); ++i) {
        if (instance.trees[i].leaves() != leaves)
            throw std::invalid_argument("all trees must share the same leaf set");
        if (static_cast<int>(vertex_duals[i].size()) != instance.trees[i].vertices())
            throw std::invalid_argument("invalid vertex dual dimensions");
        if (static_cast<int>(edge_duals[i].size()) != instance.trees[i].vertices())
            throw std::invalid_argument("invalid edge dual dimensions");
        if (static_cast<int>(edge_states[i].size()) != instance.trees[i].vertices())
            throw std::invalid_argument("invalid edge-state dimensions");
        for (int leaf = 0; leaf < instance.trees[i].leaves(); ++leaf) {
            if (!is_zero(vertex_duals[i][leaf]))
                throw std::invalid_argument("leaf vertex duals are not supported");
        }
    }
}

[[nodiscard]] double lambda_offset(
    const int leaves,
    std::span<const double> lambda
) {
    double offset = 0.0;
    for (int u = leaves; u < static_cast<int>(lambda.size()); ++u)
        offset += lambda[u];
    return offset;
}

[[nodiscard]] double forced_offset(std::span<const double> edge_duals) {
    double offset = 0.0;
    for (const double dual : edge_duals)
        offset += dual;
    return offset;
}

[[nodiscard]] double relaxed_vertex_offset(
    const AnnotatedInstance& instance,
    const std::vector<std::vector<double>>& vertex_duals
) {
    double offset = 0.0;
    for (int tree = 1; tree < static_cast<int>(instance.trees.size()); ++tree) {
        for (int u = instance.trees[tree].leaves(); u < instance.trees[tree].vertices(); ++u)
            offset += vertex_duals[tree][u];
    }
    return offset;
}

[[nodiscard]] double edge_offset(const std::vector<std::vector<double>>& edge_duals) {
    double offset = 0.0;
    for (const auto& tree_duals : edge_duals)
        offset += forced_offset(tree_duals);
    return offset;
}

template <class T>
void validate_optional_size(
    std::span<const T> values,
    const int expected,
    const char* const message
) {
    if (!values.empty() && static_cast<int>(values.size()) != expected)
        throw std::invalid_argument(message);
}

void merge_candidate_blocks(
    std::vector<std::vector<int>>& blocks,
    const std::vector<std::vector<int>>& extra
) {
    blocks.insert(blocks.end(), extra.begin(), extra.end());
    std::ranges::sort(blocks, detail::partition_block_less);
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
}

[[nodiscard]] LagrangianResult infeasible_result() {
    return {
        .leaf_partition = {},
        .candidate_blocks = {},
        .lower_bound = kInf,
    };
}

[[nodiscard]] LagrangianResult make_result(std::vector<std::vector<int>> partition, const double lower_bound) {
    return {
        .leaf_partition = partition,
        .candidate_blocks = partition,
        .lower_bound = lower_bound,
    };
}

class LagrangianForestOracle {
public:
    LagrangianForestOracle(const Tree& first, const Tree& second)
        : first_(build_binary_tree(first)),
          second_(build_binary_tree(second)),
          lambda_(second_.vertices()),
          cached_first_edge_states_(first_.vertices(), EdgeState::UNKNOWN),
          state_begin_by_first_(static_cast<std::size_t>(first_.vertices()) + 1U, 0),
          best_closable_open_(first_.vertices()),
          best_closable_open_state_(first_.vertices(), -1),
          closed_score_(first_.vertices()),
          closed_choice_(first_.vertices()),
          can_take_both_(first_.vertices(), 0),
          can_take_left_(first_.vertices(), 0),
          can_take_right_(first_.vertices(), 0),
          split_allowed_(first_.vertices(), 0),
          can_split_second_(second_.vertices(), 0),
          can_move_up_second_(second_.vertices(), 0),
          take_both_base_(first_.vertices(), 0.0),
          take_left_base_(first_.vertices(), 0.0),
          take_right_base_(first_.vertices(), 0.0),
          split_second_base_(second_.vertices(), 0.0),
          move_up_base_(second_.vertices(), 0.0) {
        if (first_.leaves != second_.leaves)
            throw std::invalid_argument("trees do not share the same leaves");
        precompute_second_tree_flags();
        build_states();
        open_single_score_.resize(states_.size());
        open_single_choice_.resize(states_.size());
        open_score_.resize(states_.size());
        open_choice_.resize(states_.size());
        closable_open_score_.resize(states_.size());
        closable_open_choice_.resize(states_.size());
    }

    [[nodiscard]] std::optional<ForestResult> solve(
        std::span<const double> second_internal_lambda,
        std::span<const EdgeState> first_edge_states,
        std::span<const double> first_forced_edge_weight,
        std::span<const double> second_forced_edge_weight
    ) {
        if (!run_primary(
                second_internal_lambda,
                first_edge_states,
                first_forced_edge_weight,
                second_forced_edge_weight
            )) {
            return std::nullopt;
        }
        return reconstruct_forest();
    }

private:
    void precompute_second_tree_flags() {
        for (int v = 0; v < second_.vertices(); ++v) {
            const int parent = second_.parent[v];
            if (parent >= 0 && second_.edge_state[v] != EdgeState::CUT) {
                const auto [left, right] = second_.child[parent];
                const int sibling = left == v ? right : left;
                can_move_up_second_[v] = second_.edge_state[sibling] != EdgeState::FORCED ? 1 : 0;
            }
            if (v < second_.leaves)
                continue;
            const auto [left, right] = second_.child[v];
            can_split_second_[v] = second_.edge_state[left] != EdgeState::CUT &&
                second_.edge_state[right] != EdgeState::CUT ? 1 : 0;
        }
    }

    void rebuild_first_tree_flags(std::span<const EdgeState> first_edge_states) {
        if (static_cast<int>(first_edge_states.size()) != first_.vertices())
            throw std::invalid_argument("invalid first-tree edge states");

        for (int u = first_.leaves; u < first_.vertices(); ++u) {
            const auto [left, right] = first_.child[u];
            can_take_both_[u] = first_edge_states[left] != EdgeState::CUT &&
                first_edge_states[right] != EdgeState::CUT ? 1 : 0;
            can_take_left_[u] = first_edge_states[left] != EdgeState::CUT &&
                first_edge_states[right] != EdgeState::FORCED ? 1 : 0;
            can_take_right_[u] = first_edge_states[right] != EdgeState::CUT &&
                first_edge_states[left] != EdgeState::FORCED ? 1 : 0;
            split_allowed_[u] = first_edge_states[left] != EdgeState::FORCED &&
                first_edge_states[right] != EdgeState::FORCED ? 1 : 0;
        }
    }

    void build_states() {
        auto state_index = build_pair_state_index(first_, second_);
        states_ = std::move(state_index.states);
        const auto lookup = [&](const int u, const int v) {
            return state_index.lookup(u, v);
        };

        for (const auto& state : states_)
            ++state_begin_by_first_[static_cast<std::size_t>(state.u) + 1U];
        for (std::size_t i = 1; i < state_begin_by_first_.size(); ++i)
            state_begin_by_first_[i] += state_begin_by_first_[i - 1];

        state_order_.resize(states_.size());
        auto next = state_begin_by_first_;
        for (int state = 0; state < static_cast<int>(states_.size()); ++state)
            state_order_[next[static_cast<std::size_t>(states_[state].u)]++] = state;

        for (int u = 0; u < first_.vertices(); ++u) {
            const auto begin = state_order_.begin() + static_cast<std::ptrdiff_t>(state_begin_by_first_[u]);
            const auto end = state_order_.begin() + static_cast<std::ptrdiff_t>(state_begin_by_first_[u + 1]);
            std::ranges::sort(begin, end, [&](const int a, const int b) {
                return states_[a].v < states_[b].v;
            });
        }

        for (auto& state : states_) {
            const int parent = second_.parent[state.v];
            if (parent >= 0)
                state.up = lookup(state.u, parent);

            if (state.u < first_.leaves)
                continue;

            const auto [left, right] = first_.child[state.u];
            state.keep_left = lookup(left, state.v);
            state.keep_right = lookup(right, state.v);
            if (state.v < second_.leaves)
                continue;

            const auto [a, b] = second_.child[state.v];
            state.both_left[0] = lookup(left, a);
            state.both_right[0] = lookup(right, b);
            if (state.both_left[0] < 0 || state.both_right[0] < 0)
                state.both_left[0] = state.both_right[0] = -1;

            state.both_left[1] = lookup(left, b);
            state.both_right[1] = lookup(right, a);
            if (state.both_left[1] < 0 || state.both_right[1] < 0)
                state.both_left[1] = state.both_right[1] = -1;
        }
    }

    void load_lambda(std::span<const double> second_internal_lambda) {
        if (static_cast<int>(second_internal_lambda.size()) == second_.vertices()) {
            std::copy(second_internal_lambda.begin(), second_internal_lambda.end(), lambda_.begin());
            return;
        }

        if (static_cast<int>(second_internal_lambda.size()) == second_.vertices() - second_.leaves) {
            std::fill_n(lambda_.begin(), second_.leaves, 0.0);
            for (int u = second_.leaves; u < second_.vertices(); ++u)
                lambda_[u] = second_internal_lambda[u - second_.leaves];
            return;
        }

        throw std::invalid_argument("invalid second-tree Lagrangian multipliers");
    }

    void prepare_transition_costs(
        std::span<const double> first_forced_edge_weight,
        std::span<const double> second_forced_edge_weight
    ) {
        const bool has_first_forced = !first_forced_edge_weight.empty();
        const bool has_second_forced = !second_forced_edge_weight.empty();

        for (int u = first_.leaves; u < first_.vertices(); ++u) {
            const auto [left, right] = first_.child[u];
            const double left_forced = has_first_forced ? first_forced_edge_weight[left] : 0.0;
            const double right_forced = has_first_forced ? first_forced_edge_weight[right] : 0.0;
            take_both_base_[u] = -left_forced - right_forced;
            take_left_base_[u] = -left_forced;
            take_right_base_[u] = -right_forced;
        }

        for (int v = 0; v < second_.vertices(); ++v) {
            const double forced = has_second_forced ? second_forced_edge_weight[v] : 0.0;
            const int parent = second_.parent[v];
            if (parent >= 0)
                move_up_base_[v] = lambda_[parent] - forced;
            if (v < second_.leaves)
                continue;
            const auto [left, right] = second_.child[v];
            const double left_forced = has_second_forced ? second_forced_edge_weight[left] : 0.0;
            const double right_forced = has_second_forced ? second_forced_edge_weight[right] : 0.0;
            split_second_base_[v] = lambda_[v] - left_forced - right_forced;
        }
    }

    [[nodiscard]] bool run_primary(
        std::span<const double> second_internal_lambda,
        std::span<const EdgeState> first_edge_states,
        std::span<const double> first_forced_edge_weight,
        std::span<const double> second_forced_edge_weight
    ) {
        load_lambda(second_internal_lambda);
        if (!first_flags_valid_ ||
            !std::equal(
                cached_first_edge_states_.begin(),
                cached_first_edge_states_.end(),
                first_edge_states.begin(),
                first_edge_states.end()
            )) {
            rebuild_first_tree_flags(first_edge_states);
            cached_first_edge_states_.assign(first_edge_states.begin(), first_edge_states.end());
            first_flags_valid_ = true;
        }
        validate_optional_size(first_forced_edge_weight, first_.vertices(), "invalid first-tree forced-edge weights");
        validate_optional_size(second_forced_edge_weight, second_.vertices(), "invalid second-tree forced-edge weights");
        prepare_transition_costs(
            first_forced_edge_weight,
            second_forced_edge_weight
        );
        std::fill(open_single_score_.begin(), open_single_score_.end(), kInf);
        std::fill(open_score_.begin(), open_score_.end(), kInf);
        std::fill(closable_open_score_.begin(), closable_open_score_.end(), kInf);

        for (int u = 0; u < first_.vertices(); ++u) {
            const bool leaf = u < first_.leaves;
            int left = -1;
            int right = -1;
            double left_closed = kInf;
            double right_closed = kInf;
            bool allow_both = false;
            bool allow_left = false;
            bool allow_right = false;
            bool allow_split = false;
            double left_base = 0.0;
            double right_base = 0.0;
            if (!leaf) {
                std::tie(left, right) = first_.child[u];
                left_closed = closed_score_[left];
                right_closed = closed_score_[right];
                allow_both = can_take_both_[u] != 0;
                allow_left = can_take_left_[u] != 0 && right_closed < kInf;
                allow_right = can_take_right_[u] != 0 && left_closed < kInf;
                allow_split = split_allowed_[u] != 0;
                left_base = right_closed + take_left_base_[u];
                right_base = left_closed + take_right_base_[u];
            }
            best_closable_open_[u] = kInf;
            best_closable_open_state_[u] = -1;

            for (std::size_t pos = state_begin_by_first_[u]; pos < state_begin_by_first_[u + 1]; ++pos) {
                const int id = state_order_[pos];
                const auto& state = states_[id];
                double best_single = open_single_score_[id];
                OpenChoice best_single_choice =
                    std::isfinite(best_single) ? open_single_choice_[id] : OpenChoice{};
                double best = open_score_[id];
                OpenChoice best_choice =
                    std::isfinite(best) ? open_choice_[id] : OpenChoice{};
                double closable = closable_open_score_[id];
                OpenChoice closable_choice =
                    std::isfinite(closable) ? closable_open_choice_[id] : OpenChoice{};
                const int state_v = state.v;
                const int keep_left = state.keep_left;
                const int keep_right = state.keep_right;
                const int state_up = state.up;
                const bool can_move_up = can_move_up_second_[state_v] != 0;

                if (leaf) {
                    if (state_v == u) {
                        relax_open(
                            lambda_[u],
                            OpenChoice{.kind = OpenChoiceKind::LEAF},
                            best_single,
                            best_single_choice
                        );
                    }
                } else {
                    if (allow_both && can_split_second_[state_v] != 0) {
                        const double both_base = take_both_base_[u] + split_second_base_[state_v];
                        const auto relax_both_candidate = [&](const int left_id,
                                                              const int right_id,
                                                              const bool left_multi,
                                                              const bool right_multi,
                                                              const double left_value,
                                                              const double right_value) {
                            if (left_value >= kInf || right_value >= kInf)
                                return;
                            const OpenChoice next{
                                .kind = OpenChoiceKind::BOTH,
                                .next0 = left_id,
                                .next1 = right_id,
                                .next0_multi = left_multi,
                                .next1_multi = right_multi,
                            };
                            const double score = both_base + left_value + right_value;
                            relax_open(score, next, best, best_choice);
                            relax_open(score, next, closable, closable_choice);
                        };
                        for (int side = 0; side < 2; ++side) {
                            const int left_id = state.both_left[side];
                            const int right_id = state.both_right[side];
                            if (left_id < 0 || right_id < 0)
                                continue;

                            const double left_single = open_single_score_[left_id];
                            const double left_multi = open_score_[left_id];
                            const double right_single = open_single_score_[right_id];
                            const double right_multi = open_score_[right_id];
                            relax_both_candidate(left_id, right_id, false, false, left_single, right_single);
                            relax_both_candidate(left_id, right_id, false, true, left_single, right_multi);
                            relax_both_candidate(left_id, right_id, true, false, left_multi, right_single);
                            relax_both_candidate(left_id, right_id, true, true, left_multi, right_multi);
                        }
                    }

                    if (allow_left && keep_left >= 0) {
                        const double left_single = open_single_score_[keep_left];
                        if (left_single < kInf) {
                            relax_open(
                                left_base + left_single,
                                OpenChoice{
                                    .kind = OpenChoiceKind::LEFT,
                                    .next0 = keep_left,
                                    .next0_multi = false,
                                },
                                best_single,
                                best_single_choice
                            );
                        }

                        const double left_score = open_score_[keep_left];
                        if (left_score < kInf) {
                            relax_open(
                                left_base + left_score,
                                OpenChoice{
                                    .kind = OpenChoiceKind::LEFT,
                                    .next0 = keep_left,
                                    .next0_multi = true,
                                },
                                best,
                                best_choice
                            );
                        }
                    }

                    if (allow_right && keep_right >= 0) {
                        const double right_single = open_single_score_[keep_right];
                        if (right_single < kInf) {
                            relax_open(
                                right_base + right_single,
                                OpenChoice{
                                    .kind = OpenChoiceKind::RIGHT,
                                    .next0 = keep_right,
                                    .next0_multi = false,
                                },
                                best_single,
                                best_single_choice
                            );
                        }

                        const double right_score = open_score_[keep_right];
                        if (right_score < kInf) {
                            relax_open(
                                right_base + right_score,
                                OpenChoice{
                                    .kind = OpenChoiceKind::RIGHT,
                                    .next0 = keep_right,
                                    .next0_multi = true,
                                },
                                best,
                                best_choice
                            );
                        }
                    }
                }

                open_single_score_[id] = best_single;
                open_single_choice_[id] = best_single_choice;
                open_score_[id] = best;
                open_choice_[id] = best_choice;
                closable_open_score_[id] = closable;
                closable_open_choice_[id] = closable_choice;
                if (better_min(closable, best_closable_open_[u]) ||
                    (tied_score(closable, best_closable_open_[u]) &&
                     (best_closable_open_state_[u] < 0 || id < best_closable_open_state_[u]))) {
                    best_closable_open_[u] = closable;
                    best_closable_open_state_[u] = id;
                }

                if (best_single < kInf && state_up >= 0 && can_move_up) {
                    relax_open(
                        move_up_base_[state_v] + best_single,
                        OpenChoice{
                            .kind = OpenChoiceKind::UP,
                            .next0 = id,
                            .next0_multi = false,
                        },
                        open_single_score_[state_up],
                        open_single_choice_[state_up]
                    );
                }

                if (best < kInf && state_up >= 0 && can_move_up) {
                    relax_open(
                        move_up_base_[state_v] + best,
                        OpenChoice{
                            .kind = OpenChoiceKind::UP,
                            .next0 = id,
                            .next0_multi = true,
                        },
                        open_score_[state_up],
                        open_choice_[state_up]
                    );
                }
            }

            if (u != first_.root && first_edge_states[u] == EdgeState::FORCED) {
                closed_score_[u] = kInf;
                continue;
            }

            if (leaf) {
                closed_score_[u] = 1.0;
                closed_choice_[u] = ClosedChoice{.kind = ClosedChoiceKind::LEAF};
                continue;
            }

            double split = kInf;
            if (allow_split)
                split = closed_score_[left] + closed_score_[right];
            const double component = best_closable_open_[u] + 1.0;
            if (!std::isfinite(split) && !std::isfinite(component)) {
                closed_score_[u] = kInf;
                continue;
            }
            if (better_min(split, component)) {
                closed_score_[u] = split;
                closed_choice_[u] = ClosedChoice{.kind = ClosedChoiceKind::SPLIT};
            } else {
                closed_score_[u] = component;
                closed_choice_[u] = ClosedChoice{
                    .kind = ClosedChoiceKind::COMPONENT,
                    .open_state = best_closable_open_state_[u],
                };
            }
        }

        return std::isfinite(closed_score_[first_.root]);
    }

    [[nodiscard]] std::optional<ForestResult> reconstruct_forest() {
        if (!std::isfinite(closed_score_[first_.root]))
            return std::nullopt;

        std::vector<ReconstructedComponent> components;
        components.reserve(first_.leaves);
        std::vector<char> leaf_seen(first_.leaves, 0);

        const auto append_component = [&](ReconstructedComponent component) {
            std::ranges::sort(component.leaves);
            const auto unique_end = std::unique(component.leaves.begin(), component.leaves.end());
            if (unique_end != component.leaves.end())
                throw std::runtime_error("reconstructed duplicate leaf in component");
            for (const int leaf : component.leaves) {
                if (leaf < 0 || leaf >= first_.leaves)
                    throw std::runtime_error("reconstructed invalid leaf");
                if (leaf_seen[leaf] != 0)
                    throw std::runtime_error("reconstructed overlapping components");
                leaf_seen[leaf] = 1;
            }
            components.push_back(std::move(component));
        };

        const auto reconstruct_open = [&](const auto& self,
                                          const int id,
                                          const bool multi,
                                          const bool closable_chain,
                                          const auto& reconstruct_closed_impl,
                                          ReconstructedComponent& component) -> int {
            const auto& state = states_[id];
            const auto& choice = multi
                ? (closable_chain ? closable_open_choice_[id] : open_choice_[id])
                : open_single_choice_[id];

            switch (choice.kind) {
            case OpenChoiceKind::LEAF:
                if (multi || state.u >= first_.leaves || state.v != state.u)
                    throw std::runtime_error("invalid open LEAF reconstruction");
                component.leaves.push_back(state.u);
                return 1;

            case OpenChoiceKind::BOTH:
                if (!multi || state.u < first_.leaves || state.v < second_.leaves)
                    throw std::runtime_error("invalid open BOTH reconstruction");
                if (choice.next0 < 0 || choice.next1 < 0)
                    throw std::runtime_error("missing BOTH child");
                return self(self, choice.next0, choice.next0_multi, false, reconstruct_closed_impl, component) +
                    self(self, choice.next1, choice.next1_multi, false, reconstruct_closed_impl, component);

            case OpenChoiceKind::LEFT: {
                if (state.u < first_.leaves || choice.next0 < 0)
                    throw std::runtime_error("invalid open LEFT reconstruction");
                const int right = first_.child[state.u][1];
                return self(self, choice.next0, choice.next0_multi, closable_chain, reconstruct_closed_impl, component) +
                    reconstruct_closed_impl(reconstruct_closed_impl, right, self);
            }

            case OpenChoiceKind::RIGHT: {
                if (state.u < first_.leaves || choice.next0 < 0)
                    throw std::runtime_error("invalid open RIGHT reconstruction");
                const int left = first_.child[state.u][0];
                return reconstruct_closed_impl(reconstruct_closed_impl, left, self) +
                    self(self, choice.next0, choice.next0_multi, closable_chain, reconstruct_closed_impl, component);
            }

            case OpenChoiceKind::UP: {
                if (choice.next0 < 0)
                    throw std::runtime_error("missing UP child");
                const auto& child_state = states_[choice.next0];
                if (child_state.up != id)
                    throw std::runtime_error("invalid UP child relation");
                if (state.v >= second_.leaves) {
                    const auto [left, right] = second_.child[state.v];
                    if (child_state.v != left && child_state.v != right)
                        throw std::runtime_error("invalid UP reconstruction");
                }
                return self(self, choice.next0, choice.next0_multi, closable_chain, reconstruct_closed_impl, component);
            }

            case OpenChoiceKind::NONE:
                break;
            }

            throw std::runtime_error("failed to reconstruct open state");
        };

        const auto reconstruct_closed = [&](const auto& self,
                                            const int u,
                                            const auto& reconstruct_open_impl) -> int {
            const auto choice = closed_choice_[u];
            switch (choice.kind) {
            case ClosedChoiceKind::LEAF:
                if (u >= first_.leaves)
                    throw std::runtime_error("invalid closed LEAF reconstruction");
                append_component(ReconstructedComponent{.leaves = {u}});
                return 1;

            case ClosedChoiceKind::COMPONENT: {
                const int id = choice.open_state;
                if (id < 0)
                    throw std::runtime_error("missing closed component state");
                ReconstructedComponent component{.leaves = {}};
                component.leaves.reserve(first_.leaf_count[u]);
                const int covered = reconstruct_open_impl(
                    reconstruct_open_impl,
                    id,
                    true,
                    true,
                    self,
                    component
                );
                if (covered != first_.leaf_count[u])
                    throw std::runtime_error("component reconstruction dropped leaves");
                append_component(std::move(component));
                return covered;
            }

            case ClosedChoiceKind::SPLIT: {
                if (u < first_.leaves)
                    throw std::runtime_error("invalid closed SPLIT reconstruction");
                const auto [left, right] = first_.child[u];
                const int covered =
                    self(self, left, reconstruct_open_impl) +
                    self(self, right, reconstruct_open_impl);
                if (covered != first_.leaf_count[u])
                    throw std::runtime_error("split reconstruction dropped leaves");
                return covered;
            }
            }

            throw std::runtime_error("failed to reconstruct closed state");
        };

        const int covered = reconstruct_closed(reconstruct_closed, first_.root, reconstruct_open);
        if (covered != first_.leaf_count[first_.root])
            throw std::runtime_error("forest reconstruction dropped root leaves");
        if (std::ranges::any_of(leaf_seen, [](const char seen) { return seen == 0; }))
            throw std::runtime_error("forest reconstruction did not cover all leaves");

        ForestResult result{
            .partition = {},
            .primary_value = closed_score_[first_.root],
        };
        result.partition.reserve(components.size());
        for (auto& component : components)
            result.partition.push_back(std::move(component.leaves));
        std::ranges::sort(result.partition, detail::partition_block_less);
        return result;
    }

    BinaryTree first_;
    BinaryTree second_;
    std::vector<double> lambda_;
    std::vector<EdgeState> cached_first_edge_states_;
    bool first_flags_valid_ = false;
    std::vector<PairState> states_;
    std::vector<int> state_order_;
    std::vector<std::size_t> state_begin_by_first_;
    std::vector<double> open_single_score_;
    std::vector<OpenChoice> open_single_choice_;
    std::vector<double> open_score_;
    std::vector<OpenChoice> open_choice_;
    std::vector<double> closable_open_score_;
    std::vector<OpenChoice> closable_open_choice_;
    std::vector<double> best_closable_open_;
    std::vector<int> best_closable_open_state_;
    std::vector<double> closed_score_;
    std::vector<ClosedChoice> closed_choice_;
    std::vector<char> can_take_both_;
    std::vector<char> can_take_left_;
    std::vector<char> can_take_right_;
    std::vector<char> split_allowed_;
    std::vector<char> can_split_second_;
    std::vector<char> can_move_up_second_;
    std::vector<double> take_both_base_;
    std::vector<double> take_left_base_;
    std::vector<double> take_right_base_;
    std::vector<double> split_second_base_;
    std::vector<double> move_up_base_;
};

class LeafPairLca {
public:
    explicit LeafPairLca(const BinaryTree& tree)
        : leaves_(tree.leaves),
          table_(static_cast<std::size_t>(tree.leaves) * static_cast<std::size_t>(tree.leaves), -1) {
        std::vector<char> marked(tree.vertices(), 0);
        for (int a = 0; a < leaves_; ++a) {
            for (const int u : tree.ancestors[a])
                marked[u] = 1;
            for (int b = a; b < leaves_; ++b) {
                int lca = -1;
                for (const int u : tree.ancestors[b]) {
                    if (marked[u] != 0) {
                        lca = u;
                        break;
                    }
                }
                if (lca < 0)
                    throw std::runtime_error("failed to compute leaf-pair LCA");
                table_[index(a, b)] = lca;
                table_[index(b, a)] = lca;
            }
            for (const int u : tree.ancestors[a])
                marked[u] = 0;
        }
    }

    [[nodiscard]] int lca(const int a, const int b) const {
        if (a < 0 || a >= leaves_ || b < 0 || b >= leaves_)
            throw std::out_of_range("leaf-pair LCA query out of range");
        return table_[index(a, b)];
    }

private:
    [[nodiscard]] std::size_t index(const int a, const int b) const {
        return static_cast<std::size_t>(a) * static_cast<std::size_t>(leaves_) + static_cast<std::size_t>(b);
    }

    int leaves_ = 0;
    std::vector<int> table_;
};

template <class T>
class Fenwick {
public:
    Fenwick() = default;
    explicit Fenwick(const int size)
        : bit_(static_cast<std::size_t>(size) + 1U, T{}) {}

    void clear() {
        std::fill(bit_.begin(), bit_.end(), T{});
    }

    void add(const int index, const T delta) {
        for (int i = index + 1; i < static_cast<int>(bit_.size()); i += i & -i)
            bit_[static_cast<std::size_t>(i)] += delta;
    }

    void add_range(const int begin, const int end, const T delta) {
        if (begin >= end)
            return;
        add(begin, delta);
        if (end < static_cast<int>(bit_.size()) - 1)
            add(end, -delta);
    }

    [[nodiscard]] T point(const int index) const {
        T value{};
        for (int i = index + 1; i > 0; i -= i & -i)
            value += bit_[static_cast<std::size_t>(i)];
        return value;
    }

private:
    std::vector<T> bit_;
};

struct ConnectedLeafPair {
    int a = -1;
    int b = -1;
    int root0 = -1;
};

struct TransitionCandidate {
    int c = -1;
    int ancestor0 = -1;
};

class MultiTreeLagrangianOracle {
public:
    explicit MultiTreeLagrangianOracle(const AnnotatedInstance& instance)
        : tree_count_(static_cast<int>(instance.trees.size())),
          node_count_(instance.trees.front().vertices()),
          leaf_count_(instance.trees.front().leaves()),
          trees_(),
          lca_(),
          climb_valid_(),
          connected_leaf_pairs_sorted_(),
          connected_in_other_trees_(
              static_cast<std::size_t>(leaf_count_) * static_cast<std::size_t>(leaf_count_),
              0
          ),
          valid_dp_trans_left_(static_cast<std::size_t>(leaf_count_) * static_cast<std::size_t>(leaf_count_)),
          valid_dp_trans_right_(static_cast<std::size_t>(leaf_count_) * static_cast<std::size_t>(leaf_count_)),
          dynamic_tree0_climb_valid_(
              static_cast<std::size_t>(node_count_) * static_cast<std::size_t>(node_count_),
              0
          ),
          dynamic_tree0_connectivity_(node_count_, -1),
          connected_in_tree0_(
              static_cast<std::size_t>(leaf_count_) * static_cast<std::size_t>(leaf_count_),
              0
          ),
          tree0_tin_(node_count_, -1),
          tree0_tout_(node_count_, -1),
          side_cost_add_(node_count_),
          side_cost_block_(node_count_),
          path_cost_(
              tree_count_,
              std::vector<double>(node_count_, 0.0)
          ),
          dp_(static_cast<std::size_t>(leaf_count_) * static_cast<std::size_t>(leaf_count_), -kInf),
          left_transition_(
              static_cast<std::size_t>(leaf_count_) * static_cast<std::size_t>(leaf_count_),
              {-1, -1}
          ),
          right_transition_(
              static_cast<std::size_t>(leaf_count_) * static_cast<std::size_t>(leaf_count_),
              {-1, -1}
          ),
          best_subtree_cover_(node_count_, -kInf),
          best_cover_choice_(node_count_, {-1, -1}),
          reconstructed_(node_count_, 0),
          cached_tree0_states_(node_count_, EdgeState::UNKNOWN) {
        trees_.reserve(tree_count_);
        lca_.reserve(tree_count_);
        climb_valid_.reserve(tree_count_);
        std::vector<UnionFind> connectivity;
        connectivity.reserve(tree_count_);

        for (int tree = 0; tree < tree_count_; ++tree) {
            const auto& input = instance.trees[tree];
            if (input.vertices() != node_count_ || input.leaves() != leaf_count_)
                throw std::invalid_argument("all trees must have the same shape dimensions");

            trees_.push_back(build_binary_tree(input));
            lca_.emplace_back(trees_.back());
            climb_valid_.push_back(build_climb_valid(trees_.back()));
            connectivity.emplace_back(node_count_);
            for (int u = 0; u < input.vertices(); ++u) {
                const int parent = input.parent[u];
                if (parent >= 0 && input.edge_state[u] != EdgeState::CUT)
                    connectivity.back().join(u, parent);
            }
        }

        const auto connected_in_all_other_trees = [&](const int a, const int b) {
            for (int tree = 1; tree < tree_count_; ++tree) {
                auto& uf = connectivity[tree];
                if (!uf.same_set(a, b))
                    return false;
            }
            return true;
        };

        connected_leaf_pairs_sorted_.reserve(static_cast<std::size_t>(leaf_count_) * (leaf_count_ + 1) / 2U);
        for (int a = 0; a < leaf_count_; ++a) {
            for (int b = a; b < leaf_count_; ++b) {
                if (!connected_in_all_other_trees(a, b))
                    continue;
                connected_in_other_trees_[leaf_pair_index(a, b)] = 1;
                connected_in_other_trees_[leaf_pair_index(b, a)] = 1;
                connected_leaf_pairs_sorted_.push_back(ConnectedLeafPair{
                    .a = a,
                    .b = b,
                    .root0 = lca_[0].lca(a, b),
                });
            }
        }
        std::ranges::sort(connected_leaf_pairs_sorted_, [&](const auto& lhs, const auto& rhs) {
            return lhs.root0 < rhs.root0;
        });

        std::vector<int> current_lca(static_cast<std::size_t>(tree_count_), -1);
        const auto append_transitions = [&](const int anchor,
                                            const int other,
                                            std::vector<TransitionCandidate>& transitions) {
            const int root0 = lca_[0].lca(anchor, other);
            for (int tree = 1; tree < tree_count_; ++tree)
                current_lca[tree] = lca_[tree].lca(anchor, other);

            for (int c = anchor; c < leaf_count_; ++c) {
                if (!other_trees_connected(anchor, c))
                    continue;
                const int ancestor0 = lca_[0].lca(anchor, c);
                if (root0 <= ancestor0)
                    continue;

                bool valid = true;
                for (int tree = 1; tree < tree_count_; ++tree) {
                    const int previous = lca_[tree].lca(anchor, c);
                    if (current_lca[tree] <= previous ||
                        climb_valid_[tree][matrix_index(previous, current_lca[tree])] == 0) {
                        valid = false;
                        break;
                    }
                }
                if (valid) {
                    transitions.push_back(TransitionCandidate{
                        .c = c,
                        .ancestor0 = ancestor0,
                    });
                }
            }
        };

        for (int a = 0; a < leaf_count_; ++a) {
            for (int b = a + 1; b < leaf_count_; ++b) {
                if (!other_trees_connected(a, b))
                    continue;
                const std::size_t pair = leaf_pair_index(a, b);
                append_transitions(a, b, valid_dp_trans_left_[pair]);
                append_transitions(b, a, valid_dp_trans_right_[pair]);
            }
        }
        build_tree0_euler();
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    [[nodiscard]] LagrangianResult solve(
        const std::vector<std::vector<double>>& vertex_duals,
        const std::vector<std::vector<double>>& edge_duals,
        std::span<const EdgeState> tree0_states
    ) {
        if (static_cast<int>(tree0_states.size()) != node_count_)
            throw std::invalid_argument("invalid first-tree edge-state dimensions");

        if (!tree0_dynamic_valid_ ||
            !std::equal(
                cached_tree0_states_.begin(),
                cached_tree0_states_.end(),
                tree0_states.begin(),
                tree0_states.end()
            )) {
            rebuild_tree0_dynamic_state(tree0_states);
            cached_tree0_states_.assign(tree0_states.begin(), tree0_states.end());
            tree0_dynamic_valid_ = true;
        }

        side_cost_add_.clear();
        side_cost_block_.clear();
        for (int tree = 0; tree < tree_count_; ++tree) {
            for (int node = node_count_ - 2; node >= 0; --node) {
                const int parent = trees_[tree].parent[node];
                const double path_vertex = tree == 0 ? 0.0 : -vertex_duals[tree][parent];
                path_cost_[tree][node] = path_cost_[tree][parent] + path_vertex + edge_duals[tree][node];
            }
        }

        std::fill(dp_.begin(), dp_.end(), -kInf);
        std::fill(left_transition_.begin(), left_transition_.end(), std::pair{-1, -1});
        std::fill(right_transition_.begin(), right_transition_.end(), std::pair{-1, -1});
        std::fill(best_subtree_cover_.begin(), best_subtree_cover_.end(), -kInf);
        std::fill(best_cover_choice_.begin(), best_cover_choice_.end(), std::pair{-1, -1});

        const auto side_cost_at = [&](const int node) {
            if (side_cost_block_.point(tree0_tin_[node]) > 0)
                return -kInf;
            return side_cost_add_.point(tree0_tin_[node]);
        };

        const auto update_side_cost = [&](const int handled) {
            const auto [left, right] = trees_[0].child[handled];
            for (const auto& [start, sibling] : std::array{
                     std::pair{left, right},
                     std::pair{right, left},
                 }) {
                const bool sibling_forced = tree0_states[sibling] == EdgeState::FORCED;
                const double sibling_cover = best_subtree_cover_[sibling];
                if (sibling_forced || !std::isfinite(sibling_cover)) {
                    side_cost_block_.add_range(tree0_tin_[start], tree0_tout_[start], 1);
                } else {
                    side_cost_add_.add_range(tree0_tin_[start], tree0_tout_[start], sibling_cover);
                }
            }
        };

        int pair_index = 0;
        const auto best_transition = [&](const int anchor,
                                         const int root0,
                                         const std::vector<TransitionCandidate>& transitions) {
            double best = -kInf;
            std::pair<int, int> choice{-1, -1};
            const std::size_t row_offset =
                static_cast<std::size_t>(anchor) * static_cast<std::size_t>(leaf_count_);
            for (const auto transition : transitions) {
                const double subproblem = dp_[row_offset + static_cast<std::size_t>(transition.c)];
                if (!std::isfinite(subproblem))
                    continue;
                if (dynamic_tree0_climb_valid_[matrix_index(transition.ancestor0, root0)] == 0)
                    continue;
                const double side = side_cost_at(transition.ancestor0);
                if (!std::isfinite(side))
                    continue;
                const double candidate = subproblem + side;
                const std::pair<int, int> next_choice{anchor, transition.c};
                if (better_max(candidate, best) ||
                    (tied_score(candidate, best) && next_choice < choice)) {
                    best = candidate;
                    choice = next_choice;
                }
            }
            return std::pair{best, choice};
        };

        for (int node0 = 0; node0 < node_count_; ++node0) {
            for (; pair_index < static_cast<int>(connected_leaf_pairs_sorted_.size()); ++pair_index) {
                const auto pair = connected_leaf_pairs_sorted_[pair_index];
                const int a = pair.a;
                const int b = pair.b;
                if (pair.root0 > node0)
                    break;
                if (!tree0_leaves_connected(a, b))
                    continue;

                if (a == b) {
                    best_subtree_cover_[node0] = 0.0;
                    best_cover_choice_[node0] = {a, a};
                    double value = 0.0;
                    for (int tree = 0; tree < tree_count_; ++tree)
                        value += path_cost_[tree][a];
                    dp_[leaf_pair_index(a, a)] = value;
                    continue;
                }

                const std::size_t leaf_pair = leaf_pair_index(a, b);
                const auto [best_left, best_left_choice] =
                    best_transition(a, pair.root0, valid_dp_trans_left_[leaf_pair]);
                const auto [best_right, best_right_choice] =
                    best_transition(b, pair.root0, valid_dp_trans_right_[leaf_pair]);

                if (!std::isfinite(best_left) || !std::isfinite(best_right))
                    continue;

                double cover_value = best_left + best_right + 1.0;
                double open_value = cover_value;
                for (int tree = 0; tree < tree_count_; ++tree) {
                    const int lca = lca_[tree].lca(a, b);
                    const double local_vertex = tree == 0 ? 0.0 : vertex_duals[tree][lca];
                    const double path = path_cost_[tree][lca];
                    cover_value += local_vertex - 2.0 * path;
                    open_value += local_vertex - path;
                }

                const std::pair<int, int> cover_choice{a, b};
                if (better_max(cover_value, best_subtree_cover_[node0]) ||
                    (tied_score(cover_value, best_subtree_cover_[node0]) &&
                     prefer_cover_choice(cover_choice, best_cover_choice_[node0]))) {
                    best_subtree_cover_[node0] = cover_value;
                    best_cover_choice_[node0] = cover_choice;
                }
                dp_[leaf_pair] = open_value;
                left_transition_[leaf_pair] = best_left_choice;
                right_transition_[leaf_pair] = best_right_choice;
            }

            if (node0 >= leaf_count_) {
                const auto [left, right] = trees_[0].child[node0];
                const double split_value =
                    tree0_states[left] == EdgeState::FORCED ||
                        tree0_states[right] == EdgeState::FORCED
                    ? -kInf
                    : best_subtree_cover_[left] + best_subtree_cover_[right];
                const std::pair<int, int> split_choice{-1, -1};
                if (better_max(split_value, best_subtree_cover_[node0]) ||
                    (tied_score(split_value, best_subtree_cover_[node0]) &&
                     prefer_cover_choice(split_choice, best_cover_choice_[node0]))) {
                    best_subtree_cover_[node0] = split_value;
                    best_cover_choice_[node0] = split_choice;
                }
                update_side_cost(node0);
            }
        }

        const int root = trees_[0].root;
        if (!std::isfinite(best_subtree_cover_[root])) {
            return infeasible_result();
        }

        std::fill(reconstructed_.begin(), reconstructed_.end(), 0);
        LagrangianResult result;
        result.lower_bound = static_cast<double>(leaf_count_) - best_subtree_cover_[root];

        const auto mark_upwards_path = [&](int node, const int subtree_root) {
            while (node != subtree_root) {
                if (node < 0 || node >= node_count_)
                    throw std::runtime_error("invalid multi-tree reconstruction path");
                reconstructed_[node] = 1;
                node = trees_[0].parent[node];
            }
        };

        const auto reconstruct_cover = [&](const auto& self, const int subtree_root) -> void {
            reconstructed_[subtree_root] = 1;
            const auto chosen = best_cover_choice_[subtree_root];
            if (chosen.first < 0) {
                if (subtree_root < leaf_count_)
                    throw std::runtime_error("invalid multi-tree split reconstruction");
                self(self, trees_[0].child[subtree_root][0]);
                self(self, trees_[0].child[subtree_root][1]);
                return;
            }

            result.leaf_partition.emplace_back();
            auto& block = result.leaf_partition.back();
            std::queue<std::pair<int, int>> queue;
            queue.push(chosen);
            while (!queue.empty()) {
                const auto [a, b] = queue.front();
                queue.pop();
                if (a == b) {
                    block.push_back(a);
                    mark_upwards_path(a, subtree_root);
                    continue;
                }
                const std::size_t leaf_pair = leaf_pair_index(a, b);
                const auto left = left_transition_[leaf_pair];
                const auto right = right_transition_[leaf_pair];
                if (left.first < 0 || right.first < 0)
                    throw std::runtime_error("failed to reconstruct multi-tree component");
                queue.push(left);
                queue.push(right);
            }
            std::ranges::sort(block);
        };

        for (int node = node_count_ - 1; node >= 0; --node) {
            if (reconstructed_[node] == 0)
                reconstruct_cover(reconstruct_cover, node);
        }
        std::ranges::sort(result.leaf_partition, detail::partition_block_less);
        return result;
    }

private:
    [[nodiscard]] int tree0_find(int x) {
        int root = x;
        while (dynamic_tree0_connectivity_[root] >= 0)
            root = dynamic_tree0_connectivity_[root];
        while (x != root) {
            const int parent = dynamic_tree0_connectivity_[x];
            dynamic_tree0_connectivity_[x] = root;
            x = parent;
        }
        return root;
    }

    [[nodiscard]] bool tree0_same_set(const int a, const int b) {
        return tree0_find(a) == tree0_find(b);
    }

    [[nodiscard]] static int sibling_of(
        const BinaryTree& tree,
        const int parent,
        const int child
    ) {
        const auto [left, right] = tree.child[parent];
        if (left == child)
            return right;
        if (right == child)
            return left;
        throw std::runtime_error("invalid parent/child relation in multi-tree oracle");
    }

    void tree0_join(const int a, const int b) {
        int ra = tree0_find(a);
        int rb = tree0_find(b);
        if (ra == rb)
            return;
        if (dynamic_tree0_connectivity_[ra] > dynamic_tree0_connectivity_[rb])
            std::swap(ra, rb);
        dynamic_tree0_connectivity_[ra] += dynamic_tree0_connectivity_[rb];
        dynamic_tree0_connectivity_[rb] = ra;
    }

    void rebuild_tree0_dynamic_state(std::span<const EdgeState> tree0_states) {
        std::fill(dynamic_tree0_connectivity_.begin(), dynamic_tree0_connectivity_.end(), -1);
        for (int node = 0; node < node_count_; ++node) {
            const int parent = trees_[0].parent[node];
            if (parent >= 0 && tree0_states[node] != EdgeState::CUT)
                tree0_join(node, parent);
        }
        for (int a = 0; a < leaf_count_; ++a) {
            for (int b = a; b < leaf_count_; ++b) {
                const char connected = tree0_same_set(a, b) ? 1 : 0;
                connected_in_tree0_[leaf_pair_index(a, b)] = connected;
                connected_in_tree0_[leaf_pair_index(b, a)] = connected;
            }
        }

        std::fill(dynamic_tree0_climb_valid_.begin(), dynamic_tree0_climb_valid_.end(), 0);
        for (int from = 0; from < node_count_; ++from) {
            dynamic_tree0_climb_valid_[matrix_index(from, from)] = 1;
            int child = from;
            int ancestor = trees_[0].parent[from];
            if (ancestor < 0)
                continue;
            dynamic_tree0_climb_valid_[matrix_index(from, ancestor)] = 1;
            while (ancestor >= 0) {
                const int next = trees_[0].parent[ancestor];
                if (next < 0)
                    break;
                const int skipped = sibling_of(trees_[0], ancestor, child);
                dynamic_tree0_climb_valid_[matrix_index(from, next)] =
                    dynamic_tree0_climb_valid_[matrix_index(from, ancestor)] != 0 &&
                    tree0_states[skipped] != EdgeState::FORCED ? 1 : 0;
                child = ancestor;
                ancestor = next;
            }
        }
    }

    void build_tree0_euler() {
        int timer = 0;
        const auto dfs = [&](const auto& self, const int node) -> void {
            tree0_tin_[node] = timer++;
            if (node >= leaf_count_) {
                self(self, trees_[0].child[node][0]);
                self(self, trees_[0].child[node][1]);
            }
            tree0_tout_[node] = timer;
        };
        dfs(dfs, trees_[0].root);
        if (timer != node_count_)
            throw std::runtime_error("failed to build tree-0 Euler order");
    }

    [[nodiscard]] std::size_t matrix_index(const int from, const int to) const {
        return static_cast<std::size_t>(from) * static_cast<std::size_t>(node_count_) +
            static_cast<std::size_t>(to);
    }

    [[nodiscard]] std::size_t leaf_pair_index(const int a, const int b) const {
        return static_cast<std::size_t>(a) * static_cast<std::size_t>(leaf_count_) +
            static_cast<std::size_t>(b);
    }

    [[nodiscard]] bool other_trees_connected(const int a, const int b) const {
        return connected_in_other_trees_[leaf_pair_index(a, b)] != 0;
    }

    [[nodiscard]] bool tree0_leaves_connected(const int a, const int b) const {
        return connected_in_tree0_[leaf_pair_index(a, b)] != 0;
    }

    // NOLINTEND(bugprone-easily-swappable-parameters)

    [[nodiscard]] std::vector<char> build_climb_valid(const BinaryTree& tree) const {
        std::vector<char> valid(static_cast<std::size_t>(node_count_) * static_cast<std::size_t>(node_count_), 0);
        for (int from = 0; from < node_count_; ++from) {
            valid[matrix_index(from, from)] = 1;
            int child = from;
            int ancestor = tree.parent[from];
            if (ancestor < 0)
                continue;
            valid[matrix_index(from, ancestor)] = 1;
            while (ancestor >= 0) {
                const int next = tree.parent[ancestor];
                if (next < 0)
                    break;
                const int skipped = sibling_of(tree, ancestor, child);
                valid[matrix_index(from, next)] =
                    valid[matrix_index(from, ancestor)] != 0 &&
                    tree.edge_state[skipped] != EdgeState::FORCED ? 1 : 0;
                child = ancestor;
                ancestor = next;
            }
        }
        return valid;
    }

    int tree_count_ = 0;
    int node_count_ = 0;
    int leaf_count_ = 0;
    std::vector<BinaryTree> trees_;
    std::vector<LeafPairLca> lca_;
    std::vector<std::vector<char>> climb_valid_;
    std::vector<ConnectedLeafPair> connected_leaf_pairs_sorted_;
    std::vector<char> connected_in_other_trees_;
    std::vector<std::vector<TransitionCandidate>> valid_dp_trans_left_;
    std::vector<std::vector<TransitionCandidate>> valid_dp_trans_right_;
    std::vector<char> dynamic_tree0_climb_valid_;
    std::vector<int> dynamic_tree0_connectivity_;
    std::vector<char> connected_in_tree0_;
    std::vector<int> tree0_tin_;
    std::vector<int> tree0_tout_;
    Fenwick<double> side_cost_add_;
    Fenwick<int> side_cost_block_;
    std::vector<std::vector<double>> path_cost_;
    std::vector<double> dp_;
    std::vector<std::pair<int, int>> left_transition_;
    std::vector<std::pair<int, int>> right_transition_;
    std::vector<double> best_subtree_cover_;
    std::vector<std::pair<int, int>> best_cover_choice_;
    std::vector<char> reconstructed_;
    std::vector<EdgeState> cached_tree0_states_;
    bool tree0_dynamic_valid_ = false;
};

} // namespace

struct Lagrangian::Impl {
    struct OracleCache {
        std::vector<EdgeState> dynamic_second_states;
        std::unique_ptr<LagrangianForestOracle> oracle;
    };

    explicit Impl(const AnnotatedInstance& instance)
        : base_instance(instance) {}

    [[nodiscard]] LagrangianResult solve(
        const std::vector<std::vector<double>>& vertex_duals,
        const std::vector<std::vector<double>>& edge_duals,
        const std::vector<std::vector<EdgeState>>& edge_states
    ) {
        validate_inputs(base_instance, vertex_duals, edge_duals, edge_states);

        if (base_instance.trees.size() > 2) {
            auto result = multi_tree_oracle(edge_states).solve(
                vertex_duals,
                edge_duals,
                edge_states[0]
            );
            result.candidate_blocks = result.leaf_partition;
            result.lower_bound -= relaxed_vertex_offset(base_instance, vertex_duals);
            result.lower_bound += edge_offset(edge_duals);
            return result;
        }

        auto result = solve_two_tree_orientation(
            two_tree_caches[0],
            0,
            1,
            vertex_duals,
            edge_duals,
            edge_states
        );
        const auto swapped = solve_two_tree_orientation(
            two_tree_caches[1],
            1,
            0,
            vertex_duals,
            edge_duals,
            edge_states
        );
        merge_candidate_blocks(result.candidate_blocks, swapped.pricing_blocks());
        if (better_min(result.lower_bound, swapped.lower_bound)) {
            result.leaf_partition = swapped.leaf_partition;
            result.lower_bound = swapped.lower_bound;
        }
        return result;
    }

    [[nodiscard]] LagrangianResult solve_fixed_two_tree_orientation(
        const int first_tree,
        const int second_tree,
        const std::vector<std::vector<double>>& vertex_duals,
        const std::vector<std::vector<double>>& edge_duals,
        const std::vector<std::vector<EdgeState>>& edge_states
    ) {
        validate_inputs(base_instance, vertex_duals, edge_duals, edge_states);
        if (base_instance.trees.size() != 2)
            throw std::invalid_argument("fixed two-tree orientation requires exactly two trees");
        if (first_tree < 0 || first_tree >= 2 || second_tree < 0 || second_tree >= 2 || first_tree == second_tree)
            throw std::invalid_argument("invalid two-tree orientation");
        return solve_two_tree_orientation(
            two_tree_caches[static_cast<std::size_t>(first_tree)],
            first_tree,
            second_tree,
            vertex_duals,
            edge_duals,
            edge_states
        );
    }

private:
    [[nodiscard]] LagrangianResult solve_two_tree_orientation(
        OracleCache& cache,
        const int first_tree,
        const int second_tree,
        const std::vector<std::vector<double>>& vertex_duals,
        const std::vector<std::vector<double>>& edge_duals,
        const std::vector<std::vector<EdgeState>>& edge_states
    ) {
        auto& oracle_ref = oracle(cache, first_tree, second_tree, edge_states[second_tree]);
        const auto forest = oracle_ref.solve(
            vertex_duals[second_tree],
            edge_states[first_tree],
            edge_duals[first_tree],
            edge_duals[second_tree]
        );
        if (!forest) {
            return infeasible_result();
        }
        return make_result(
            forest->partition,
            forest->primary_value -
                lambda_offset(base_instance.trees[second_tree].leaves(), vertex_duals[second_tree]) +
                forced_offset(edge_duals[first_tree]) +
                forced_offset(edge_duals[second_tree])
        );
    }

    [[nodiscard]] LagrangianForestOracle& oracle(
        OracleCache& cache,
        const int first_tree,
        const int second_tree,
        std::span<const EdgeState> second_edge_states
    ) {
        if (static_cast<int>(second_edge_states.size()) != base_instance.trees[second_tree].vertices())
            throw std::invalid_argument("invalid second-tree edge states");
        const bool second_tree_changed =
            !std::ranges::equal(cache.dynamic_second_states, second_edge_states);
        if (!cache.oracle || second_tree_changed) {
            Tree second = base_instance.trees[second_tree];
            second.edge_state.assign(second_edge_states.begin(), second_edge_states.end());
            cache.oracle = std::make_unique<LagrangianForestOracle>(base_instance.trees[first_tree], second);
            cache.dynamic_second_states.assign(second_edge_states.begin(), second_edge_states.end());
        }
        return *cache.oracle;
    }

    [[nodiscard]] MultiTreeLagrangianOracle& multi_tree_oracle(
        const std::vector<std::vector<EdgeState>>& edge_states
    ) {
        const auto other_states_match = [&]() {
            if (cached_multi_tree_other_states.size() + 1U != edge_states.size())
                return false;
            for (int tree = 1; tree < static_cast<int>(edge_states.size()); ++tree) {
                if (cached_multi_tree_other_states[static_cast<std::size_t>(tree - 1)] != edge_states[tree])
                    return false;
            }
            return true;
        };
        if (!cached_multi_tree_oracle || !other_states_match()) {
            AnnotatedInstance current = base_instance;
            cached_multi_tree_other_states.resize(current.trees.size() - 1U);
            for (int tree = 1; tree < static_cast<int>(current.trees.size()); ++tree) {
                current.trees[tree].edge_state = edge_states[tree];
                cached_multi_tree_other_states[static_cast<std::size_t>(tree - 1)] = edge_states[tree];
            }
            cached_multi_tree_oracle = std::make_unique<MultiTreeLagrangianOracle>(current);
        }
        return *cached_multi_tree_oracle;
    }

    AnnotatedInstance base_instance;
    std::vector<std::vector<EdgeState>> cached_multi_tree_other_states;
    std::array<OracleCache, 2> two_tree_caches;
    std::unique_ptr<MultiTreeLagrangianOracle> cached_multi_tree_oracle;
};

Lagrangian::Lagrangian(const AnnotatedInstance& instance)
    : impl_(std::make_unique<Impl>(instance)) {}

Lagrangian::~Lagrangian() = default;
Lagrangian::Lagrangian(Lagrangian&&) noexcept = default;
Lagrangian& Lagrangian::operator=(Lagrangian&&) noexcept = default;

LagrangianResult Lagrangian::solve(
    const std::vector<std::vector<double>>& vertex_duals,
    const std::vector<std::vector<double>>& edge_duals,
    const std::vector<std::vector<EdgeState>>& edge_states
) {
    return impl_->solve(
        vertex_duals,
        edge_duals,
        edge_states
    );
}

LagrangianResult Lagrangian::solve_two_tree_orientation(
    const int first_tree,
    const int second_tree,
    const std::vector<std::vector<double>>& vertex_duals,
    const std::vector<std::vector<double>>& edge_duals,
    const std::vector<std::vector<EdgeState>>& edge_states
) {
    return impl_->solve_fixed_two_tree_orientation(
        first_tree,
        second_tree,
        vertex_duals,
        edge_duals,
        edge_states
    );
}

} // namespace maffe
