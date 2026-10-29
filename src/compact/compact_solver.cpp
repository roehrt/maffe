#include "compact/compact_solver.hpp"

#include "config.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "branchandprice/highs/root_lp.hpp"
#include "compact/lca.hpp"
#include "reductions/detail/binary_tree_view.hpp"

// SCIP typedefs these plugin payload tags globally. Give this translation unit
// distinct tag names so C++ LTO does not report false cross-plugin ODR clashes.
#define SCIP_ConshdlrData MaffeCompactScipConshdlrData
#define SCIP_ConsData MaffeCompactScipConsData
#define SCIP_SepaData MaffeCompactScipSepaData

#include "scip/cons_linear.h"
#include "scip/cons_setppc.h"
#include "scip/pub_message.h"
#include "scip/pub_sepa.h"
#include "scip/pub_var.h"
#include "scip/scip.h"
#include "scip/scip_cons.h"
#include "scip/scip_lp.h"
#include "scip/scip_param.h"
#include "scip/scip_prob.h"
#include "scip/scip_sepa.h"
#include "scip/scip_sol.h"
#include "scip/scip_var.h"
#include "scip/scipdefplugins.h"

#undef SCIP_ConshdlrData
#undef SCIP_ConsData
#undef SCIP_SepaData

#include "util/scip_utils.hpp"

#include "util/log.hpp"
#include "util/partition_ops.hpp"
#include "util/scip_error_log.hpp"
#include "util/constants.hpp"
#include "util/union_find.hpp"

#if MAFFE_HAVE_GUROBI
#include "gurobi_c.h"
#endif

#ifndef MAFFE_COMPETITION_HEURISTIC_BUILD
#define MAFFE_COMPETITION_HEURISTIC_BUILD 0
#endif

namespace maffe {

struct CompactSolveData;

} // namespace maffe

struct MaffeCompactScipConshdlrData {
    maffe::CompactSolveData* data = nullptr;
};

struct MaffeCompactScipConsData {
    std::vector<SCIP_VAR*> vars;
};

struct MaffeCompactScipSepaData {
    maffe::CompactSolveData* data = nullptr;
};

namespace maffe {
namespace {

using compact::LcaHelper;

constexpr double kInf = std::numeric_limits<double>::infinity();

using scip_error_log::ignore_unused;
using scip_error_log::status_name;

void check(const SCIP_RETCODE status) {
    if (status != SCIP_OKAY)
        throw std::runtime_error("SCIP call failed");
}

[[nodiscard]] std::optional<std::chrono::steady_clock::time_point> deadline_after(
    const std::optional<double> time_limit_seconds
) {
    if (!time_limit_seconds.has_value())
        return std::nullopt;
    return std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(*time_limit_seconds));
}

[[nodiscard]] std::optional<double> remaining_seconds(
    const std::optional<std::chrono::steady_clock::time_point> deadline
) {
    if (!deadline.has_value())
        return std::nullopt;
    return std::max(0.0, std::chrono::duration<double>(*deadline - std::chrono::steady_clock::now()).count());
}

void apply_scip_time_limit(
    SCIP* scip,
    const std::optional<std::chrono::steady_clock::time_point> deadline
) {
    const auto remaining = remaining_seconds(deadline);
    if (!remaining.has_value())
        return;
    if (*remaining <= 0.0)
        throw std::runtime_error("compact LP solver timed out before SCIP solve");
    check(SCIPsetRealParam(scip, "limits/time", *remaining));
}

void apply_compact_scip_settings(SCIP* scip, const LogLevel log_level) {
    check(SCIPsetBoolParam(scip, "misc/catchctrlc", FALSE));
    if (logging::enabled(log_level, LogLevel::VERBOSE))
        check(SCIPsetIntParam(scip, "display/freq", constants::scip_verbose_display_frequency));
    check(SCIPsetIntParam(scip, "propagating/probing/maxprerounds", constants::compact_scip_probing_max_prerounds));
    check(SCIPsetIntParam(scip, "separating/maxcuts", constants::compact_scip_max_cuts));
    check(SCIPsetIntParam(scip, "separating/maxcutsroot", constants::compact_scip_max_cuts_root));
}

SCIP_DECL_MESSAGEWARNING(scip_message_warning) {
    ignore_unused(messagehdlr, file);
    scip_error_log::log_prefixed_lines("compact-scip: ", msg);
}

struct StateIndex {
    int leaves = 0;

    [[nodiscard]] int operator()(const int a, const int b) const {
        return b * (b + 1) / 2 + a;
    }

    [[nodiscard]] int states() const {
        return leaves * (leaves + 1) / 2;
    }
};

struct TreeData {
    explicit TreeData(const Tree& tree, const StateIndex& index)
        : helper(tree),
          vertices(tree.vertices()),
          state_lca(index.states()) {
        for (int b = 0; b < index.leaves; ++b) {
            for (int a = 0; a <= b; ++a)
                state_lca[index(a, b)] = helper.lca(a, b);
        }
    }

    LcaHelper helper;
    int vertices = 0;
    std::vector<int> state_lca;
};

struct Variable {
    int source = -1;
    int target = -1;
    bool left = false;
};

enum class CompactColumnKind : unsigned char {
    Transition,
    RectangleSource,
    RectangleTarget,
};

enum class CompactVarType : unsigned char {
    Binary,
    Integer,
};

enum class RectangleCapacitySide : unsigned char {
    None,
    Source,
    Target,
};

struct RectangleFactor {
    bool left = false;
    std::vector<int> sources;
    std::vector<int> targets;
    std::vector<int> source_cols;
    std::vector<int> target_cols;
    std::vector<RectangleCapacitySide> capacity_side;
};

struct CompactModel {
    std::vector<double> row_lower;
    std::vector<double> row_upper;
    std::vector<double> col_cost;
    std::vector<double> col_lower;
    std::vector<double> col_upper;
    std::vector<CompactVarType> col_type;
    std::vector<int> start{0};
    std::vector<int> index;
    std::vector<double> value;
    std::vector<Variable> variables;
    std::vector<CompactColumnKind> column_kind;
    std::vector<int> column_factor;
    std::vector<RectangleFactor> rectangles;
    double objective_offset = 0.0;
    int capacity_vertex_stride = 0;
    int rectangle_groups = 0;
    long long rectangle_original_vars = 0;
    long long rectangle_folded_vars = 0;

    [[nodiscard]] int rows() const {
        return static_cast<int>(row_lower.size());
    }

    [[nodiscard]] int cols() const {
        return static_cast<int>(col_cost.size());
    }
};

struct ModelRows {
    std::vector<int> balance;
    std::vector<int> postflow;
};

struct ColumnScratch {
    std::vector<int> index;
    std::vector<double> value;
};

struct ScipModel {
    std::vector<SCIP_VAR*> vars;
    int root_left_rows = 0;
    std::size_t root_left_nonzeros = 0;
};

using CapacityColumns = std::vector<std::vector<std::vector<int>>>;

struct RowRef {
    int group = -1;
    int row = -1;
};

struct CapacityViolation {
    int tree = -1;
    int vertex = -1;
    int nonzeros = 0;
    double violation = 0.0;
};

struct CapacitySeedStats {
    int rows = 0;
    std::size_t nonzeros = 0;
};

struct CapacitySeedCandidate {
    int tree = -1;
    int vertex = -1;
    int nonzeros = 0;
    double dual = 0.0;
    double support = 0.0;
    double priority = 0.0;
};

struct CapacitySeedRow {
    int tree = -1;
    int vertex = -1;
    int nonzeros = 0;
    std::vector<int> cols;
};

struct VariablePruneStats {
    int removed = 0;
    int rounds = 0;
};

struct CapacityIndex {
    std::vector<std::vector<int>> row_nonzeros;
    int active_rows = 0;
    std::size_t nonzeros = 0;
};

void simplify_at_most_one_rows(CapacityColumns& rows, int col_count);

struct CompactStates {
    int leaves = 0;
    int original_states = 0;
    std::vector<int> original_to_state;
    std::vector<int> representative;
    std::vector<int> first_leaf;
    std::vector<int> last_leaf;
    std::vector<std::vector<int>> tree_lca;

    [[nodiscard]] int states() const {
        return static_cast<int>(representative.size());
    }
};

struct CompactPrepared {
    AnnotatedInstance instance;
    StateIndex state_index;
    std::vector<TreeData> trees;
    CompactStates states;
    CompactModel model;
    CapacityIndex capacity_index;
    VariablePruneStats prune_stats;
};

[[nodiscard]] bool clean_tree(const Tree& tree) {
    return std::ranges::all_of(tree.edge_state, [](const auto state) { return state == EdgeState::UNKNOWN; });
}

[[nodiscard]] bool internal_state(const int state, const std::span<const int> first_leaf, const std::span<const int> last_leaf) {
    return first_leaf[state] != last_leaf[state];
}

[[nodiscard]] CompactStates build_compact_states(
    const StateIndex& index,
    const std::vector<TreeData>& trees
) {
    CompactStates states{
        .leaves = index.leaves,
        .original_states = index.states(),
        .original_to_state = std::vector<int>(index.states(), -1),
        .representative = {},
        .first_leaf = {},
        .last_leaf = {},
        .tree_lca = {},
    };

    std::map<std::vector<int>, int> key_to_state;
    std::vector<int> key;
    key.reserve(trees.size());
    for (int b = 0; b < index.leaves; ++b) {
        for (int a = 0; a <= b; ++a) {
            const int original = index(a, b);
            key.clear();
            for (const TreeData& tree : trees)
                key.push_back(tree.state_lca[original]);

            const auto [it, inserted] = key_to_state.emplace(key, states.states());
            const int state = it->second;
            if (inserted) {
                states.representative.push_back(original);
                states.first_leaf.push_back(a);
                states.last_leaf.push_back(b);
                states.tree_lca.emplace_back(key);
            }
            states.original_to_state[original] = state;
        }
    }

    return states;
}

[[nodiscard]] int add_row(CompactModel& model, const double lower, const double upper) {
    const int row = model.rows();
    model.row_lower.push_back(lower);
    model.row_upper.push_back(upper);
    return row;
}

[[nodiscard]] ModelRows build_rows(CompactModel& model, const CompactStates& states) {
    ModelRows rows{
        .balance = std::vector<int>(states.states(), -1),
        .postflow = std::vector<int>(states.states(), -1),
    };

    model.row_lower.reserve(states.states() * 2);
    model.row_upper.reserve(states.states() * 2);

    for (int state = 0; state < states.states(); ++state) {
        if (!internal_state(state, states.first_leaf, states.last_leaf))
            continue;
        rows.balance[state] = add_row(model, 0.0, 0.0);
        rows.postflow[state] = add_row(model, 0.0, kInf);
    }

    return rows;
}

[[nodiscard]] bool valid_transition(
    const std::vector<TreeData>& trees,
    const int source,
    const int target
) {
    for (const auto& tree : trees) {
        const int top = tree.state_lca[source];
        const int bottom = tree.state_lca[target];
        if (top == bottom || !tree.helper.ancestor(top, bottom))
            return false;
    }
    return true;
}

void add_coeff(std::vector<int>& index, std::vector<double>& value, const int row, const double coeff) {
    if (row < 0 || coeff == 0.0)
        return;
    index.push_back(row);
    value.push_back(coeff);
}

void append_column(
    CompactModel& model,
    const double cost,
    const double lower,
    const double upper,
    const CompactVarType type,
    const std::span<const int> index,
    const std::span<const double> value,
    Variable meta,
    const CompactColumnKind kind = CompactColumnKind::Transition,
    const int factor = -1
) {
    model.col_cost.push_back(cost);
    model.col_lower.push_back(lower);
    model.col_upper.push_back(upper);
    model.col_type.push_back(type);
    model.index.insert(model.index.end(), index.begin(), index.end());
    model.value.insert(model.value.end(), value.begin(), value.end());
    model.start.push_back(static_cast<int>(model.index.size()));
    model.variables.push_back(meta);
    model.column_kind.push_back(kind);
    model.column_factor.push_back(factor);
}

void append_column(
    CompactModel& model,
    const double cost,
    const std::span<const int> index,
    const std::span<const double> value,
    Variable meta
) {
    append_column(
        model,
        cost,
        0.0,
        1.0,
        CompactVarType::Binary,
        index,
        value,
        meta
    );
}

void add_variable(
    CompactModel& model,
    const ModelRows& rows,
    const std::span<const int> first_leaf,
    const std::span<const int> last_leaf,
    ColumnScratch& scratch,
    const bool left,
    const int source,
    const int target
) {
    const bool source_internal = internal_state(source, first_leaf, last_leaf);
    const bool target_internal = internal_state(target, first_leaf, last_leaf);
    auto& row_index = scratch.index;
    auto& row_value = scratch.value;
    row_index.clear();
    row_value.clear();
    row_index.reserve(4);
    row_value.reserve(row_index.capacity());

    if (source_internal) {
        add_coeff(row_index, row_value, rows.balance[source], left ? 1.0 : -1.0);
        if (left)
            add_coeff(row_index, row_value, rows.postflow[source], 1.0);
    }
    if (target_internal)
        add_coeff(row_index, row_value, rows.postflow[target], -1.0);

    append_column(
        model,
        -1.0 + (left && source_internal ? 1.0 : 0.0),
        row_index,
        row_value,
        Variable{.source = source, .target = target, .left = left}
    );
}

[[nodiscard]] CompactModel build_compact_model(
    const StateIndex& state_index,
    const CompactStates& states,
    const std::vector<TreeData>& trees
) {
    CompactModel model;
    // The transition costs count components relative to the all-singleton forest.
    // Keep the solver objective in the public "number of components" convention.
    model.objective_offset = static_cast<double>(state_index.leaves);

    const auto rows = build_rows(model, states);
    ColumnScratch scratch;
    std::unordered_map<std::uint64_t, int> seen_variables;
    seen_variables.reserve(state_index.states() * 4);
    const auto add_unique_variable = [&](const bool left, const int original_source, const int original_target) {
        const int source = states.original_to_state[original_source];
        const int target = states.original_to_state[original_target];
        const auto key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(source)) << 33) |
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(target)) << 1) |
            static_cast<std::uint64_t>(left ? 1 : 0);
        if (!seen_variables.emplace(key, model.cols()).second)
            return;
        add_variable(model, rows, states.first_leaf, states.last_leaf, scratch, left, source, target);
    };

    for (int i1 = 0; i1 < state_index.leaves; ++i1) {
        for (int i2 = i1; i2 < state_index.leaves; ++i2) {
            const int source = state_index(i1, i2);
            for (int j2 = i1; j2 < state_index.leaves; ++j2) {
                const int target = state_index(i1, j2);
                if (valid_transition(trees, source, target))
                    add_unique_variable(true, source, target);
            }
        }
    }

    for (int i2 = 0; i2 < state_index.leaves; ++i2) {
        for (int i1 = 0; i1 <= i2; ++i1) {
            const int source = state_index(i1, i2);
            for (int j2 = i2; j2 < state_index.leaves; ++j2) {
                const int target = state_index(i2, j2);
                if (valid_transition(trees, source, target))
                    add_unique_variable(false, source, target);
            }
        }
    }

    return model;
}

[[nodiscard]] VariablePruneStats prune_dead_transition_variables(
    CompactModel& model,
    const CompactStates& states
) {
    std::vector<char> active(model.cols(), true);
    VariablePruneStats stats;

    std::vector<char> productive(states.states(), false);
    for (int state = 0; state < states.states(); ++state)
        productive[state] = !internal_state(state, states.first_leaf, states.last_leaf);

    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<char> left_departure_to_productive(states.states(), false);
        std::vector<char> right_departure_to_productive(states.states(), false);

        for (int col = 0; col < model.cols(); ++col) {
            const Variable& variable = model.variables[col];
            if (!internal_state(variable.source, states.first_leaf, states.last_leaf))
                continue;
            if (internal_state(variable.target, states.first_leaf, states.last_leaf) && !productive[variable.target])
                continue;
            if (variable.left)
                left_departure_to_productive[variable.source] = true;
            else
                right_departure_to_productive[variable.source] = true;
        }

        for (int state = 0; state < states.states(); ++state) {
            if (productive[state] || !internal_state(state, states.first_leaf, states.last_leaf))
                continue;
            if (!left_departure_to_productive[state] || !right_departure_to_productive[state])
                continue;
            productive[state] = true;
            changed = true;
        }
        ++stats.rounds;
    }

    for (int col = 0; col < model.cols(); ++col) {
        const Variable& variable = model.variables[col];
        const bool dead_source =
            internal_state(variable.source, states.first_leaf, states.last_leaf) &&
            !productive[variable.source];
        const bool dead_target =
            internal_state(variable.target, states.first_leaf, states.last_leaf) &&
            !productive[variable.target];
        if (!dead_source && !dead_target)
            continue;
        active[col] = false;
        ++stats.removed;
    }

    if (stats.removed == 0) {
        stats.rounds = 0;
        return stats;
    }

    CompactModel pruned;
    pruned.row_lower = model.row_lower;
    pruned.row_upper = model.row_upper;
    pruned.objective_offset = model.objective_offset;
    pruned.start.reserve(model.cols() - stats.removed + 1);
    pruned.start.push_back(0);
    pruned.col_cost.reserve(model.cols() - stats.removed);
    pruned.col_lower.reserve(model.cols() - stats.removed);
    pruned.col_upper.reserve(model.cols() - stats.removed);
    pruned.col_type.reserve(model.cols() - stats.removed);
    pruned.variables.reserve(model.cols() - stats.removed);
    pruned.column_kind.reserve(model.cols() - stats.removed);
    pruned.column_factor.reserve(model.cols() - stats.removed);
    pruned.capacity_vertex_stride = model.capacity_vertex_stride;
    pruned.rectangle_groups = model.rectangle_groups;
    pruned.rectangle_original_vars = model.rectangle_original_vars;
    pruned.rectangle_folded_vars = model.rectangle_folded_vars;

    for (int col = 0; col < model.cols(); ++col) {
        if (!active[col])
            continue;
        pruned.col_cost.push_back(model.col_cost[col]);
        pruned.col_lower.push_back(model.col_lower[col]);
        pruned.col_upper.push_back(model.col_upper[col]);
        pruned.col_type.push_back(model.col_type[col]);
        pruned.index.insert(
            pruned.index.end(),
            model.index.begin() + model.start[col],
            model.index.begin() + model.start[col + 1]
        );
        pruned.value.insert(
            pruned.value.end(),
            model.value.begin() + model.start[col],
            model.value.begin() + model.start[col + 1]
        );
        pruned.start.push_back(static_cast<int>(pruned.index.size()));
        pruned.variables.push_back(model.variables[col]);
        pruned.column_kind.push_back(model.column_kind[col]);
        pruned.column_factor.push_back(model.column_factor[col]);
    }

    model = std::move(pruned);
    return stats;
}

template <class Visit>
void for_each_transition_capacity_vertex(
    const std::vector<TreeData>& trees,
    const CompactStates& states,
    const Variable& variable,
    const int tree_index,
    Visit&& visit
) {
    const bool source_internal = internal_state(variable.source, states.first_leaf, states.last_leaf);
    const bool target_internal = internal_state(variable.target, states.first_leaf, states.last_leaf);
    const TreeData& tree = trees[tree_index];
    const int top = states.tree_lca[variable.source][tree_index];
    const int bottom = states.tree_lca[variable.target][tree_index];

    if (variable.left && source_internal)
        visit(top);

    for (int u = bottom; u != top; u = tree.helper.parent(u)) {
        if (u < tree.helper.leaves() || (target_internal && u == bottom))
            continue;
        visit(u);
    }
}

[[nodiscard]] bool transition_uses_capacity_vertex(
    const std::vector<TreeData>& trees,
    const CompactStates& states,
    const Variable& variable,
    const int tree_index,
    const int vertex
) {
    const bool source_internal = internal_state(variable.source, states.first_leaf, states.last_leaf);
    const bool target_internal = internal_state(variable.target, states.first_leaf, states.last_leaf);
    const TreeData& tree = trees[tree_index];
    const int top = states.tree_lca[variable.source][tree_index];
    const int bottom = states.tree_lca[variable.target][tree_index];

    if (variable.left && source_internal && vertex == top)
        return true;
    if (vertex == top || vertex < tree.helper.leaves())
        return false;
    if (target_internal && vertex == bottom)
        return false;
    return tree.helper.ancestor(top, vertex) && tree.helper.ancestor(vertex, bottom);
}

[[nodiscard]] RectangleCapacitySide rectangle_capacity_side(
    const CompactModel& model,
    const int factor,
    const int tree_index,
    const int vertex
) {
    if (factor < 0 || factor >= static_cast<int>(model.rectangles.size()))
        return RectangleCapacitySide::None;
    const RectangleFactor& rectangle = model.rectangles[factor];
    const int code = tree_index * model.capacity_vertex_stride + vertex;
    if (code < 0 || code >= static_cast<int>(rectangle.capacity_side.size()))
        return RectangleCapacitySide::None;
    return rectangle.capacity_side[code];
}

template <class Visit>
void for_each_capacity_vertex(
    const CompactModel& model,
    const std::vector<TreeData>& trees,
    const CompactStates& states,
    const int col,
    const int tree_index,
    Visit&& visit
) {
    const Variable& variable = model.variables[col];
    if (model.column_kind[col] == CompactColumnKind::Transition) {
        for_each_transition_capacity_vertex(trees, states, variable, tree_index, std::forward<Visit>(visit));
        return;
    }

    const RectangleFactor& rectangle = model.rectangles[model.column_factor[col]];
    if (rectangle.sources.empty() || rectangle.targets.empty())
        return;

    const bool source_column = model.column_kind[col] == CompactColumnKind::RectangleSource;
    const int source = source_column ? variable.source : rectangle.sources.front();
    const int target = source_column ? rectangle.targets.front() : variable.target;
    const RectangleCapacitySide expected_side = source_column
        ? RectangleCapacitySide::Source
        : RectangleCapacitySide::Target;
    const Variable representative{.source = source, .target = target, .left = rectangle.left};
    for_each_transition_capacity_vertex(trees, states, representative, tree_index, [&](const int vertex) {
        if (rectangle_capacity_side(model, model.column_factor[col], tree_index, vertex) == expected_side)
            visit(vertex);
    });
}

[[nodiscard]] bool variable_uses_capacity_vertex(
    const CompactModel& model,
    const std::vector<TreeData>& trees,
    const CompactStates& states,
    const int col,
    const int tree_index,
    const int vertex
) {
    const Variable& variable = model.variables[col];
    if (model.column_kind[col] == CompactColumnKind::Transition)
        return transition_uses_capacity_vertex(trees, states, variable, tree_index, vertex);

    const RectangleFactor& rectangle = model.rectangles[model.column_factor[col]];
    if (rectangle.sources.empty() || rectangle.targets.empty())
        return false;

    const bool source_column = model.column_kind[col] == CompactColumnKind::RectangleSource;
    const RectangleCapacitySide expected_side = source_column
        ? RectangleCapacitySide::Source
        : RectangleCapacitySide::Target;
    if (rectangle_capacity_side(model, model.column_factor[col], tree_index, vertex) != expected_side)
        return false;

    const int source = source_column ? variable.source : rectangle.sources.front();
    const int target = source_column ? rectangle.targets.front() : variable.target;
    return transition_uses_capacity_vertex(
        trees,
        states,
        Variable{.source = source, .target = target, .left = rectangle.left},
        tree_index,
        vertex
    );
}

[[nodiscard]] CapacityIndex build_capacity_index(
    const CompactModel& model,
    const std::vector<TreeData>& trees,
    const CompactStates& states
) {
    CapacityIndex index;
    index.row_nonzeros.reserve(trees.size());
    for (const auto& tree : trees)
        index.row_nonzeros.emplace_back(tree.vertices, 0);

    for (int col = 0; col < model.cols(); ++col) {
        for (int tree_index = 0; tree_index < static_cast<int>(trees.size()); ++tree_index) {
            for_each_capacity_vertex(model, trees, states, col, tree_index, [&](const int vertex) {
                ++index.row_nonzeros[tree_index][vertex];
                ++index.nonzeros;
            });
        }
    }

    for (int tree_index = 0; tree_index < static_cast<int>(trees.size()); ++tree_index) {
        const int leaves = trees[tree_index].helper.leaves();
        for (int vertex = leaves; vertex < trees[tree_index].vertices; ++vertex) {
            if (index.row_nonzeros[tree_index][vertex] > 1)
                ++index.active_rows;
        }
    }
    return index;
}

struct VectorHash {
    [[nodiscard]] std::size_t operator()(const std::vector<int>& values) const noexcept {
        std::uint64_t hash = 1469598103934665603ULL;
        for (const int value : values) {
            hash ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(value));
            hash *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(hash);
    }
};

[[nodiscard]] ModelRows rows_for_states(const CompactStates& states) {
    ModelRows rows{
        .balance = std::vector<int>(states.states(), -1),
        .postflow = std::vector<int>(states.states(), -1),
    };
    int row = 0;
    for (int state = 0; state < states.states(); ++state) {
        if (!internal_state(state, states.first_leaf, states.last_leaf))
            continue;
        rows.balance[state] = row++;
        rows.postflow[state] = row++;
    }
    return rows;
}

[[nodiscard]] std::uint64_t transition_key(const bool left, const int source, const int target) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(source)) << 33) |
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(target)) << 1) |
        static_cast<std::uint64_t>(left ? 1 : 0);
}

[[nodiscard]] std::uint64_t unordered_pair_key(const int first, const int second) {
    const auto a = static_cast<std::uint32_t>(std::min(first, second));
    const auto b = static_cast<std::uint32_t>(std::max(first, second));
    return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
}

void intersect_sorted_in_place(std::vector<int>& into, const std::vector<int>& with) {
    std::vector<int> result;
    result.reserve(std::min(into.size(), with.size()));
    std::ranges::set_intersection(into, with, std::back_inserter(result));
    into = std::move(result);
}

[[nodiscard]] std::optional<std::vector<RectangleCapacitySide>> rectangle_capacity_sides(
    const std::vector<TreeData>& trees,
    const CompactStates& states,
    const CapacityIndex& capacity_index,
    const std::span<const int> sources,
    const std::span<const int> targets,
    const bool left,
    const int vertex_stride
) {
    struct Pattern {
        int total = 0;
        std::unordered_map<int, int> by_source;
        std::unordered_map<int, int> by_target;
    };

    std::vector<Pattern> patterns(trees.size() * vertex_stride);
    for (const int source : sources) {
        for (const int target : targets) {
            const Variable variable{.source = source, .target = target, .left = left};
            for (int tree_index = 0; tree_index < static_cast<int>(trees.size()); ++tree_index) {
                for_each_transition_capacity_vertex(trees, states, variable, tree_index, [&](const int vertex) {
                    if (capacity_index.row_nonzeros[tree_index][vertex] <= 1)
                        return;
                    Pattern& pattern = patterns[tree_index * vertex_stride + vertex];
                    if (pattern.total == 0) {
                        pattern.by_source.reserve(sources.size());
                        pattern.by_target.reserve(targets.size());
                    }
                    ++pattern.total;
                    ++pattern.by_source[source];
                    ++pattern.by_target[target];
                });
            }
        }
    }

    std::vector<RectangleCapacitySide> sides(patterns.size(), RectangleCapacitySide::None);
    for (int code = 0; code < static_cast<int>(patterns.size()); ++code) {
        const Pattern& pattern = patterns[code];
        if (pattern.total == 0)
            continue;

        bool source_complete = true;
        for (const auto& [_, count] : pattern.by_source) {
            if (count != static_cast<int>(targets.size())) {
                source_complete = false;
                break;
            }
        }

        bool target_complete = true;
        for (const auto& [_, count] : pattern.by_target) {
            if (count != static_cast<int>(sources.size())) {
                target_complete = false;
                break;
            }
        }

        if (source_complete)
            sides[code] = RectangleCapacitySide::Source;
        else if (target_complete)
            sides[code] = RectangleCapacitySide::Target;
        else
            return std::nullopt;
    }
    return sides;
}

void append_copied_column(CompactModel& folded, const CompactModel& model, const int col) {
    append_column(
        folded,
        model.col_cost[col],
        model.col_lower[col],
        model.col_upper[col],
        model.col_type[col],
        std::span<const int>(model.index.data() + model.start[col], model.start[col + 1] - model.start[col]),
        std::span<const double>(model.value.data() + model.start[col], model.start[col + 1] - model.start[col]),
        model.variables[col],
        model.column_kind[col],
        model.column_factor[col]
    );
}

void append_rectangle_source_column(
    CompactModel& folded,
    const ModelRows& rows,
    const CompactStates& states,
    const int factor,
    const bool left,
    const int source,
    const int equality_row,
    ColumnScratch& scratch
) {
    scratch.index.clear();
    scratch.value.clear();
    const bool source_internal = internal_state(source, states.first_leaf, states.last_leaf);
    if (source_internal) {
        add_coeff(scratch.index, scratch.value, rows.balance[source], left ? 1.0 : -1.0);
        if (left)
            add_coeff(scratch.index, scratch.value, rows.postflow[source], 1.0);
    }
    add_coeff(scratch.index, scratch.value, equality_row, 1.0);
    append_column(
        folded,
        -1.0 + (left && source_internal ? 1.0 : 0.0),
        0.0,
        1.0,
        CompactVarType::Binary,
        scratch.index,
        scratch.value,
        Variable{.source = source, .target = -1, .left = left},
        CompactColumnKind::RectangleSource,
        factor
    );
}

void append_rectangle_target_column(
    CompactModel& folded,
    const ModelRows& rows,
    const CompactStates& states,
    const int factor,
    const bool left,
    const int target,
    const int source_count,
    const int equality_row,
    ColumnScratch& scratch
) {
    scratch.index.clear();
    scratch.value.clear();
    if (internal_state(target, states.first_leaf, states.last_leaf))
        add_coeff(scratch.index, scratch.value, rows.postflow[target], -1.0);
    add_coeff(scratch.index, scratch.value, equality_row, -1.0);
    append_column(
        folded,
        0.0,
        0.0,
        static_cast<double>(source_count),
        CompactVarType::Integer,
        scratch.index,
        scratch.value,
        Variable{.source = -1, .target = target, .left = left},
        CompactColumnKind::RectangleTarget,
        factor
    );
}

[[nodiscard]] CompactModel fold_transition_rectangles(
    const CompactModel& model,
    const std::vector<TreeData>& trees,
    const CompactStates& states,
    const CapacityIndex& capacity_index
) {
    std::vector<std::vector<int>> targets_by_source_left(states.states());
    std::vector<std::vector<int>> targets_by_source_right(states.states());
    std::vector<std::vector<int>> sources_by_target_left(states.states());
    std::vector<std::vector<int>> sources_by_target_right(states.states());
    std::unordered_map<std::uint64_t, int> transition_to_col;
    transition_to_col.reserve(model.cols());

    for (int col = 0; col < model.cols(); ++col) {
        if (model.column_kind[col] != CompactColumnKind::Transition)
            continue;
        const Variable& variable = model.variables[col];
        transition_to_col.emplace(transition_key(variable.left, variable.source, variable.target), col);
        auto& targets = variable.left ? targets_by_source_left[variable.source] : targets_by_source_right[variable.source];
        targets.push_back(variable.target);
        auto& sources = variable.left ? sources_by_target_left[variable.target] : sources_by_target_right[variable.target];
        sources.push_back(variable.source);
    }
    const auto normalize_lists = [](std::vector<std::vector<int>>& lists) {
        for (std::vector<int>& values : lists) {
            std::ranges::sort(values);
            values.erase(std::unique(values.begin(), values.end()), values.end());
        }
    };
    normalize_lists(targets_by_source_left);
    normalize_lists(targets_by_source_right);
    normalize_lists(sources_by_target_left);
    normalize_lists(sources_by_target_right);

    int vertex_stride = 0;
    for (const TreeData& tree : trees)
        vertex_stride = std::max(vertex_stride, tree.vertices);

    struct Candidate {
        bool left = false;
        std::vector<int> sources;
        std::vector<int> targets;

        [[nodiscard]] long long original_vars() const {
            return static_cast<long long>(sources.size()) * static_cast<long long>(targets.size());
        }

        [[nodiscard]] long long saved_vars() const {
            return original_vars() - static_cast<long long>(sources.size() + targets.size());
        }
    };

    std::vector<Candidate> candidates;
    std::unordered_set<std::vector<int>, VectorHash> seen_candidates;
    const auto add_candidate = [&](std::vector<int> sources, std::vector<int> targets, const bool left) {
        std::ranges::sort(sources);
        std::ranges::sort(targets);
        Candidate candidate{
            .left = left,
            .sources = std::move(sources),
            .targets = std::move(targets),
        };
        if (candidate.saved_vars() <= 0)
            return;

        std::vector<int> key;
        key.reserve(candidate.sources.size() + candidate.targets.size() + 3);
        key.push_back(candidate.left ? 1 : 0);
        key.push_back(static_cast<int>(candidate.sources.size()));
        key.insert(key.end(), candidate.sources.begin(), candidate.sources.end());
        key.insert(key.end(), candidate.targets.begin(), candidate.targets.end());
        if (seen_candidates.emplace(std::move(key)).second)
            candidates.push_back(std::move(candidate));
    };
    const auto collect_same_targets = [&](const std::vector<std::vector<int>>& targets_by_source, const bool left) {
        std::unordered_map<std::vector<int>, std::vector<int>, VectorHash> groups;
        groups.reserve(targets_by_source.size());
        for (int source = 0; source < static_cast<int>(targets_by_source.size()); ++source) {
            std::vector<int> targets = targets_by_source[source];
            if (targets.size() <= 1)
                continue;
            groups[std::move(targets)].push_back(source);
        }

        for (auto& [targets, sources] : groups) {
            if (sources.size() <= 1)
                continue;
            add_candidate(std::move(sources), targets, left);
        }
    };
    const auto collect_same_sources = [&](const std::vector<std::vector<int>>& sources_by_target, const bool left) {
        std::unordered_map<std::vector<int>, std::vector<int>, VectorHash> groups;
        groups.reserve(sources_by_target.size());
        for (int target = 0; target < static_cast<int>(sources_by_target.size()); ++target) {
            std::vector<int> sources = sources_by_target[target];
            if (sources.size() <= 1)
                continue;
            groups[std::move(sources)].push_back(target);
        }

        for (auto& [sources, targets] : groups) {
            if (targets.size() <= 1)
                continue;
            add_candidate(sources, std::move(targets), left);
        }
    };
    collect_same_targets(targets_by_source_left, true);
    collect_same_targets(targets_by_source_right, false);
    collect_same_sources(sources_by_target_left, true);
    collect_same_sources(sources_by_target_right, false);
    const auto collect_closed_pair_rectangles = [&](
        const std::vector<std::vector<int>>& targets_by_source,
        const std::vector<std::vector<int>>& sources_by_target,
        const bool left
    ) {
        struct PairSeed {
            std::uint64_t key = 0;
            std::vector<int> sources;
        };

        std::unordered_map<std::uint64_t, std::vector<int>> sources_by_pair;
        for (int source = 0; source < static_cast<int>(targets_by_source.size()); ++source) {
            const std::vector<int>& targets = targets_by_source[source];
            if (targets.size() < 2 ||
                targets.size() > constants::compact_rectangle_pair_seed_max_degree) {
                continue;
            }
            for (int i = 0; i + 1 < static_cast<int>(targets.size()); ++i) {
                for (int j = i + 1; j < static_cast<int>(targets.size()); ++j)
                    sources_by_pair[unordered_pair_key(targets[i], targets[j])].push_back(source);
            }
        }

        std::vector<PairSeed> seeds;
        seeds.reserve(sources_by_pair.size());
        for (auto& [key, sources] : sources_by_pair) {
            if (static_cast<int>(sources.size()) < constants::compact_rectangle_pair_seed_min_sources)
                continue;
            seeds.push_back(PairSeed{.key = key, .sources = std::move(sources)});
        }
        std::ranges::sort(seeds, [](const PairSeed& lhs, const PairSeed& rhs) {
            if (lhs.sources.size() != rhs.sources.size())
                return lhs.sources.size() > rhs.sources.size();
            return lhs.key < rhs.key;
        });

        int attempted = 0;
        for (PairSeed& seed : seeds) {
            if (attempted >= constants::compact_rectangle_pair_seed_candidate_limit)
                break;
            std::ranges::sort(seed.sources);
            seed.sources.erase(std::unique(seed.sources.begin(), seed.sources.end()), seed.sources.end());
            if (static_cast<int>(seed.sources.size()) < constants::compact_rectangle_pair_seed_min_sources)
                continue;

            std::vector<int> targets = targets_by_source[seed.sources.front()];
            for (int i = 1; i < static_cast<int>(seed.sources.size()) && targets.size() > 1; ++i)
                intersect_sorted_in_place(targets, targets_by_source[seed.sources[i]]);
            if (targets.size() <= 1)
                continue;

            std::vector<int> sources = sources_by_target[targets.front()];
            for (int i = 1; i < static_cast<int>(targets.size()) && sources.size() > 1; ++i)
                intersect_sorted_in_place(sources, sources_by_target[targets[i]]);
            if (sources.size() <= 1)
                continue;

            ++attempted;
            add_candidate(std::move(sources), std::move(targets), left);
        }
    };
    collect_closed_pair_rectangles(targets_by_source_left, sources_by_target_left, true);
    collect_closed_pair_rectangles(targets_by_source_right, sources_by_target_right, false);

    if (candidates.empty())
        return model;

    const auto better_candidate = [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.saved_vars() != rhs.saved_vars())
            return lhs.saved_vars() > rhs.saved_vars();
        if (lhs.original_vars() != rhs.original_vars())
            return lhs.original_vars() > rhs.original_vars();
        if (lhs.sources.size() != rhs.sources.size())
            return lhs.sources.size() > rhs.sources.size();
        return lhs.targets.size() > rhs.targets.size();
    };
    std::ranges::sort(candidates, better_candidate);

    CompactModel folded;
    folded.row_lower = model.row_lower;
    folded.row_upper = model.row_upper;
    folded.objective_offset = model.objective_offset;
    folded.capacity_vertex_stride = vertex_stride;
    folded.start.reserve(model.start.size());
    folded.col_cost.reserve(model.cols());
    folded.col_lower.reserve(model.cols());
    folded.col_upper.reserve(model.cols());
    folded.col_type.reserve(model.cols());
    folded.variables.reserve(model.cols());
    folded.column_kind.reserve(model.cols());
    folded.column_factor.reserve(model.cols());

    std::vector<char> covered(model.cols(), false);
    const ModelRows rows = rows_for_states(states);
    ColumnScratch scratch;

    const auto add_folded_candidate = [&](Candidate& candidate) {
        std::vector<int> transition_cols;
        transition_cols.reserve(candidate.sources.size() * candidate.targets.size());
        bool overlaps = false;
        for (const int source : candidate.sources) {
            for (const int target : candidate.targets) {
                const auto it = transition_to_col.find(transition_key(candidate.left, source, target));
                if (it == transition_to_col.end())
                    throw std::runtime_error("compact rectangle folding lost transition column");
                if (covered[it->second]) {
                    overlaps = true;
                    break;
                }
                transition_cols.push_back(it->second);
            }
            if (overlaps)
                break;
        }
        if (overlaps)
            return;

        const auto sides = rectangle_capacity_sides(
            trees,
            states,
            capacity_index,
            candidate.sources,
            candidate.targets,
            candidate.left,
            vertex_stride
        );
        if (!sides.has_value())
            return;

        const int factor = static_cast<int>(folded.rectangles.size());
        const int equality_row = add_row(folded, 0.0, 0.0);
        RectangleFactor rectangle{
            .left = candidate.left,
            .sources = std::move(candidate.sources),
            .targets = std::move(candidate.targets),
            .source_cols = {},
            .target_cols = {},
            .capacity_side = std::move(*sides),
        };
        folded.rectangles.push_back(std::move(rectangle));
        RectangleFactor& stored = folded.rectangles.back();

        for (const int source : stored.sources) {
            const int col = folded.cols();
            append_rectangle_source_column(
                folded,
                rows,
                states,
                factor,
                stored.left,
                source,
                equality_row,
                scratch
            );
            stored.source_cols.push_back(col);
        }
        for (const int target : stored.targets) {
            const int col = folded.cols();
            append_rectangle_target_column(
                folded,
                rows,
                states,
                factor,
                stored.left,
                target,
                static_cast<int>(stored.sources.size()),
                equality_row,
                scratch
            );
            stored.target_cols.push_back(col);
        }

        const long long original_vars =
            static_cast<long long>(stored.sources.size()) * static_cast<long long>(stored.targets.size());
        folded.rectangle_original_vars += original_vars;
        folded.rectangle_folded_vars += original_vars -
            static_cast<long long>(stored.sources.size() + stored.targets.size());
        ++folded.rectangle_groups;

        for (const int col : transition_cols)
            covered[col] = true;
    };

    for (Candidate& candidate : candidates)
        add_folded_candidate(candidate);

    if (folded.rectangles.empty())
        return model;

    for (int col = 0; col < model.cols(); ++col) {
        if (!covered[col])
            append_copied_column(folded, model, col);
    }

    return folded;
}

[[nodiscard]] CompactPrepared prepare_compact_model(const AnnotatedInstance& instance) {
    CompactPrepared prepared;
    prepared.instance = instance;
    prepared.state_index = StateIndex{.leaves = prepared.instance.trees.front().leaves()};
    prepared.trees.reserve(prepared.instance.trees.size());
    for (const auto& tree : prepared.instance.trees)
        prepared.trees.emplace_back(tree, prepared.state_index);

    prepared.states = build_compact_states(prepared.state_index, prepared.trees);
    prepared.model = build_compact_model(prepared.state_index, prepared.states, prepared.trees);
    prepared.prune_stats = prune_dead_transition_variables(prepared.model, prepared.states);
    prepared.capacity_index = build_capacity_index(prepared.model, prepared.trees, prepared.states);
    prepared.model = fold_transition_rectangles(
        prepared.model,
        prepared.trees,
        prepared.states,
        prepared.capacity_index
    );
    prepared.capacity_index = build_capacity_index(prepared.model, prepared.trees, prepared.states);
    return prepared;
}

[[nodiscard]] SCIP_VAR* transformed_var(SCIP* scip, SCIP_VAR* var) {
    if (SCIPvarIsTransformed(var))
        return var;
    SCIP_VAR* transformed = nullptr;
    check(SCIPgetTransformedVar(scip, var, &transformed));
    if (transformed == nullptr)
        throw std::runtime_error("SCIP did not provide transformed compact variable");
    return transformed;
}

SCIP_RETCODE add_capacity_constraint(
    SCIP* scip,
    CompactSolveData& data,
    const int tree_index,
    const int vertex,
    bool* added_row = nullptr
);

[[nodiscard]] double original_objective(const CompactModel& model, const std::span<const double> values) {
    if (values.size() != model.col_cost.size())
        return kInf;

    double objective = model.objective_offset;
    for (int col = 0; col < model.cols(); ++col)
        objective += model.col_cost[col] * values[col];
    return objective;
}

Result reconstruct_solution(
    const int leaves,
    const int vertices,
    const std::span<const int> first_tree_lca,
    const CompactModel& model,
    const std::span<const double> value
) {
    UnionFind components(vertices);
    std::vector<char> active(vertices, false);
    std::vector<char> covered(leaves);
    std::vector<int> singleton_in(leaves);

    const auto add_transition = [&](const Variable& variable) {
        const int top = first_tree_lca[variable.source];
        const int bottom = first_tree_lca[variable.target];
        components.join(top, bottom);
        active[top] = true;
        active[bottom] = true;
        if (bottom < leaves)
            ++singleton_in[bottom];
    };

    for (int i = 0; i < model.cols(); ++i) {
        if (model.column_kind[i] != CompactColumnKind::Transition || value[i] <= 0.5)
            continue;
        add_transition(model.variables[i]);
    }

    for (const RectangleFactor& rectangle : model.rectangles) {
        std::vector<int> sources;
        int supply = 0;
        for (const int col : rectangle.source_cols) {
            const int amount = static_cast<int>(std::llround(value[col]));
            if (amount < 0 || amount > 1)
                return {};
            if (amount == 0)
                continue;
            sources.push_back(model.variables[col].source);
            ++supply;
        }

        std::vector<int> target_demand;
        target_demand.reserve(rectangle.target_cols.size());
        int demand = 0;
        for (const int col : rectangle.target_cols) {
            const int amount = static_cast<int>(std::llround(value[col]));
            if (amount < 0)
                return {};
            target_demand.push_back(amount);
            demand += amount;
        }
        if (supply != demand)
            return {};

        int source_pos = 0;
        for (int target_pos = 0; target_pos < static_cast<int>(rectangle.target_cols.size()); ++target_pos) {
            const int target = model.variables[rectangle.target_cols[target_pos]].target;
            for (int copy = 0; copy < target_demand[target_pos]; ++copy) {
                if (source_pos >= static_cast<int>(sources.size()))
                    return {};
                add_transition(Variable{.source = sources[source_pos++], .target = target, .left = rectangle.left});
            }
        }
    }

    std::unordered_map<int, std::vector<int>> blocks;
    for (int node = 0; node < vertices; ++node) {
        if (!active[node])
            continue;
        if (node >= leaves)
            continue;
        auto& block = blocks[components.find(node)];
        block.push_back(node);
    }

    Result result;
    result.feasible = true;
    result.partition.reserve(blocks.size() + leaves);
    for (auto& [_, block] : blocks) {
        std::ranges::sort(block);
        block.erase(std::unique(block.begin(), block.end()), block.end());
        if (block.empty())
            continue;
        for (const int leaf : block)
            covered[leaf] = true;
        result.partition.push_back(std::move(block));
    }

    for (int leaf = 0; leaf < leaves; ++leaf) {
        if (singleton_in[leaf] == 0 && !covered[leaf])
            result.partition.push_back({leaf});
    }

    std::ranges::sort(result.partition, [](const auto& a, const auto& b) { return a.front() < b.front(); });

    std::vector<int> cover_count(leaves);
    for (const auto& block : result.partition) {
        for (const int leaf : block)
            ++cover_count[leaf];
    }
    if (std::ranges::any_of(cover_count, [](const int count) { return count != 1; }))
        return {};
    return result;
}

[[nodiscard]] bool valid_result_for_instance(
    const AnnotatedInstance& instance,
    const Result& result
) {
    if (!result.feasible || result.partition.empty())
        return false;

    const int leaves = instance.trees.front().leaves();
    std::vector<int> cover_count(leaves);
    for (const auto& block : result.partition) {
        if (block.empty())
            return false;
        for (const int leaf : block) {
            if (leaf < 0 || leaf >= leaves)
                return false;
            ++cover_count[leaf];
        }
    }
    if (std::ranges::any_of(cover_count, [](const int count) { return count != 1; }))
        return false;

    const auto views = detail::build_binary_tree_views(instance);
    return detail::partition_feasible(views, result.partition);
}

[[nodiscard]] std::optional<Result> valid_root_lp_heuristic_incumbent(
    const AnnotatedInstance& instance,
    const RootLpResult& root_lp_result
) {
    if (!root_lp_result.heuristic_solution.has_value())
        return std::nullopt;
    if (!valid_result_for_instance(instance, *root_lp_result.heuristic_solution))
        return std::nullopt;
    return root_lp_result.heuristic_solution;
}

[[nodiscard]] Result reconstruct_valid_solution(
    const AnnotatedInstance& instance,
    const StateIndex& state_index,
    const CompactStates& states,
    const CompactModel& model,
    const std::span<const double> original_values,
    const double expected_objective = kInf
) {
    if (original_values.size() != static_cast<std::size_t>(model.cols()))
        return {};
    if (std::isfinite(expected_objective) &&
        std::abs(original_objective(model, original_values) - expected_objective) > 1e-5)
        return {};

    std::vector<int> first_tree_lca(states.states());
    for (int state = 0; state < states.states(); ++state)
        first_tree_lca[state] = states.tree_lca[state].front();

    Result result = reconstruct_solution(
        state_index.leaves,
        instance.trees.front().vertices(),
        first_tree_lca,
        model,
        original_values
    );
    if (!valid_result_for_instance(instance, result))
        return {};
    return result;
}

} // namespace

struct CompactSolveData {
    const CompactModel* model = nullptr;
    const std::vector<TreeData>* trees = nullptr;
    const CapacityIndex* capacity_index = nullptr;
    const CompactStates* states = nullptr;
    std::vector<SCIP_VAR*>* vars = nullptr;
    std::vector<std::vector<char>> capacity_added;
    std::set<std::vector<int>> capacity_row_signatures;
    LogLevel log_level = LogLevel::QUIET;
    std::string_view log_prefix = "compact-scip";
    int separated_capacity_rows = 0;
    int capacity_separation_calls = 0;
    double capacity_separation_seconds = 0.0;
};

namespace {

[[nodiscard]] CapacityColumns build_root_left_rows(
    const CompactModel& model,
    const std::vector<TreeData>& trees,
    const CompactStates& states
) {
    CapacityColumns root_left;
    root_left.reserve(trees.size());
    for (const auto& tree : trees)
        root_left.emplace_back(tree.vertices);

    for (int col = 0; col < model.cols(); ++col) {
        const Variable& variable = model.variables[col];
        if (model.column_kind[col] == CompactColumnKind::RectangleTarget)
            continue;
        if (!variable.left || !internal_state(variable.source, states.first_leaf, states.last_leaf))
            continue;

        for (int tree_index = 0; tree_index < static_cast<int>(trees.size()); ++tree_index) {
            const int root = states.tree_lca[variable.source][tree_index];
            root_left[tree_index][root].push_back(col);
        }
    }

    simplify_at_most_one_rows(root_left, model.cols());
    return root_left;
}

void log_compact_size_analysis(
    const char* const prefix,
    const CompactPrepared& prepared
) {
    logging::line(
        prefix, "-rect-fold: groups=", prepared.model.rectangle_groups,
        " original-vars=", prepared.model.rectangle_original_vars,
        " saved-vars=", prepared.model.rectangle_folded_vars,
        " vars-after=", prepared.model.cols()
    );
}

void add_root_left_constraints(
    SCIP* scip,
    const CompactModel& model,
    const std::vector<TreeData>& trees,
    const CompactStates& states,
    ScipModel& scip_model
) {
    CapacityColumns root_left = build_root_left_rows(model, trees, states);
    for (int tree_index = 0; tree_index < static_cast<int>(root_left.size()); ++tree_index) {
        for (int root = trees[tree_index].helper.leaves(); root < static_cast<int>(root_left[tree_index].size()); ++root) {
            const auto& cols = root_left[tree_index][root];
            if (cols.size() <= 1)
                continue;

            std::vector<SCIP_VAR*> vars;
            vars.reserve(cols.size());
            for (const int col : cols)
                vars.push_back(scip_model.vars[col]);
            SCIP_CONS* cons = nullptr;
            const std::string name = std::format("root_left_{}_{}", tree_index, root);
            check(SCIPcreateConsBasicSetpack(
                scip,
                &cons,
                name.c_str(),
                static_cast<int>(vars.size()),
                vars.data()
            ));
            check(SCIPaddCons(scip, cons));
            check(SCIPreleaseCons(scip, &cons));
            ++scip_model.root_left_rows;
            scip_model.root_left_nonzeros += vars.size();
        }
    }
}

ScipModel pass_model_to_scip(
    SCIP* scip,
    const CompactModel& model,
    const std::vector<TreeData>& trees,
    const CompactStates& states,
    const bool add_root_left = true
) {
    ScipModel scip_model;
    scip_model.vars.reserve(model.cols());

    for (int col = 0; col < model.cols(); ++col) {
        SCIP_VAR* var = nullptr;
        const std::string name = std::format("x_{}", col);
        check(SCIPcreateVarBasic(
            scip,
            &var,
            name.c_str(),
            model.col_lower[col],
            model.col_upper[col],
            model.col_cost[col],
            model.col_type[col] == CompactVarType::Binary ? SCIP_VARTYPE_BINARY : SCIP_VARTYPE_INTEGER
        ));
        check(SCIPaddVar(scip, var));
        scip_model.vars.push_back(var);
        check(SCIPreleaseVar(scip, &var));
    }

    std::vector<SCIP_CONS*> rows(model.rows(), nullptr);
    for (int row = 0; row < model.rows(); ++row) {
        SCIP_CONS* cons = nullptr;
        const std::string name = std::format("flow_{}", row);
        check(SCIPcreateConsBasicLinear(
            scip,
            &cons,
            name.c_str(),
            0,
            nullptr,
            nullptr,
            std::isinf(model.row_lower[row]) && model.row_lower[row] < 0.0 ? -SCIPinfinity(scip) : model.row_lower[row],
            std::isinf(model.row_upper[row]) && model.row_upper[row] > 0.0 ? SCIPinfinity(scip) : model.row_upper[row]
        ));
        check(SCIPaddCons(scip, cons));
        rows[row] = cons;
    }

    for (int col = 0; col < model.cols(); ++col) {
        for (int p = model.start[col]; p < model.start[col + 1]; ++p)
            check(SCIPaddCoefLinear(scip, rows[model.index[p]], scip_model.vars[col], model.value[p]));
    }

    for (SCIP_CONS*& cons : rows)
        check(SCIPreleaseCons(scip, &cons));

    if (add_root_left)
        add_root_left_constraints(scip, model, trees, states, scip_model);
    return scip_model;
}

[[nodiscard]] bool install_zero_compact_solution(
    SCIP* scip,
    const std::vector<SCIP_VAR*>& vars
) {
    SCIP_SOL* sol = nullptr;
    check(SCIPcreateSol(scip, &sol, nullptr));
    try {
        for (SCIP_VAR* var : vars)
            check(SCIPsetSolVal(scip, sol, var, 0.0));

        SCIP_Bool stored = FALSE;
        check(SCIPaddSolFree(scip, &sol, &stored));
        return stored;
    } catch (...) {
        if (sol != nullptr)
            ignore_unused(SCIPfreeSol(scip, &sol));
        throw;
    }
}

struct CompactTransitionValueTarget {
    int transition_col = -1;
    int rectangle_source_col = -1;
    int rectangle_target_col = -1;
};

[[nodiscard]] std::unordered_map<std::uint64_t, CompactTransitionValueTarget> compact_transition_value_targets(
    const CompactModel& model
) {
    std::unordered_map<std::uint64_t, CompactTransitionValueTarget> targets;
    targets.reserve(model.cols() + static_cast<std::size_t>(model.rectangle_original_vars));
    for (int col = 0; col < model.cols(); ++col) {
        if (model.column_kind[col] != CompactColumnKind::Transition)
            continue;
        const Variable& variable = model.variables[col];
        targets.emplace(
            transition_key(variable.left, variable.source, variable.target),
            CompactTransitionValueTarget{.transition_col = col}
        );
    }

    for (const RectangleFactor& rectangle : model.rectangles) {
        for (int source_pos = 0; source_pos < static_cast<int>(rectangle.sources.size()); ++source_pos) {
            const int source = rectangle.sources[source_pos];
            for (int target_pos = 0; target_pos < static_cast<int>(rectangle.targets.size()); ++target_pos) {
                const int target = rectangle.targets[target_pos];
                targets.emplace(
                    transition_key(rectangle.left, source, target),
                    CompactTransitionValueTarget{
                        .rectangle_source_col = rectangle.source_cols[source_pos],
                        .rectangle_target_col = rectangle.target_cols[target_pos],
                    }
                );
            }
        }
    }
    return targets;
}

[[nodiscard]] bool add_compact_transition_value(
    const std::unordered_map<std::uint64_t, CompactTransitionValueTarget>& targets,
    std::vector<double>& values,
    const bool left,
    const int source,
    const int target
) {
    const auto it = targets.find(transition_key(left, source, target));
    if (it == targets.end())
        return false;

    const CompactTransitionValueTarget& value_target = it->second;
    if (value_target.transition_col >= 0) {
        if (values[value_target.transition_col] > 0.5)
            return false;
        values[value_target.transition_col] = 1.0;
        return true;
    }

    if (value_target.rectangle_source_col < 0 || value_target.rectangle_target_col < 0)
        return false;
    if (values[value_target.rectangle_source_col] > 0.5)
        return false;
    values[value_target.rectangle_source_col] = 1.0;
    values[value_target.rectangle_target_col] += 1.0;
    return true;
}

struct CompactStartPlan {
    int state = -1;
    std::shared_ptr<const CompactStartPlan> left;
    std::shared_ptr<const CompactStartPlan> right;
};

struct CompactStartStats {
    std::string failure;
    int block_size = 0;
    int plans = 0;
};

void record_compact_start_failure(
    CompactStartStats* stats,
    std::string failure,
    const int block_size = 0,
    const int plans = 0
) {
    if (stats == nullptr || !stats->failure.empty())
        return;
    stats->failure = std::move(failure);
    stats->block_size = block_size;
    stats->plans = plans;
}

[[nodiscard]] bool compact_transition_exists(
    const std::unordered_map<std::uint64_t, CompactTransitionValueTarget>& targets,
    const bool left,
    const int source,
    const int target
) {
    return targets.contains(transition_key(left, source, target));
}

[[nodiscard]] std::vector<std::shared_ptr<const CompactStartPlan>> compact_start_plans(
    const StateIndex& state_index,
    const CompactStates& states,
    const std::vector<TreeData>& trees,
    const std::vector<std::array<int, 2>>& children,
    const std::unordered_map<std::uint64_t, CompactTransitionValueTarget>& transition_targets,
    std::vector<int> leaves
) {
    std::ranges::sort(leaves);
    leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
    if (leaves.empty())
        return {};
    if (leaves.size() == 1) {
        return {
            std::make_shared<CompactStartPlan>(CompactStartPlan{
                .state = states.original_to_state[state_index(leaves.front(), leaves.front())],
                .left = nullptr,
                .right = nullptr,
            }),
        };
    }

    int root = leaves.front();
    for (std::size_t pos = 1; pos < leaves.size(); ++pos)
        root = trees.front().helper.lca(root, leaves[pos]);
    if (root < state_index.leaves || root >= static_cast<int>(children.size()))
        return {};

    const auto [left_child, right_child] = children[root];
    if (left_child < 0 || right_child < 0)
        return {};

    std::vector<int> left_leaves;
    std::vector<int> right_leaves;
    left_leaves.reserve(leaves.size());
    right_leaves.reserve(leaves.size());
    for (const int leaf : leaves) {
        if (trees.front().helper.ancestor(left_child, leaf))
            left_leaves.push_back(leaf);
        else if (trees.front().helper.ancestor(right_child, leaf))
            right_leaves.push_back(leaf);
        else
            return {};
    }
    if (left_leaves.empty() || right_leaves.empty())
        return {};

    const auto left_plans = compact_start_plans(
        state_index,
        states,
        trees,
        children,
        transition_targets,
        left_leaves
    );
    const auto right_plans = compact_start_plans(
        state_index,
        states,
        trees,
        children,
        transition_targets,
        right_leaves
    );
    if (left_plans.empty() || right_plans.empty())
        return {};

    std::vector<int> source_states;
    source_states.reserve(left_leaves.size() * right_leaves.size());
    std::unordered_set<int> seen_sources;
    seen_sources.reserve(left_leaves.size() * right_leaves.size());
    for (const int left_leaf : left_leaves) {
        for (const int right_leaf : right_leaves) {
            const int a = std::min(left_leaf, right_leaf);
            const int b = std::max(left_leaf, right_leaf);
            const int state = states.original_to_state[state_index(a, b)];
            if (seen_sources.emplace(state).second)
                source_states.push_back(state);
        }
    }
    std::ranges::sort(source_states);

    std::vector<std::shared_ptr<const CompactStartPlan>> plans;
    plans.reserve(source_states.size());
    for (const int source : source_states) {
        std::unordered_set<std::uint64_t> seen_plan_states;
        const auto add_orientation = [&](
            const std::vector<std::shared_ptr<const CompactStartPlan>>& left_side_plans,
            const std::vector<std::shared_ptr<const CompactStartPlan>>& right_side_plans
        ) {
            for (const auto& left_plan : left_side_plans) {
                if (!compact_transition_exists(transition_targets, true, source, left_plan->state))
                    continue;
                for (const auto& right_plan : right_side_plans) {
                    if (!compact_transition_exists(transition_targets, false, source, right_plan->state))
                        continue;
                    const std::uint64_t key =
                        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left_plan->state)) << 32) |
                        static_cast<std::uint64_t>(static_cast<std::uint32_t>(right_plan->state));
                    if (!seen_plan_states.emplace(key).second)
                        continue;
                    plans.push_back(std::make_shared<CompactStartPlan>(CompactStartPlan{
                        .state = source,
                        .left = left_plan,
                        .right = right_plan,
                    }));
                }
            }
        };
        add_orientation(left_plans, right_plans);
        add_orientation(right_plans, left_plans);
    }
    return plans;
}

[[nodiscard]] std::optional<std::vector<double>> compact_values_for_result(
    const CompactPrepared& prepared,
    const Result& result,
    CompactStartStats* stats = nullptr
) {
    if (stats != nullptr)
        *stats = {};
    if (!valid_result_for_instance(prepared.instance, result)) {
        record_compact_start_failure(stats, "invalid-result");
        return std::nullopt;
    }

    const auto children = detail::tree_children(
        prepared.instance.trees.front(),
        detail::CutHandling::INCLUDE_CUT
    );
    const auto transition_targets = compact_transition_value_targets(prepared.model);
    std::vector<double> values(prepared.model.cols(), 0.0);

    const auto emit = [&](this auto&& self, const CompactStartPlan& plan, std::vector<double>& target_values) -> bool {
        if (!plan.left && !plan.right)
            return true;
        if (!plan.left || !plan.right)
            return false;
        if (!add_compact_transition_value(transition_targets, target_values, true, plan.state, plan.left->state))
            return false;
        if (!add_compact_transition_value(transition_targets, target_values, false, plan.state, plan.right->state))
            return false;
        return self(*plan.left, target_values) && self(*plan.right, target_values);
    };

    for (const auto& block : result.partition) {
        const auto plans = compact_start_plans(
            prepared.state_index,
            prepared.states,
            prepared.trees,
            children,
            transition_targets,
            block
        );
        if (plans.empty()) {
            record_compact_start_failure(stats, "no-transition-plan", static_cast<int>(block.size()));
            return std::nullopt;
        }

        bool installed_block = false;
        for (const auto& plan : plans) {
            std::vector<double> trial = values;
            if (emit(*plan, trial)) {
                values = std::move(trial);
                installed_block = true;
                break;
            }
        }
        if (!installed_block) {
            record_compact_start_failure(
                stats,
                "transition-plan-conflict",
                static_cast<int>(block.size()),
                static_cast<int>(plans.size())
            );
            return std::nullopt;
        }
    }

    for (int col = 0; col < prepared.model.cols(); ++col) {
        if (values[col] < prepared.model.col_lower[col] - 1e-6 ||
            values[col] > prepared.model.col_upper[col] + 1e-6) {
            record_compact_start_failure(stats, "column-bound");
            return std::nullopt;
        }
    }

    const double expected_objective = static_cast<double>(result.partition.size());
    Result reconstructed = reconstruct_valid_solution(
        prepared.instance,
        prepared.state_index,
        prepared.states,
        prepared.model,
        values,
        expected_objective
    );
    if (!reconstructed.feasible || static_cast<int>(reconstructed.partition.size()) != static_cast<int>(result.partition.size())) {
        record_compact_start_failure(stats, "reconstruction-mismatch");
        return std::nullopt;
    }
    return values;
}

struct CompactRootLpSeed {
    RootLpResult root_lp;
    std::optional<Result> incumbent;
    std::optional<std::vector<double>> start_values;
    CompactStartStats start_stats;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point done;
    int incumbent_components = -1;
    double incumbent_objective = kInf;
};

[[nodiscard]] CompactRootLpSeed build_compact_root_lp_seed(
    const CompactPrepared& prepared,
    const int objective_offset,
    const std::optional<std::chrono::steady_clock::time_point> deadline
) {
    CompactRootLpSeed seed;
    seed.start = std::chrono::steady_clock::now();
    seed.root_lp = solve_root_lp_with_highs(
        prepared.instance,
        LogLevel::QUIET,
        {},
        objective_offset,
        remaining_seconds(deadline),
        true
    );
    seed.done = std::chrono::steady_clock::now();
    seed.incumbent = valid_root_lp_heuristic_incumbent(prepared.instance, seed.root_lp);
    if (seed.incumbent.has_value()) {
        seed.incumbent_components = static_cast<int>(seed.incumbent->partition.size());
        seed.incumbent_objective = static_cast<double>(seed.incumbent_components + objective_offset);
        seed.start_values = compact_values_for_result(prepared, *seed.incumbent, &seed.start_stats);
    }
    return seed;
}

[[nodiscard]] bool install_compact_solution(
    SCIP* scip,
    const std::vector<SCIP_VAR*>& vars,
    const std::span<const double> values
) {
    if (values.size() != vars.size())
        return false;

    SCIP_SOL* sol = nullptr;
    check(SCIPcreateSol(scip, &sol, nullptr));
    try {
        for (int col = 0; col < static_cast<int>(vars.size()); ++col)
            check(SCIPsetSolVal(scip, sol, vars[col], values[col]));

        SCIP_Bool stored = FALSE;
        check(SCIPaddSolFree(scip, &sol, &stored));
        return stored;
    } catch (...) {
        if (sol != nullptr)
            ignore_unused(SCIPfreeSol(scip, &sol));
        throw;
    }
}

[[nodiscard]] std::vector<RowRef> nonempty_rows(const CapacityColumns& rows) {
    std::vector<RowRef> refs;
    for (int group = 0; group < static_cast<int>(rows.size()); ++group) {
        for (int row = 0; row < static_cast<int>(rows[group].size()); ++row) {
            if (!rows[group][row].empty())
                refs.push_back(RowRef{.group = group, .row = row});
        }
    }
    return refs;
}

void simplify_at_most_one_rows(CapacityColumns& rows, const int col_count) {
    std::set<std::vector<int>> seen;
    for (auto& group_rows : rows) {
        for (auto& cols : group_rows) {
            if (cols.size() <= 1) {
                cols.clear();
                continue;
            }
            if (!seen.insert(cols).second)
                cols.clear();
        }
    }

    const std::vector<RowRef> refs = nonempty_rows(rows);
    std::vector<std::vector<int>> rows_by_col(col_count);
    for (int id = 0; id < static_cast<int>(refs.size()); ++id) {
        const auto& cols = rows[refs[id].group][refs[id].row];
        for (const int col : cols)
            rows_by_col[col].push_back(id);
    }

    for (int id = 0; id < static_cast<int>(refs.size()); ++id) {
        auto& cols = rows[refs[id].group][refs[id].row];
        if (cols.empty())
            continue;

        int rarest_col = -1;
        std::size_t rarest_count = std::numeric_limits<std::size_t>::max();
        for (const int col : cols) {
            if (rows_by_col[col].size() < rarest_count) {
                rarest_col = col;
                rarest_count = rows_by_col[col].size();
            }
        }

        bool dominated = false;
        for (const int candidate : rows_by_col[rarest_col]) {
            if (candidate == id)
                continue;
            const auto& superset = rows[refs[candidate].group][refs[candidate].row];
            if (superset.size() <= cols.size())
                continue;
            if (std::includes(superset.begin(), superset.end(), cols.begin(), cols.end())) {
                dominated = true;
                break;
            }
        }
        if (dominated)
            cols.clear();
    }
}

[[nodiscard]] std::vector<int> materialize_capacity_row(
    const CompactModel& model,
    const std::vector<TreeData>& trees,
    const CompactStates& states,
    const CapacityIndex& capacity_index,
    const int tree_index,
    const int vertex
) {
    std::vector<int> cols;
    if (capacity_index.row_nonzeros[tree_index][vertex] <= 1)
        return cols;

    cols.reserve(capacity_index.row_nonzeros[tree_index][vertex]);
    for (int col = 0; col < model.cols(); ++col) {
        if (variable_uses_capacity_vertex(model, trees, states, col, tree_index, vertex))
            cols.push_back(col);
    }
    return cols;
}

[[nodiscard]] std::vector<int> materialize_capacity_row(
    const CompactSolveData& data,
    const int tree_index,
    const int vertex
) {
    return materialize_capacity_row(
        *data.model,
        *data.trees,
        *data.states,
        *data.capacity_index,
        tree_index,
        vertex
    );
}

[[nodiscard]] std::vector<double> solution_values(
    SCIP* scip,
    SCIP_SOL* sol,
    const CompactSolveData& data
) {
    std::vector<double> values;
    values.reserve(data.vars->size());
    for (SCIP_VAR* var : *data.vars)
        values.push_back(SCIPgetSolVal(scip, sol, var));
    return values;
}

[[nodiscard]] std::vector<double> capacity_usage(
    const CompactModel& model,
    const std::vector<TreeData>& trees,
    const CompactStates& states,
    const CapacityIndex& capacity_index,
    const std::span<const double> values,
    const int tree_index
) {
    const TreeData& tree = trees[tree_index];

    std::vector<double> usage(tree.vertices);
    for (int col = 0; col < static_cast<int>(values.size()); ++col) {
        const double x = values[col];
        if (x <= constants::compact_capacity_tol)
            continue;
        for_each_capacity_vertex(model, trees, states, col, tree_index, [&](const int vertex) {
            if (capacity_index.row_nonzeros[tree_index][vertex] > 1)
                usage[vertex] += x;
        });
    }
    return usage;
}

[[nodiscard]] std::vector<double> capacity_usage(
    const CompactSolveData& data,
    const std::span<const double> values,
    const int tree_index
) {
    return capacity_usage(
        *data.model,
        *data.trees,
        *data.states,
        *data.capacity_index,
        values,
        tree_index
    );
}

SCIP_RETCODE add_capacity_constraint_impl(
    SCIP* scip,
    CompactSolveData& data,
    const int tree_index,
    const int vertex,
    std::vector<int> cols,
    const bool transformed,
    const bool count_separated,
    bool* const added_row = nullptr
) {
    if (added_row != nullptr)
        *added_row = false;
    auto& added = data.capacity_added[tree_index][vertex];
    if (added)
        return SCIP_OKAY;
    added = true;
    if (cols.size() <= 1)
        return SCIP_OKAY;
    if (!data.capacity_row_signatures.insert(cols).second)
        return SCIP_OKAY;

    std::vector<SCIP_VAR*> vars;
    vars.reserve(cols.size());
    std::vector<SCIP_Real> coefficients(cols.size(), 1.0);
    bool all_binary = true;

    for (const int col : cols) {
        SCIP_VAR* var = (*data.vars)[col];
        vars.push_back(transformed ? transformed_var(scip, var) : var);
        all_binary = all_binary && data.model->col_type[col] == CompactVarType::Binary;
    }

    SCIP_CONS* cons = nullptr;
    const std::string name = std::format(
        "{}_{}_{}",
        count_separated ? "capacity" : "capacity_seed",
        tree_index,
        vertex
    );
    if (!all_binary) {
        if (transformed) {
            SCIP_CALL(SCIPcreateConsLinear(
                scip,
                &cons,
                name.c_str(),
                static_cast<int>(vars.size()),
                vars.data(),
                coefficients.data(),
                0.0,
                1.0,
                TRUE,
                TRUE,
                TRUE,
                TRUE,
                TRUE,
                FALSE,
                FALSE,
                FALSE,
                FALSE,
                FALSE
            ));
        } else {
            SCIP_CALL(SCIPcreateConsBasicLinear(
                scip,
                &cons,
                name.c_str(),
                static_cast<int>(vars.size()),
                vars.data(),
                coefficients.data(),
                0.0,
                1.0
            ));
        }
    } else if (transformed) {
        SCIP_CALL(SCIPcreateConsSetpack(
            scip,
            &cons,
            name.c_str(),
            static_cast<int>(vars.size()),
            vars.data(),
            TRUE,
            TRUE,
            TRUE,
            TRUE,
            TRUE,
            FALSE,
            FALSE,
            FALSE,
            FALSE,
            FALSE
        ));
    } else {
        SCIP_CALL(SCIPcreateConsBasicSetpack(
            scip,
            &cons,
            name.c_str(),
            static_cast<int>(vars.size()),
            vars.data()
        ));
    }
    SCIP_CALL(SCIPaddCons(scip, cons));
    SCIP_CALL(SCIPreleaseCons(scip, &cons));

    if (added_row != nullptr)
        *added_row = true;
    if (count_separated)
        ++data.separated_capacity_rows;
    return SCIP_OKAY;
}

SCIP_RETCODE add_capacity_constraint_impl(
    SCIP* scip,
    CompactSolveData& data,
    const int tree_index,
    const int vertex,
    const bool transformed,
    const bool count_separated,
    bool* const added_row = nullptr
) {
    return add_capacity_constraint_impl(
        scip,
        data,
        tree_index,
        vertex,
        materialize_capacity_row(data, tree_index, vertex),
        transformed,
        count_separated,
        added_row
    );
}

SCIP_RETCODE add_capacity_constraint(
    SCIP* scip,
    CompactSolveData& data,
    const int tree_index,
    const int vertex,
    bool* const added_row
) {
    return add_capacity_constraint_impl(scip, data, tree_index, vertex, true, true, added_row);
}

[[nodiscard]] double root_lp_capacity_dual(
    const RootLpResult& root_lp,
    const int tree_index,
    const int vertex
) {
    if (tree_index >= static_cast<int>(root_lp.vertex_duals.size()))
        return 0.0;
    if (vertex >= static_cast<int>(root_lp.vertex_duals[tree_index].size()))
        return 0.0;
    return root_lp.vertex_duals[tree_index][vertex];
}

[[nodiscard]] std::vector<std::vector<double>> root_lp_vertex_support(
    const AnnotatedInstance& instance,
    const RootLpResult& root_lp
) {
    std::vector<std::vector<double>> support;
    support.reserve(instance.trees.size());
    for (const Tree& tree : instance.trees)
        support.emplace_back(tree.vertices(), 0.0);

    const int columns = std::min(
        static_cast<int>(root_lp.warm_start.columns.size()),
        static_cast<int>(root_lp.warm_start.column_values.size())
    );
    for (int col = 0; col < columns; ++col) {
        const double value = root_lp.warm_start.column_values[col];
        if (value <= constants::compact_capacity_tol)
            continue;
        const RootMasterColumn column = build_root_master_column(
            instance,
            root_lp.layout,
            root_lp.warm_start.columns[col]
        );
        for (int tree = 0; tree < static_cast<int>(column.used_vertices.size()); ++tree) {
            for (const int vertex : column.used_vertices[tree])
                support[tree][vertex] += value;
        }
    }
    return support;
}

[[nodiscard]] std::vector<CapacitySeedRow> select_root_lp_seed_capacity_rows(
    const CompactModel& model,
    const std::vector<TreeData>& trees,
    const CompactStates& states,
    const CapacityIndex& capacity_index,
    const AnnotatedInstance& instance,
    const RootLpResult& root_lp
) {
    const std::vector<std::vector<double>> support = root_lp_vertex_support(instance, root_lp);
    std::vector<CapacitySeedCandidate> candidates;
    for (int tree_index = 0; tree_index < static_cast<int>(trees.size()); ++tree_index) {
        const int leaves = trees[tree_index].helper.leaves();
        for (int vertex = leaves; vertex < trees[tree_index].vertices; ++vertex) {
            const int nonzeros = capacity_index.row_nonzeros[tree_index][vertex];
            if (nonzeros <= 1)
                continue;
            const double dual = root_lp_capacity_dual(root_lp, tree_index, vertex);
            const double row_support =
                tree_index < static_cast<int>(support.size()) && vertex < static_cast<int>(support[tree_index].size())
                ? support[tree_index][vertex]
                : 0.0;
            const bool has_dual = dual > constants::compact_seed_dual_tol;
            const bool has_support = row_support > constants::compact_seed_support_tol;
            if (!has_dual && !has_support)
                continue;

            const double dual_score = has_dual ? 1.0 + dual : 0.0;
            const double support_score = has_support ? row_support - constants::compact_seed_support_tol : 0.0;
            candidates.push_back(CapacitySeedCandidate{
                .tree = tree_index,
                .vertex = vertex,
                .nonzeros = nonzeros,
                .dual = dual,
                .support = row_support,
                .priority = (dual_score + support_score) / std::sqrt(static_cast<double>(nonzeros)),
            });
        }
    }

    std::ranges::sort(candidates, [](const CapacitySeedCandidate& lhs, const CapacitySeedCandidate& rhs) {
        const bool lhs_has_dual = lhs.dual > constants::compact_seed_dual_tol;
        const bool rhs_has_dual = rhs.dual > constants::compact_seed_dual_tol;
        if (lhs_has_dual != rhs_has_dual)
            return lhs_has_dual;
        if (lhs.priority != rhs.priority)
            return lhs.priority > rhs.priority;
        if (lhs.support != rhs.support)
            return lhs.support > rhs.support;
        return lhs.nonzeros < rhs.nonzeros;
    });

    std::vector<CapacitySeedCandidate> selected;
    selected.reserve(std::min(constants::compact_capacity_seed_row_limit, static_cast<int>(candidates.size())));
    std::size_t selected_nonzeros = 0;
    for (const CapacitySeedCandidate& candidate : candidates) {
        if (static_cast<int>(selected.size()) >= constants::compact_capacity_seed_row_limit)
            break;
        if (!selected.empty() && selected_nonzeros + static_cast<std::size_t>(candidate.nonzeros) > constants::compact_capacity_seed_nonzero_limit)
            continue;
        selected.push_back(candidate);
        selected_nonzeros += candidate.nonzeros;
    }

    std::vector<std::vector<int>> selected_row_index;
    selected_row_index.reserve(trees.size());
    for (const TreeData& tree : trees)
        selected_row_index.emplace_back(tree.vertices, -1);
    for (int i = 0; i < static_cast<int>(selected.size()); ++i)
        selected_row_index[selected[i].tree][selected[i].vertex] = i;

    std::vector<std::vector<int>> selected_cols(selected.size());
    for (int i = 0; i < static_cast<int>(selected.size()); ++i)
        selected_cols[i].reserve(selected[i].nonzeros);
    for (int col = 0; col < model.cols(); ++col) {
        for (int tree_index = 0; tree_index < static_cast<int>(trees.size()); ++tree_index) {
            for_each_capacity_vertex(model, trees, states, col, tree_index, [&](const int vertex) {
                const int row = selected_row_index[tree_index][vertex];
                if (row >= 0)
                    selected_cols[row].push_back(col);
            });
        }
    }

    std::vector<CapacitySeedRow> rows;
    rows.reserve(selected.size());
    for (int i = 0; i < static_cast<int>(selected.size()); ++i) {
        const CapacitySeedCandidate& candidate = selected[i];
        rows.push_back(CapacitySeedRow{
            .tree = candidate.tree,
            .vertex = candidate.vertex,
            .nonzeros = candidate.nonzeros,
            .cols = std::move(selected_cols[i]),
        });
    }
    return rows;
}

[[nodiscard]] CapacitySeedStats add_root_lp_seed_capacity_constraints(
    SCIP* scip,
    CompactSolveData& data,
    const AnnotatedInstance& instance,
    const RootLpResult& root_lp
) {
    std::vector<CapacitySeedRow> rows = select_root_lp_seed_capacity_rows(
        *data.model,
        *data.trees,
        *data.states,
        *data.capacity_index,
        instance,
        root_lp
    );

    CapacitySeedStats stats;
    for (CapacitySeedRow& row : rows) {
        bool added = false;
        check(add_capacity_constraint_impl(
            scip,
            data,
            row.tree,
            row.vertex,
            std::move(row.cols),
            false,
            false,
            &added
        ));
        if (!added)
            continue;
        ++stats.rows;
        stats.nonzeros += row.nonzeros;
    }
    return stats;
}

[[nodiscard]] bool has_capacity_violation(
    SCIP* scip,
    SCIP_SOL* sol,
    const CompactSolveData& data
) {
    const std::vector<double> values = solution_values(scip, sol, data);
    for (int tree_index = 0; tree_index < static_cast<int>(data.trees->size()); ++tree_index) {
        const std::vector<double> usage = capacity_usage(data, values, tree_index);
        const int leaves = (*data.trees)[tree_index].helper.leaves();
        for (int vertex = leaves; vertex < static_cast<int>(usage.size()); ++vertex) {
            if (usage[vertex] > 1.0 + constants::compact_capacity_tol)
                return true;
        }
    }
    return false;
}

[[nodiscard]] int capacity_separation_row_limit(SCIP* scip) {
    return SCIPgetStage(scip) == SCIP_STAGE_SOLVING && SCIPgetDepth(scip) == 0
        ? constants::compact_capacity_root_separation_row_limit
        : constants::compact_capacity_separation_row_limit;
}

SCIP_RETCODE separate_capacity_data(
    SCIP* scip,
    SCIP_SOL* sol,
    CompactSolveData& data,
    SCIP_RESULT* result
) {
    const auto start = std::chrono::steady_clock::now();
    *result = SCIP_DIDNOTFIND;
    const int row_limit = capacity_separation_row_limit(scip);

    const std::vector<double> values = solution_values(scip, sol, data);
    std::vector<CapacityViolation> violations;
    for (int tree_index = 0; tree_index < static_cast<int>(data.trees->size()); ++tree_index) {
        const std::vector<double> usage = capacity_usage(data, values, tree_index);
        const int leaves = (*data.trees)[tree_index].helper.leaves();
        for (int vertex = leaves; vertex < static_cast<int>(usage.size()); ++vertex) {
            if (data.capacity_added[tree_index][vertex])
                continue;
            if (usage[vertex] <= 1.0 + constants::compact_capacity_tol)
                continue;
            violations.push_back(CapacityViolation{
                .tree = tree_index,
                .vertex = vertex,
                .nonzeros = data.capacity_index->row_nonzeros[tree_index][vertex],
                .violation = usage[vertex] - 1.0,
            });
        }
    }
    std::ranges::sort(violations, [](const CapacityViolation& lhs, const CapacityViolation& rhs) {
        return lhs.violation > rhs.violation;
    });

    int added = 0;
    std::vector<char> selected(violations.size(), false);
    const int high_violation_limit = std::min(
        row_limit * 3 / 5,
        static_cast<int>(violations.size())
    );
    for (int i = 0; i < high_violation_limit; ++i) {
        bool row_added = false;
        SCIP_CALL(add_capacity_constraint(scip, data, violations[i].tree, violations[i].vertex, &row_added));
        selected[i] = true;
        if (row_added)
            ++added;
    }

    std::vector<int> density_order;
    density_order.reserve(violations.size());
    for (int i = 0; i < static_cast<int>(violations.size()); ++i) {
        if (!selected[i])
            density_order.push_back(i);
    }
    std::ranges::sort(density_order, [&](const int lhs, const int rhs) {
        const double lhs_density = violations[lhs].violation / std::max(1, violations[lhs].nonzeros);
        const double rhs_density = violations[rhs].violation / std::max(1, violations[rhs].nonzeros);
        if (lhs_density != rhs_density)
            return lhs_density > rhs_density;
        return violations[lhs].nonzeros < violations[rhs].nonzeros;
    });
    for (const int index : density_order) {
        if (added >= row_limit)
            break;
        bool row_added = false;
        SCIP_CALL(add_capacity_constraint(scip, data, violations[index].tree, violations[index].vertex, &row_added));
        if (row_added)
            ++added;
    }

    if (added > 0) {
        *result = SCIP_CONSADDED;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ++data.capacity_separation_calls;
    data.capacity_separation_seconds += seconds;
    if ((added > 0 || seconds > 1.0) && logging::enabled(data.log_level, LogLevel::VERBOSE)) {
        logging::line(
            data.log_prefix, ": separated capacity rows=", added,
            " violations=", violations.size(),
            " limit=", row_limit,
            " time=", seconds,
            "s"
        );
    }
    return SCIP_OKAY;
}

[[nodiscard]] std::vector<SCIP_VAR*> capacity_lock_vars(const CompactSolveData& data) {
    std::vector<char> used(data.vars->size(), false);
    for (int col = 0; col < data.model->cols(); ++col) {
        for (int tree_index = 0; tree_index < static_cast<int>(data.trees->size()); ++tree_index) {
            bool locked = false;
            for_each_capacity_vertex(*data.model, *data.trees, *data.states, col, tree_index, [&](const int vertex) {
                if (data.capacity_index->row_nonzeros[tree_index][vertex] > 1)
                    locked = true;
            });
            if (locked) {
                used[col] = true;
                break;
            }
        }
    }

    std::vector<SCIP_VAR*> vars;
    vars.reserve(data.vars->size());
    for (int col = 0; col < static_cast<int>(used.size()); ++col) {
        if (used[col])
            vars.push_back((*data.vars)[col]);
    }
    return vars;
}

SCIP_DECL_CONSFREE(capacity_cons_free) {
    ignore_unused(scip);
    delete SCIPconshdlrGetData(conshdlr);
    return SCIP_OKAY;
}

SCIP_DECL_CONSDELETE(capacity_cons_delete) {
    ignore_unused(scip, conshdlr, cons);
    delete *consdata;
    *consdata = nullptr;
    return SCIP_OKAY;
}

SCIP_DECL_CONSTRANS(capacity_cons_trans) {
    const auto* source_data = SCIPconsGetData(sourcecons);
    auto* target_data = new SCIP_CONSDATA{};
    if (source_data != nullptr) {
        target_data->vars.reserve(source_data->vars.size());
        for (SCIP_VAR* var : source_data->vars) {
            SCIP_VAR* transformed = nullptr;
            SCIP_CALL(SCIPgetTransformedVar(scip, var, &transformed));
            if (transformed != nullptr)
                target_data->vars.push_back(transformed);
        }
    }

    SCIP_CALL(SCIPcreateCons(
        scip,
        targetcons,
        SCIPconsGetName(sourcecons),
        conshdlr,
        target_data,
        SCIPconsIsInitial(sourcecons),
        SCIPconsIsSeparated(sourcecons),
        SCIPconsIsEnforced(sourcecons),
        SCIPconsIsChecked(sourcecons),
        SCIPconsIsPropagated(sourcecons),
        SCIPconsIsLocal(sourcecons),
        SCIPconsIsModifiable(sourcecons),
        SCIPconsIsDynamic(sourcecons),
        SCIPconsIsRemovable(sourcecons),
        SCIPconsIsStickingAtNode(sourcecons)
    ));
    return SCIP_OKAY;
}

SCIP_DECL_CONSENFOLP(capacity_cons_enfolp) {
    ignore_unused(conss, nconss, nusefulconss, solinfeasible);
    auto* handler_data = SCIPconshdlrGetData(conshdlr);
    if (handler_data == nullptr || handler_data->data == nullptr) {
        *result = SCIP_DIDNOTRUN;
        return SCIP_OKAY;
    }

    SCIP_RESULT separated = SCIP_DIDNOTFIND;
    SCIP_CALL(separate_capacity_data(scip, nullptr, *handler_data->data, &separated));
    if (separated == SCIP_CONSADDED) {
        *result = SCIP_CONSADDED;
        return SCIP_OKAY;
    }

    *result = has_capacity_violation(scip, nullptr, *handler_data->data) ? SCIP_INFEASIBLE : SCIP_FEASIBLE;
    return SCIP_OKAY;
}

SCIP_DECL_CONSENFOPS(capacity_cons_enfops) {
    ignore_unused(conss, nconss, nusefulconss, solinfeasible);
    if (objinfeasible) {
        *result = SCIP_DIDNOTRUN;
        return SCIP_OKAY;
    }

    auto* handler_data = SCIPconshdlrGetData(conshdlr);
    if (handler_data == nullptr || handler_data->data == nullptr) {
        *result = SCIP_DIDNOTRUN;
        return SCIP_OKAY;
    }

    SCIP_RESULT separated = SCIP_DIDNOTFIND;
    SCIP_CALL(separate_capacity_data(scip, nullptr, *handler_data->data, &separated));
    if (separated == SCIP_CONSADDED) {
        *result = SCIP_CONSADDED;
        return SCIP_OKAY;
    }

    *result = has_capacity_violation(scip, nullptr, *handler_data->data) ? SCIP_INFEASIBLE : SCIP_FEASIBLE;
    return SCIP_OKAY;
}

SCIP_DECL_CONSCHECK(capacity_cons_check) {
    ignore_unused(conss, nconss, checkintegrality, checklprows, printreason, completely);
    auto* handler_data = SCIPconshdlrGetData(conshdlr);
    if (handler_data == nullptr || handler_data->data == nullptr) {
        *result = SCIP_FEASIBLE;
        return SCIP_OKAY;
    }

    *result = has_capacity_violation(scip, sol, *handler_data->data) ? SCIP_INFEASIBLE : SCIP_FEASIBLE;
    return SCIP_OKAY;
}

SCIP_DECL_CONSLOCK(capacity_cons_lock) {
    ignore_unused(conshdlr);
    const auto* cons_data = SCIPconsGetData(cons);
    if (cons_data == nullptr)
        return SCIP_OKAY;

    for (SCIP_VAR* var : cons_data->vars)
        SCIP_CALL(SCIPaddVarLocksType(scip, var, locktype, nlocksneg, nlockspos));

    return SCIP_OKAY;
}

SCIP_DECL_SEPAFREE(capacity_sepa_free) {
    ignore_unused(scip);
    delete SCIPsepaGetData(sepa);
    return SCIP_OKAY;
}

SCIP_DECL_SEPAEXECLP(capacity_sepa_execlp) {
    ignore_unused(sepa, allowlocal, depth);
    auto* sepa_data = SCIPsepaGetData(sepa);
    if (sepa_data == nullptr || sepa_data->data == nullptr) {
        *result = SCIP_DIDNOTRUN;
        return SCIP_OKAY;
    }
    return separate_capacity_data(scip, nullptr, *sepa_data->data, result);
}

void include_capacity_separator(SCIP* scip, CompactSolveData& data) {
    auto* sepa_data = new SCIP_SEPADATA{.data = &data};
    SCIP_SEPA* sepa = nullptr;
    check(SCIPincludeSepaBasic(
        scip,
        &sepa,
        "compact_capacity_sepa",
        "compact MAFFE capacity separator",
        1000000,
        1,
        1.0,
        FALSE,
        FALSE,
        capacity_sepa_execlp,
        nullptr,
        sepa_data
    ));
    check(SCIPsetSepaFree(scip, sepa, capacity_sepa_free));
}

void include_capacity_constraint_handler(SCIP* scip, CompactSolveData& data) {
    auto* handler_data = new SCIP_CONSHDLRDATA{.data = &data};
    SCIP_CONSHDLR* conshdlr = nullptr;
    check(SCIPincludeConshdlrBasic(
        scip,
        &conshdlr,
        "compact_capacity",
        "compact MAFFE lazy capacity constraints",
        1000000,
        1000000,
        1,
        TRUE,
        capacity_cons_enfolp,
        capacity_cons_enfops,
        capacity_cons_check,
        capacity_cons_lock,
        handler_data
    ));
    check(SCIPsetConshdlrFree(scip, conshdlr, capacity_cons_free));
    check(SCIPsetConshdlrDelete(scip, conshdlr, capacity_cons_delete));
    check(SCIPsetConshdlrTrans(scip, conshdlr, capacity_cons_trans));

    auto* consdata = new SCIP_CONSDATA{.vars = capacity_lock_vars(data)};
    SCIP_CONS* cons = nullptr;
    check(SCIPcreateCons(
        scip,
        &cons,
        "compact_capacity_lazy",
        conshdlr,
        consdata,
        FALSE,
        TRUE,
        TRUE,
        TRUE,
        FALSE,
        FALSE,
        FALSE,
        FALSE,
        FALSE,
        FALSE
    ));
    check(SCIPaddCons(scip, cons));
    check(SCIPreleaseCons(scip, &cons));
}

[[nodiscard]] std::vector<double> extract_solution_values(
    SCIP* scip,
    SCIP_SOL* solution,
    const std::vector<SCIP_VAR*>& vars
) {
    std::vector<double> values;
    values.reserve(vars.size());
    for (SCIP_VAR* var : vars)
        values.push_back(SCIPgetSolVal(scip, solution, var));
    return values;
}

#if MAFFE_HAVE_GUROBI

struct GurobiDeleter {
    void operator()(GRBmodel* model) const noexcept {
        if (model != nullptr)
            GRBfreemodel(model);
    }

    void operator()(GRBenv* env) const noexcept {
        if (env != nullptr)
            GRBfreeenv(env);
    }
};

using GurobiModelPtr = std::unique_ptr<GRBmodel, GurobiDeleter>;
using GurobiEnvPtr = std::unique_ptr<GRBenv, GurobiDeleter>;

[[nodiscard]] const char* gurobi_status_name(const int status) {
    switch (status) {
    case GRB_OPTIMAL:
        return "optimal";
    case GRB_INFEASIBLE:
        return "infeasible";
    case GRB_INF_OR_UNBD:
        return "inf-or-unbd";
    case GRB_UNBOUNDED:
        return "unbounded";
    case GRB_CUTOFF:
        return "cutoff";
    case GRB_ITERATION_LIMIT:
        return "iteration-limit";
    case GRB_NODE_LIMIT:
        return "node-limit";
    case GRB_TIME_LIMIT:
        return "time-limit";
    case GRB_SOLUTION_LIMIT:
        return "solution-limit";
    case GRB_INTERRUPTED:
        return "interrupted";
    case GRB_NUMERIC:
        return "numeric";
    case GRB_SUBOPTIMAL:
        return "suboptimal";
    default:
        return "unknown";
    }
}

[[nodiscard]] const char* gurobi_error_name(const int error) {
    switch (error) {
    case GRB_ERROR_NO_LICENSE:
        return "no-license";
    case GRB_ERROR_SIZE_LIMIT_EXCEEDED:
        return "size-limit-exceeded";
    case GRB_ERROR_CALLBACK:
        return "callback";
    case GRB_ERROR_OUT_OF_MEMORY:
        return "out-of-memory";
    case GRB_ERROR_INVALID_ARGUMENT:
        return "invalid-argument";
    default:
        return "unknown";
    }
}

void check_gurobi(GRBenv* env, const int error) {
    if (error == 0)
        return;
    const char* const message = env == nullptr ? nullptr : GRBgeterrormsg(env);
    throw std::runtime_error(std::format(
        "Gurobi call failed: code={} ({}) message={}",
        error,
        gurobi_error_name(error),
        message == nullptr ? "unknown" : message
    ));
}

void apply_gurobi_time_limit(
    GRBmodel* model,
    const std::optional<std::chrono::steady_clock::time_point>& deadline
) {
    const auto remaining = remaining_seconds(deadline);
    if (!remaining.has_value())
        return;
    check_gurobi(GRBgetenv(model), GRBsetdblparam(GRBgetenv(model), GRB_DBL_PAR_TIMELIMIT, std::max(0.0, *remaining)));
}

struct GurobiCallbackData {
    const CompactModel* model = nullptr;
    const std::vector<TreeData>* trees = nullptr;
    const CompactStates* states = nullptr;
    const CapacityIndex* capacity_index = nullptr;
    std::vector<std::vector<char>> cut_added;
    std::vector<std::vector<char>> lazy_added;
    std::set<std::vector<int>> cut_signatures;
    std::set<std::vector<int>> lazy_signatures;
    LogLevel log_level = LogLevel::QUIET;
    int cut_rows = 0;
    int lazy_rows = 0;
    int separation_calls = 0;
    int callback_error = 0;
    double separation_seconds = 0.0;
};

[[nodiscard]] bool add_gurobi_capacity_row(
    void* cbdata,
    GurobiCallbackData& data,
    const int tree_index,
    const int vertex,
    const bool lazy
) {
    auto& added_matrix = lazy ? data.lazy_added : data.cut_added;
    auto& signatures = lazy ? data.lazy_signatures : data.cut_signatures;
    auto& added = added_matrix[tree_index][vertex];
    if (added)
        return false;
    added = true;

    std::vector<int> cols = materialize_capacity_row(
        *data.model,
        *data.trees,
        *data.states,
        *data.capacity_index,
        tree_index,
        vertex
    );
    if (cols.size() <= 1)
        return false;
    if (!signatures.insert(cols).second)
        return false;

    std::vector<double> coefficients(cols.size(), 1.0);
    const int error = lazy
        ? GRBcblazy(cbdata, static_cast<int>(cols.size()), cols.data(), coefficients.data(), GRB_LESS_EQUAL, 1.0)
        : GRBcbcut(cbdata, static_cast<int>(cols.size()), cols.data(), coefficients.data(), GRB_LESS_EQUAL, 1.0);
    if (error != 0) {
        data.callback_error = error;
        return false;
    }
    if (lazy)
        ++data.lazy_rows;
    else
        ++data.cut_rows;
    return true;
}

void separate_gurobi_capacity(
    void* cbdata,
    GurobiCallbackData& data,
    const std::span<const double> values,
    const bool lazy,
    const int row_limit
) {
    const auto start = std::chrono::steady_clock::now();
    std::vector<CapacityViolation> violations;
    auto& added_matrix = lazy ? data.lazy_added : data.cut_added;
    for (int tree_index = 0; tree_index < static_cast<int>(data.trees->size()); ++tree_index) {
        const std::vector<double> usage = capacity_usage(
            *data.model,
            *data.trees,
            *data.states,
            *data.capacity_index,
            values,
            tree_index
        );
        const int leaves = (*data.trees)[tree_index].helper.leaves();
        for (int vertex = leaves; vertex < static_cast<int>(usage.size()); ++vertex) {
            if (added_matrix[tree_index][vertex])
                continue;
            if (usage[vertex] <= 1.0 + constants::compact_capacity_tol)
                continue;
            violations.push_back(CapacityViolation{
                .tree = tree_index,
                .vertex = vertex,
                .nonzeros = data.capacity_index->row_nonzeros[tree_index][vertex],
                .violation = usage[vertex] - 1.0,
            });
        }
    }
    std::ranges::sort(violations, [](const CapacityViolation& lhs, const CapacityViolation& rhs) {
        return lhs.violation > rhs.violation;
    });

    int added = 0;
    std::vector<char> selected(violations.size(), false);
    const int high_violation_limit = std::min(
        row_limit * 3 / 5,
        static_cast<int>(violations.size())
    );
    for (int i = 0; i < high_violation_limit; ++i) {
        if (add_gurobi_capacity_row(cbdata, data, violations[i].tree, violations[i].vertex, lazy))
            ++added;
        selected[i] = true;
    }

    std::vector<int> density_order;
    density_order.reserve(violations.size());
    for (int i = 0; i < static_cast<int>(violations.size()); ++i) {
        if (!selected[i])
            density_order.push_back(i);
    }
    std::ranges::sort(density_order, [&](const int lhs, const int rhs) {
        const double lhs_density = violations[lhs].violation / std::max(1, violations[lhs].nonzeros);
        const double rhs_density = violations[rhs].violation / std::max(1, violations[rhs].nonzeros);
        if (lhs_density != rhs_density)
            return lhs_density > rhs_density;
        return violations[lhs].nonzeros < violations[rhs].nonzeros;
    });
    for (const int index : density_order) {
        if (added >= row_limit)
            break;
        if (add_gurobi_capacity_row(cbdata, data, violations[index].tree, violations[index].vertex, lazy))
            ++added;
    }

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ++data.separation_calls;
    data.separation_seconds += seconds;
    if ((added > 0 || seconds > 1.0) && logging::enabled(data.log_level, LogLevel::VERBOSE)) {
        logging::line(
            "compact-gurobi: separated capacity ",
            lazy ? "lazy" : "cuts",
            "=", added,
            " violations=", violations.size(),
            " limit=", row_limit,
            " time=", seconds,
            "s"
        );
    }
}

int gurobi_capacity_callback(
    GRBmodel* model,
    void* cbdata,
    const int where,
    void* usrdata
) {
    ignore_unused(model);
    auto& data = *static_cast<GurobiCallbackData*>(usrdata);

    if (where == GRB_CB_MESSAGE) {
        if (logging::enabled(data.log_level)) {
            char* message = nullptr;
            if (GRBcbget(cbdata, where, GRB_CB_MSG_STRING, &message) == 0)
                scip_error_log::log_prefixed_lines("compact-gurobi: ", message);
        }
        return 0;
    }

    if (where == GRB_CB_MIPNODE) {
        std::vector<double> values(data.model->cols());
        int status = 0;
        if (GRBcbget(cbdata, where, GRB_CB_MIPNODE_STATUS, &status) != 0 || status != GRB_OPTIMAL)
            return 0;
        if (GRBcbget(cbdata, where, GRB_CB_MIPNODE_REL, values.data()) != 0)
            return 0;
        double node_count = 0.0;
        if (GRBcbget(cbdata, where, GRB_CB_MIPNODE_NODCNT, &node_count) != 0)
            node_count = 1.0;
        const int row_limit = node_count <= 0.5 ? constants::compact_capacity_root_separation_row_limit : constants::compact_capacity_separation_row_limit;
        separate_gurobi_capacity(cbdata, data, values, false, row_limit);
        if (data.callback_error != 0)
            return data.callback_error;
    } else if (where == GRB_CB_MIPSOL) {
        std::vector<double> values(data.model->cols());
        if (GRBcbget(cbdata, where, GRB_CB_MIPSOL_SOL, values.data()) != 0)
            return 0;
        separate_gurobi_capacity(cbdata, data, values, true, constants::compact_capacity_root_separation_row_limit);
        if (data.callback_error != 0)
            return data.callback_error;
    }
    return 0;
}

#endif

} // namespace

Result solve_with_compact_lp_impl(
    const AnnotatedInstance& instance,
    const LogLevel log_level,
    const char* build_log_prefix,
    const char* solve_log_prefix,
    const std::optional<double> time_limit_seconds,
    const bool allow_abort_with_incumbent,
    const int objective_offset
) {
    if (instance.trees.size() < 2)
        return {};
    if (instance.trees.front().leaves() == 1)
        return Result{.partition = {{0}}, .feasible = true};
    if (!std::ranges::all_of(instance.trees, clean_tree))
        throw std::invalid_argument("compact LP solver currently requires unannotated trees");
    scip_error_log::ScopedPrefix scip_error_prefix(std::string(solve_log_prefix) + ": ");

    const auto start = std::chrono::steady_clock::now();
    const auto deadline = deadline_after(time_limit_seconds);
    CompactPrepared prepared = prepare_compact_model(instance);
    const auto built = std::chrono::steady_clock::now();
    if (logging::enabled(log_level, LogLevel::VERBOSE))
        log_compact_size_analysis(build_log_prefix, prepared);

    auto scip = scip_utils::create();
    scip_utils::install_message_handler(scip.get(), logging::enabled(log_level), scip_message_warning);

    check(SCIPincludeDefaultPlugins(scip.get()));
    apply_compact_scip_settings(scip.get(), log_level);
    check(SCIPcreateProbBasic(scip.get(), "compact"));
    check(SCIPsetObjsense(scip.get(), SCIP_OBJSENSE_MINIMIZE));
    check(SCIPaddOrigObjoffset(scip.get(), prepared.model.objective_offset + static_cast<double>(objective_offset)));
    check(SCIPsetObjIntegral(scip.get()));
    if (const auto remaining = remaining_seconds(deadline); remaining.has_value() && *remaining <= 0.0)
        return allow_abort_with_incumbent ? detail::singleton_result(instance.trees.front().leaves()) : Result{};
    apply_scip_time_limit(scip.get(), deadline);

    ScipModel scip_model = pass_model_to_scip(scip.get(), prepared.model, prepared.trees, prepared.states);
    CompactSolveData solve_data{
        .model = &prepared.model,
        .trees = &prepared.trees,
        .capacity_index = &prepared.capacity_index,
        .states = &prepared.states,
        .vars = &scip_model.vars,
        .capacity_added = {},
        .capacity_row_signatures = {},
        .log_level = log_level,
        .log_prefix = solve_log_prefix,
    };
    solve_data.capacity_added.reserve(prepared.trees.size());
    for (const auto& tree : prepared.trees)
        solve_data.capacity_added.emplace_back(tree.vertices, false);

    CompactRootLpSeed root_lp_seed = build_compact_root_lp_seed(prepared, objective_offset, deadline);
    CapacitySeedStats seeded_capacity =
        add_root_lp_seed_capacity_constraints(scip.get(), solve_data, prepared.instance, root_lp_seed.root_lp);
    const auto capacity_seed_done = std::chrono::steady_clock::now();
    const double root_lp_seed_seconds = std::chrono::duration<double>(root_lp_seed.done - root_lp_seed.start).count();
    const double capacity_seed_seconds = std::chrono::duration<double>(capacity_seed_done - root_lp_seed.done).count();
    include_capacity_separator(scip.get(), solve_data);
    include_capacity_constraint_handler(scip.get(), solve_data);
    const bool root_lp_incumbent_installed = root_lp_seed.start_values.has_value() &&
        install_compact_solution(scip.get(), scip_model.vars, *root_lp_seed.start_values);
    const bool zero_incumbent = !root_lp_incumbent_installed &&
        install_zero_compact_solution(scip.get(), scip_model.vars);

    if (logging::enabled(log_level)) {
        logging::line(
            build_log_prefix, ": trees=", instance.trees.size(),
            " leaves=", prepared.state_index.leaves,
            " pair-states=", prepared.state_index.states(),
            " states=", prepared.states.states(),
            " vars=", prepared.model.cols(),
            " rect-fold-groups=", prepared.model.rectangle_groups,
            " rect-fold-saved-vars=", prepared.model.rectangle_folded_vars,
            " pruned-vars=", prepared.prune_stats.removed,
            " prune-rounds=", prepared.prune_stats.rounds,
            " flow-rows=", prepared.model.rows(),
            " root-left-rows=", scip_model.root_left_rows,
            " seeded-capacity-rows=", seeded_capacity.rows,
            " seeded-capacity-nnz=", seeded_capacity.nonzeros,
            " flow-nnz=", prepared.model.value.size(),
            " root-left-nnz=", scip_model.root_left_nonzeros,
            " capacity-rows=", prepared.capacity_index.active_rows,
            " capacity-nnz=", prepared.capacity_index.nonzeros,
            " rootlp-objective=", root_lp_seed.root_lp.objective,
            " rootlp-rounds=", root_lp_seed.root_lp.rounds,
            " rootlp-columns=", root_lp_seed.root_lp.warm_start.columns.size(),
            " rootlp-heuristic-components=", root_lp_seed.incumbent_components,
            " rootlp-heuristic-objective=", root_lp_incumbent_installed ? root_lp_seed.incumbent_objective : -1.0,
            " rootlp-heuristic-installed=", root_lp_incumbent_installed ? 1 : 0,
            " zero-incumbent=", zero_incumbent ? 1 : 0,
            " scip-probing-maxprerounds=", constants::compact_scip_probing_max_prerounds,
            " scip-maxcutsroot=", constants::compact_scip_max_cuts_root,
            " build=", std::chrono::duration<double>(built - start).count(),
            "s rootlp-time=", root_lp_seed_seconds,
            "s seed-time=", capacity_seed_seconds,
            "s"
        );
        if (root_lp_seed.incumbent.has_value() && !root_lp_incumbent_installed && !root_lp_seed.start_stats.failure.empty()) {
            logging::line(
                build_log_prefix,
                ": rootlp-heuristic-start-failure=",
                root_lp_seed.start_stats.failure,
                " block-size=",
                root_lp_seed.start_stats.block_size,
                " plans=",
                root_lp_seed.start_stats.plans
            );
        }
    }

    const auto solve_start = std::chrono::steady_clock::now();
    if (const auto remaining = remaining_seconds(deadline); remaining.has_value() && *remaining <= 0.0)
        return allow_abort_with_incumbent ? detail::singleton_result(instance.trees.front().leaves()) : Result{};
    apply_scip_time_limit(scip.get(), deadline);
    check(SCIPsolve(scip.get()));
    const auto solved = std::chrono::steady_clock::now();
    const SCIP_STATUS status = SCIPgetStatus(scip.get());
    const double scip_primal_bound = SCIPgetPrimalbound(scip.get());
    const double scip_dual_bound = SCIPgetDualbound(scip.get());
    if (logging::enabled(log_level)) {
        logging::line(
            solve_log_prefix, ": status=", status_name(status),
            " objective=", scip_primal_bound,
            " dual=", scip_dual_bound,
            " nodes=", SCIPgetNNodes(scip.get()),
            " lp-iters=", SCIPgetNLPIterations(scip.get()),
            " seeded-capacity-rows=", seeded_capacity.rows,
            " capacity-rows=", solve_data.separated_capacity_rows,
            " capacity-sepa-calls=", solve_data.capacity_separation_calls,
            " capacity-sepa-time=", solve_data.capacity_separation_seconds,
            "s",
            " time=", std::chrono::duration<double>(solved - solve_start).count(),
            "s"
        );
    }

    if (status == SCIP_STATUS_INFEASIBLE) {
        return {};
    }
    if (status != SCIP_STATUS_OPTIMAL && !allow_abort_with_incumbent)
        return {};

    SCIP_SOL* solution = SCIPgetBestSol(scip.get());
    if (solution == nullptr) {
        if (allow_abort_with_incumbent && root_lp_seed.incumbent.has_value()) {
            return std::move(*root_lp_seed.incumbent);
        }
        return {};
    }

    const std::vector<double> values = extract_solution_values(scip.get(), solution, scip_model.vars);
    Result result = reconstruct_valid_solution(
        prepared.instance,
        prepared.state_index,
        prepared.states,
        prepared.model,
        values,
        SCIPgetSolOrigObj(scip.get(), solution) - static_cast<double>(objective_offset)
    );
    if (!result.feasible || result.partition.empty())
        return {};
    if (root_lp_seed.incumbent.has_value() &&
        root_lp_seed.incumbent->partition.size() < result.partition.size()) {
        return std::move(*root_lp_seed.incumbent);
    }
    return result;
}

Result solve_with_compact_gurobi(
    const AnnotatedInstance& instance,
    const LogLevel log_level,
    const std::optional<double> time_limit_seconds,
    const bool allow_abort_with_incumbent,
    const int objective_offset
) {
#if !MAFFE_HAVE_GUROBI
    ignore_unused(instance, log_level, time_limit_seconds, allow_abort_with_incumbent, objective_offset);
    throw std::runtime_error("Gurobi compact backend requested, but this build was configured without Gurobi");
#else
    if (instance.trees.size() < 2)
        return {};
    if (instance.trees.front().leaves() == 1)
        return Result{.partition = {{0}}, .feasible = true};
    if (!std::ranges::all_of(instance.trees, clean_tree))
        throw std::invalid_argument("compact Gurobi solver currently requires unannotated trees");

    const auto start = std::chrono::steady_clock::now();
    const auto deadline = deadline_after(time_limit_seconds);
    CompactPrepared prepared = prepare_compact_model(instance);
    const auto built = std::chrono::steady_clock::now();
    if (logging::enabled(log_level, LogLevel::VERBOSE))
        log_compact_size_analysis("compact-gurobi", prepared);

    if (const auto remaining = remaining_seconds(deadline); remaining.has_value() && *remaining <= 0.0)
        return allow_abort_with_incumbent ? detail::singleton_result(instance.trees.front().leaves()) : Result{};

    CompactRootLpSeed root_lp_seed = build_compact_root_lp_seed(prepared, objective_offset, deadline);
    const bool root_lp_incumbent_installed = root_lp_seed.start_values.has_value();
    std::vector<CapacitySeedRow> seed_rows = select_root_lp_seed_capacity_rows(
        prepared.model,
        prepared.trees,
        prepared.states,
        prepared.capacity_index,
        prepared.instance,
        root_lp_seed.root_lp
    );
    const auto capacity_seed_done = std::chrono::steady_clock::now();

    GRBenv* raw_env = nullptr;
    check_gurobi(nullptr, GRBemptyenv(&raw_env));
    GurobiEnvPtr env(raw_env);
    const int gurobi_output = logging::enabled(log_level) ? 1 : 0;
    check_gurobi(env.get(), GRBsetintparam(env.get(), GRB_INT_PAR_OUTPUTFLAG, gurobi_output));
    check_gurobi(env.get(), GRBsetintparam(env.get(), GRB_INT_PAR_LOGTOCONSOLE, 0));
    check_gurobi(env.get(), GRBstartenv(env.get()));

    std::vector<double> obj = prepared.model.col_cost;
    std::vector<double> lower = prepared.model.col_lower;
    std::vector<double> upper = prepared.model.col_upper;
    std::vector<char> type(prepared.model.cols(), GRB_BINARY);
    for (int col = 0; col < prepared.model.cols(); ++col) {
        if (prepared.model.col_type[col] == CompactVarType::Integer)
            type[col] = GRB_INTEGER;
    }

    GRBmodel* raw_model = nullptr;
    check_gurobi(env.get(), GRBnewmodel(
        env.get(),
        &raw_model,
        "compact-gurobi",
        prepared.model.cols(),
        obj.data(),
        lower.data(),
        upper.data(),
        type.data(),
        nullptr
    ));
    GurobiModelPtr model(raw_model);
    GRBenv* const model_env = GRBgetenv(model.get());
    check_gurobi(model_env, GRBsetintparam(model_env, GRB_INT_PAR_OUTPUTFLAG, gurobi_output));
    check_gurobi(model_env, GRBsetintparam(model_env, GRB_INT_PAR_LOGTOCONSOLE, 0));
    check_gurobi(model_env, GRBsetintparam(model_env, GRB_INT_PAR_THREADS, 0));
    check_gurobi(model_env, GRBsetintparam(model_env, GRB_INT_PAR_LAZYCONSTRAINTS, 1));
    check_gurobi(model_env, GRBsetintparam(model_env, GRB_INT_PAR_PRECRUSH, 1));
    check_gurobi(env.get(), GRBsetdblattr(
        model.get(),
        GRB_DBL_ATTR_OBJCON,
        prepared.model.objective_offset + static_cast<double>(objective_offset)
    ));
    std::vector<double> start_values = root_lp_seed.start_values.value_or(std::vector<double>(prepared.model.cols(), 0.0));
    check_gurobi(env.get(), GRBsetdblattrarray(
        model.get(),
        GRB_DBL_ATTR_START,
        0,
        prepared.model.cols(),
        start_values.data()
    ));
    apply_gurobi_time_limit(model.get(), deadline);

    std::vector<std::vector<int>> row_indices(prepared.model.rows());
    std::vector<std::vector<double>> row_values(prepared.model.rows());
    for (int col = 0; col < prepared.model.cols(); ++col) {
        for (int p = prepared.model.start[col]; p < prepared.model.start[col + 1]; ++p) {
            row_indices[prepared.model.index[p]].push_back(col);
            row_values[prepared.model.index[p]].push_back(prepared.model.value[p]);
        }
    }

    int flow_rows = 0;
    for (int row = 0; row < prepared.model.rows(); ++row) {
        const double row_lower = prepared.model.row_lower[row];
        const double row_upper = prepared.model.row_upper[row];
        const bool has_lower = !(std::isinf(row_lower) && row_lower < 0.0);
        const bool has_upper = !(std::isinf(row_upper) && row_upper > 0.0);
        if (has_lower && has_upper && std::abs(row_lower - row_upper) <= 1e-12) {
            check_gurobi(env.get(), GRBaddconstr(
                model.get(),
                static_cast<int>(row_indices[row].size()),
                row_indices[row].data(),
                row_values[row].data(),
                GRB_EQUAL,
                row_lower,
                nullptr
            ));
            ++flow_rows;
        } else {
            if (has_lower) {
                check_gurobi(env.get(), GRBaddconstr(
                    model.get(),
                    static_cast<int>(row_indices[row].size()),
                    row_indices[row].data(),
                    row_values[row].data(),
                    GRB_GREATER_EQUAL,
                    row_lower,
                    nullptr
                ));
                ++flow_rows;
            }
            if (has_upper) {
                check_gurobi(env.get(), GRBaddconstr(
                    model.get(),
                    static_cast<int>(row_indices[row].size()),
                    row_indices[row].data(),
                    row_values[row].data(),
                    GRB_LESS_EQUAL,
                    row_upper,
                    nullptr
                ));
                ++flow_rows;
            }
        }
    }

    std::vector<double> ones;
    CapacitySeedStats root_left_stats;
    CapacityColumns root_left = build_root_left_rows(prepared.model, prepared.trees, prepared.states);
    for (int tree_index = 0; tree_index < static_cast<int>(root_left.size()); ++tree_index) {
        for (int root = prepared.trees[tree_index].helper.leaves();
             root < static_cast<int>(root_left[tree_index].size());
             ++root) {
            const auto& cols = root_left[tree_index][root];
            if (cols.size() <= 1)
                continue;
            ones.assign(cols.size(), 1.0);
            check_gurobi(env.get(), GRBaddconstr(
                model.get(),
                static_cast<int>(cols.size()),
                const_cast<int*>(cols.data()),
                ones.data(),
                GRB_LESS_EQUAL,
                1.0,
                nullptr
            ));
            ++root_left_stats.rows;
            root_left_stats.nonzeros += cols.size();
        }
    }

    std::vector<std::vector<char>> initial_capacity_added;
    initial_capacity_added.reserve(prepared.trees.size());
    for (const auto& tree : prepared.trees)
        initial_capacity_added.emplace_back(tree.vertices, false);
    std::set<std::vector<int>> initial_capacity_signatures;

    CapacitySeedStats seeded_capacity;
    for (CapacitySeedRow& row : seed_rows) {
        if (row.cols.size() <= 1)
            continue;
        if (!initial_capacity_signatures.insert(row.cols).second)
            continue;
        ones.assign(row.cols.size(), 1.0);
        check_gurobi(env.get(), GRBaddconstr(
            model.get(),
            static_cast<int>(row.cols.size()),
            row.cols.data(),
            ones.data(),
            GRB_LESS_EQUAL,
            1.0,
            nullptr
        ));
        initial_capacity_added[row.tree][row.vertex] = true;
        ++seeded_capacity.rows;
        seeded_capacity.nonzeros += row.cols.size();
    }
    const auto gurobi_built = std::chrono::steady_clock::now();

    GurobiCallbackData callback_data{
        .model = &prepared.model,
        .trees = &prepared.trees,
        .states = &prepared.states,
        .capacity_index = &prepared.capacity_index,
        .cut_added = initial_capacity_added,
        .lazy_added = std::move(initial_capacity_added),
        .cut_signatures = initial_capacity_signatures,
        .lazy_signatures = std::move(initial_capacity_signatures),
        .log_level = log_level,
    };
    check_gurobi(env.get(), GRBsetcallbackfunc(model.get(), gurobi_capacity_callback, &callback_data));

    if (logging::enabled(log_level)) {
        logging::line(
            "compact-gurobi: trees=", instance.trees.size(),
            " leaves=", prepared.state_index.leaves,
            " pair-states=", prepared.state_index.states(),
            " states=", prepared.states.states(),
            " vars=", prepared.model.cols(),
            " rect-fold-groups=", prepared.model.rectangle_groups,
            " rect-fold-saved-vars=", prepared.model.rectangle_folded_vars,
            " pruned-vars=", prepared.prune_stats.removed,
            " prune-rounds=", prepared.prune_stats.rounds,
            " flow-rows=", flow_rows,
            " root-left-rows=", root_left_stats.rows,
            " seeded-capacity-rows=", seeded_capacity.rows,
            " seeded-capacity-nnz=", seeded_capacity.nonzeros,
            " flow-nnz=", prepared.model.value.size(),
            " root-left-nnz=", root_left_stats.nonzeros,
            " capacity-rows=", prepared.capacity_index.active_rows,
            " capacity-nnz=", prepared.capacity_index.nonzeros,
            " rootlp-objective=", root_lp_seed.root_lp.objective,
            " rootlp-rounds=", root_lp_seed.root_lp.rounds,
            " rootlp-columns=", root_lp_seed.root_lp.warm_start.columns.size(),
            " rootlp-heuristic-components=", root_lp_seed.incumbent_components,
            " rootlp-heuristic-objective=", root_lp_incumbent_installed ? root_lp_seed.incumbent_objective : -1.0,
            " rootlp-heuristic-installed=", root_lp_incumbent_installed ? 1 : 0,
            " zero-start=1",
            " build=", std::chrono::duration<double>(built - start).count(),
            "s rootlp-time=", std::chrono::duration<double>(root_lp_seed.done - root_lp_seed.start).count(),
            "s seed-time=", std::chrono::duration<double>(capacity_seed_done - root_lp_seed.done).count(),
            "s gurobi-build=", std::chrono::duration<double>(gurobi_built - capacity_seed_done).count(),
            "s"
        );
        if (root_lp_seed.incumbent.has_value() && !root_lp_incumbent_installed && !root_lp_seed.start_stats.failure.empty()) {
            logging::line(
                "compact-gurobi: rootlp-heuristic-start-failure=",
                root_lp_seed.start_stats.failure,
                " block-size=",
                root_lp_seed.start_stats.block_size,
                " plans=",
                root_lp_seed.start_stats.plans
            );
        }
    }

    const auto solve_start = std::chrono::steady_clock::now();
    if (const auto remaining = remaining_seconds(deadline); remaining.has_value() && *remaining <= 0.0)
        return allow_abort_with_incumbent ? detail::singleton_result(instance.trees.front().leaves()) : Result{};
    apply_gurobi_time_limit(model.get(), deadline);
    check_gurobi(env.get(), GRBoptimize(model.get()));
    const auto solved = std::chrono::steady_clock::now();

    int status = 0;
    int sol_count = 0;
    check_gurobi(env.get(), GRBgetintattr(model.get(), GRB_INT_ATTR_STATUS, &status));
    check_gurobi(env.get(), GRBgetintattr(model.get(), GRB_INT_ATTR_SOLCOUNT, &sol_count));
    double objective = kInf;
    if (sol_count > 0)
        check_gurobi(env.get(), GRBgetdblattr(model.get(), GRB_DBL_ATTR_OBJVAL, &objective));
    double dual = kInf;
    double nodes = 0.0;
    double iterations = 0.0;
    ignore_unused(GRBgetdblattr(model.get(), GRB_DBL_ATTR_OBJBOUND, &dual));
    ignore_unused(GRBgetdblattr(model.get(), GRB_DBL_ATTR_NODECOUNT, &nodes));
    ignore_unused(GRBgetdblattr(model.get(), GRB_DBL_ATTR_ITERCOUNT, &iterations));
    if (logging::enabled(log_level)) {
        logging::line(
            "compact-gurobi: status=", gurobi_status_name(status),
            " objective=", objective,
            " dual=", dual,
            " nodes=", nodes,
            " lp-iters=", iterations,
            " seeded-capacity-rows=", seeded_capacity.rows,
            " capacity-cut-rows=", callback_data.cut_rows,
            " capacity-lazy-rows=", callback_data.lazy_rows,
            " capacity-sepa-calls=", callback_data.separation_calls,
            " capacity-sepa-time=", callback_data.separation_seconds,
            "s",
            " time=", std::chrono::duration<double>(solved - solve_start).count(),
            "s"
        );
    }

    if (status == GRB_INFEASIBLE || status == GRB_INF_OR_UNBD || status == GRB_CUTOFF) {
        return {};
    }
    if (status != GRB_OPTIMAL && !allow_abort_with_incumbent)
        return {};
    if (sol_count == 0) {
        if (allow_abort_with_incumbent && root_lp_seed.incumbent.has_value()) {
            return std::move(*root_lp_seed.incumbent);
        }
        return {};
    }

    std::vector<double> values(prepared.model.cols());
    check_gurobi(env.get(), GRBgetdblattrarray(
        model.get(),
        GRB_DBL_ATTR_X,
        0,
        prepared.model.cols(),
        values.data()
    ));
    Result result = reconstruct_valid_solution(
        prepared.instance,
        prepared.state_index,
        prepared.states,
        prepared.model,
        values,
        objective - static_cast<double>(objective_offset)
    );
    if (!result.feasible || result.partition.empty())
        return {};
    if (root_lp_seed.incumbent.has_value() &&
        root_lp_seed.incumbent->partition.size() < result.partition.size()) {
        return std::move(*root_lp_seed.incumbent);
    }
    return result;
#endif
}

Result solve_with_compact_root_lp_seeded(
    const AnnotatedInstance& instance,
    const LogLevel log_level,
    const std::optional<double> time_limit_seconds,
    const bool allow_abort_with_incumbent,
    const int objective_offset
) {
    return solve_with_compact_lp_impl(
        instance,
        log_level,
        "compact-scip",
        "compact-scip",
        time_limit_seconds,
        allow_abort_with_incumbent,
        objective_offset
    );
}

} // namespace maffe
