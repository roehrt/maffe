#include "branchandprice/scip/branch_and_price.hpp"

#include "branchandprice/master/root_master.hpp"
#include "branchandprice/master/set_packing_heuristic.hpp"
#include "branchandprice/master/set_packing_scip.hpp"
#include "branchandprice/pricer/lagrangian.hpp"
#include "util/hash.hpp"
#include "util/partition_ops.hpp"
#include "util/log.hpp"
#include "util/constants.hpp"
#include "util/tree_ops.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// SCIP typedefs these plugin payload tags globally. Give this translation unit
// distinct tag names so C++ LTO does not report false cross-plugin ODR clashes.
#define SCIP_ConshdlrData MaffeBranchPriceScipConshdlrData
#define SCIP_ConsData MaffeBranchPriceScipConsData
#define SCIP_PricerData MaffeBranchPriceScipPricerData
#define SCIP_EventhdlrData MaffeBranchPriceScipEventhdlrData
#define SCIP_BranchruleData MaffeBranchPriceScipBranchruleData

#include "scip/cons_linear.h"
#include "scip/cons_setppc.h"
#include "scip/pub_event.h"
#include "scip/pub_message.h"
#include "scip/pub_sepa.h"
#include "scip/pub_var.h"
#include "scip/scip.h"
#include "scip/scip_branch.h"
#include "scip/scip_cons.h"
#include "scip/scip_event.h"
#include "scip/scip_heur.h"
#include "scip/scip_lp.h"
#include "scip/scip_message.h"
#include "scip/scip_param.h"
#include "scip/scip_pricer.h"
#include "scip/scip_probing.h"
#include "scip/scip_prob.h"
#include "scip/scip_sol.h"
#include "scip/scip_var.h"
#include "scip/scipdefplugins.h"
#include "scip/pub_heur.h"
#include "util/scip_error_log.hpp"

#include <array>

#ifndef MAFFE_COMPETITION_HEURISTIC_BUILD
#define MAFFE_COMPETITION_HEURISTIC_BUILD 0
#endif

#ifndef MAFFE_COMPETITION_LOWERBOUND_BUILD
#define MAFFE_COMPETITION_LOWERBOUND_BUILD 0
#endif

#undef SCIP_ConshdlrData
#undef SCIP_ConsData
#undef SCIP_PricerData
#undef SCIP_EventhdlrData
#undef SCIP_BranchruleData

#include "util/scip_utils.hpp"

namespace maffe {

struct LeafSetHash {
    [[nodiscard]] std::size_t operator()(const std::vector<int>& leaves) const {
        std::size_t seed = leaves.size();
        for (const int leaf : leaves)
            hash_combine(seed, leaf);
        return seed;
    }
};

struct Column {
    SCIP_VAR* var = nullptr;
    RootMasterColumn master;
    bool active = true;
    int delete_filterpos = -1;
};

struct MasterRows {
    std::vector<std::vector<SCIP_CONS*>> vertices;
};

struct DirectionalPseudocost {
    double gain_sum = 0.0;
    double distance_sum = 0.0;
    int samples = 0;
};

struct EdgePseudocost {
    DirectionalPseudocost cut;
    DirectionalPseudocost force;
};

struct PendingBranchObservation {
    int tree = -1;
    int edge = -1;
    EdgeState direction = EdgeState::UNKNOWN;
    double parent_lowerbound = 0.0;
    double distance = 0.0;
};

struct DualStabilizationState {
    bool initialized = false;
    int ineffective_rounds = 0;
    std::vector<std::vector<double>> vertex_center;
    std::vector<std::vector<double>> edge_center;
    std::vector<std::vector<EdgeState>> edge_states;
};

struct SetPackingMemory {
    std::vector<double> column_history;
    std::vector<int> best_columns;
    SCIP_HEUR* solution_source = nullptr;
    int best_saving = 0;
    int root_submip_probe_count = 0;
    int node_submip_probe_count = 0;
    int last_root_submip_probe_columns = 0;
    int last_node_submip_probe_columns = 0;
};

struct BranchAndPriceData {
    AnnotatedInstance instance;
    int objective_offset = 0;
    double acceptable_factor = 1.0;
    int acceptable_offset = 0;
    RootMasterLayout layout;
    MasterRows rows;
    std::vector<std::vector<int>> subtree_leaf_count;
    std::unordered_map<std::vector<int>, int, LeafSetHash> column_index;
    std::unordered_map<SCIP_VAR*, int> column_var_index;
    std::vector<Column> columns;
    std::unique_ptr<Lagrangian> lagrangian;
    int branch_pseudocost_filter = -1;
    int acceptance_filter = -1;
    LogLevel log_level = LogLevel::NORMAL;
    bool acceptance_triggered = false;
    double acceptance_dualbound = std::numeric_limits<double>::quiet_NaN();
    double acceptance_primalbound = std::numeric_limits<double>::quiet_NaN();
    double redcost_pricing_seconds = 0.0;
    double farkas_pricing_seconds = 0.0;
    double pricing_setup_seconds = 0.0;
    double pricing_lagrangian_seconds = 0.0;
    double pricing_column_seconds = 0.0;
    int redcost_pricing_calls = 0;
    int farkas_pricing_calls = 0;
    SCIP_NODE* node_dual_stabilization_node = nullptr;
    DualStabilizationState root_dual_stabilization;
    SCIP_Longint node_dual_stabilization_number = -1;
    int node_dual_stabilization_depth = -1;
    DualStabilizationState node_dual_stabilization;
    SetPackingMemory set_packing_memory;
    std::uint64_t redcost_candidate_blocks = 0;
    std::uint64_t redcost_duplicate_blocks = 0;
    std::uint64_t redcost_early_duplicate_blocks = 0;
    std::vector<std::vector<int>> edge_depth;
    std::vector<std::vector<EdgePseudocost>> edge_pseudocost;
    std::unordered_map<SCIP_Longint, PendingBranchObservation> pending_branch_observations;
};

struct PricingContext {
    std::vector<std::vector<double>> vertex_duals;
    std::vector<std::vector<double>> edge_duals;
    std::vector<double> rc_flat_duals;
    std::vector<std::vector<EdgeState>> edge_states;
};

struct PricingRoundStats {
    int added = 0;
    int candidate_blocks = 0;
    int negative_blocks = 0;
    int duplicate_blocks = 0;
    int early_duplicate_blocks = 0;
    int duplicate_nonlp_blocks = 0;
    double best_reduced_cost = std::numeric_limits<double>::infinity();
};

SCIP_DECL_HEUREXEC(set_packing_source_heur_exec) {
    (void)scip;
    (void)heur;
    (void)heurtiming;
    (void)nodeinfeasible;
    *result = SCIP_DIDNOTRUN;
    return SCIP_OKAY;
}

void merge_pricing_stats(PricingRoundStats& into, const PricingRoundStats& from) {
    into.added += from.added;
    into.candidate_blocks += from.candidate_blocks;
    into.negative_blocks += from.negative_blocks;
    into.duplicate_blocks += from.duplicate_blocks;
    into.early_duplicate_blocks += from.early_duplicate_blocks;
    into.duplicate_nonlp_blocks += from.duplicate_nonlp_blocks;
    into.best_reduced_cost = std::min(into.best_reduced_cost, from.best_reduced_cost);
}

} // namespace maffe

struct MaffeBranchPriceScipConshdlrData {
    maffe::BranchAndPriceData* data = nullptr;
    SCIP_NODE* cache_node = nullptr;
    std::vector<std::vector<maffe::EdgeState>> cache_states;
};

struct MaffeBranchPriceScipConsData {
    int tree = -1;
    std::vector<int> edges;
    maffe::EdgeState state = maffe::EdgeState::UNKNOWN;
    int npropagatedcols = 0;
    bool propagated = false;
    SCIP_NODE* node = nullptr;
    SCIP_CONS* row = nullptr;
};

struct MaffeBranchPriceScipPricerData {
    maffe::BranchAndPriceData* data = nullptr;
};

struct MaffeBranchPriceScipEventhdlrData {
    maffe::BranchAndPriceData* data = nullptr;
};

struct MaffeBranchPriceScipBranchruleData {
    maffe::BranchAndPriceData* data = nullptr;
};

namespace maffe {
namespace {

using scip_error_log::ignore_unused;
using scip_error_log::status_name;

[[nodiscard]] double pseudocost_value(const DirectionalPseudocost& stats) {
    if (stats.distance_sum <= constants::scip_branch_tol)
        return 0.0;
    return stats.gain_sum / stats.distance_sum;
}

[[nodiscard]] bool pseudocost_reliable(const DirectionalPseudocost& stats) {
    return stats.samples >= constants::scip_pseudocost_reliable_samples;
}

SCIP_DECL_MESSAGEWARNING(scip_message_warning) {
    ignore_unused(messagehdlr, file);
    scip_error_log::log_prefixed_lines("scip: ", msg);
}

void check(const SCIP_RETCODE code) {
    if (code != SCIP_OKAY)
        throw std::runtime_error("SCIP call failed");
}

[[nodiscard]] double snap_pricing_dual(const double value) {
    return std::abs(value) <= constants::pricing_dual_snap_tol ? 0.0 : value;
}

void deactivate_column(BranchAndPriceData& data, const int index) {
    if (index < 0 || index >= static_cast<int>(data.columns.size()))
        return;
    Column& column = data.columns[index];
    if (!column.active)
        return;
    column.active = false;
    column.delete_filterpos = -1;
    data.column_index.erase(column.master.leaves);
    data.column_var_index.erase(column.var);
}

void apply_branch_observation(
    BranchAndPriceData& data,
    const PendingBranchObservation& observation,
    double node_lowerbound
);

[[nodiscard]] std::vector<std::vector<EdgeState>> current_edge_states(
    SCIP* scip,
    const BranchAndPriceData& data
);

[[nodiscard]] const DirectionalPseudocost& directional_pseudocost(
    const EdgePseudocost& pseudocost,
    const EdgeState direction
) {
    if (direction == EdgeState::CUT)
        return pseudocost.cut;
    if (direction == EdgeState::FORCED)
        return pseudocost.force;
    throw std::runtime_error("invalid edge pseudocost direction");
}

[[nodiscard]] DirectionalPseudocost& directional_pseudocost(
    EdgePseudocost& pseudocost,
    const EdgeState direction
) {
    if (direction == EdgeState::CUT)
        return pseudocost.cut;
    if (direction == EdgeState::FORCED)
        return pseudocost.force;
    throw std::runtime_error("invalid edge pseudocost direction");
}

struct ScopedSeconds {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double* target = nullptr;

    explicit ScopedSeconds(double& total)
        : target(&total) {}

    ~ScopedSeconds() {
        if (target == nullptr)
            return;
        *target += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};

using SignalHandler = void (*)(int);

[[nodiscard]] SCIP*& active_sigint_scip() {
    static SCIP* scip = nullptr;
    return scip;
}

[[nodiscard]] SignalHandler& previous_sigint_handler() {
    static SignalHandler handler = SIG_DFL;
    return handler;
}

extern "C" void scip_sigint_handler(int signum) {
    ignore_unused(signum);
    if (SCIP* const scip = active_sigint_scip(); scip != nullptr)
        ignore_unused(SCIPinterruptSolve(scip));
}

struct ScopedSigintHandler {
    SCIP* previous_scip = nullptr;
    bool active = false;
    bool installed_handler = false;

    explicit ScopedSigintHandler(SCIP* scip) {
#if MAFFE_COMPETITION_HEURISTIC_BUILD
        ignore_unused(scip);
#else
        previous_scip = active_sigint_scip();
        active_sigint_scip() = scip;
        if (previous_scip == nullptr) {
            const SignalHandler previous = std::signal(SIGINT, scip_sigint_handler);
            if (previous == SIG_ERR) {
                active_sigint_scip() = nullptr;
                throw std::runtime_error("failed to install SIGINT handler");
            }
            previous_sigint_handler() = previous;
            installed_handler = true;
        }
        active = true;
#endif
    }

    ScopedSigintHandler(const ScopedSigintHandler&) = delete;
    ScopedSigintHandler& operator=(const ScopedSigintHandler&) = delete;

    ~ScopedSigintHandler() {
        if (!active)
            return;
        active_sigint_scip() = previous_scip;
        if (!installed_handler)
            return;
        const SignalHandler previous = previous_sigint_handler();
        previous_sigint_handler() = SIG_DFL;
        ignore_unused(std::signal(SIGINT, previous));
    }
};

[[nodiscard]] bool interrupted_status(const SCIP_STATUS status) {
    switch (status) {
    case SCIP_STATUS_USERINTERRUPT:
    case SCIP_STATUS_NODELIMIT:
    case SCIP_STATUS_TOTALNODELIMIT:
    case SCIP_STATUS_STALLNODELIMIT:
    case SCIP_STATUS_TIMELIMIT:
    case SCIP_STATUS_MEMLIMIT:
    case SCIP_STATUS_PRIMALLIMIT:
    case SCIP_STATUS_DUALLIMIT:
    case SCIP_STATUS_GAPLIMIT:
    case SCIP_STATUS_SOLLIMIT:
    case SCIP_STATUS_BESTSOLLIMIT:
    case SCIP_STATUS_RESTARTLIMIT:
    case SCIP_STATUS_UNBOUNDED:
    case SCIP_STATUS_INFORUNBD:
    case SCIP_STATUS_TERMINATE:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool acceptance_rule_enabled(const BranchAndPriceData& data) {
    return data.acceptable_factor != 1.0 || data.acceptable_offset != 0;
}

[[nodiscard]] double acceptance_target(const BranchAndPriceData& data, const double dualbound) {
    const double integral_lowerbound = std::ceil(dualbound - constants::scip_bound_tol);
    return std::floor(data.acceptable_factor * integral_lowerbound + constants::scip_bound_tol) +
        static_cast<double>(data.acceptable_offset);
}

[[nodiscard]] bool acceptance_rule_satisfied(SCIP* scip, BranchAndPriceData& data) {
    if (!acceptance_rule_enabled(data))
        return false;
    SCIP_SOL* const solution = SCIPgetBestSol(scip);
    if (solution == nullptr)
        return false;

    const double primalbound = SCIPgetPrimalbound(scip);
    const double dualbound = SCIPgetDualbound(scip);
    if (!std::isfinite(primalbound) || SCIPisInfinity(scip, std::abs(primalbound)) ||
        !std::isfinite(dualbound) || SCIPisInfinity(scip, std::abs(dualbound))) {
        return false;
    }

    const double target = acceptance_target(data, dualbound);
    if (primalbound > target + constants::scip_bound_tol)
        return false;

    data.acceptance_triggered = true;
    data.acceptance_primalbound = primalbound;
    data.acceptance_dualbound = dualbound;
    return true;
}

void interrupt_if_acceptance_rule_satisfied(SCIP* scip, BranchAndPriceData& data) {
    if (data.acceptance_triggered)
        return;
    if (acceptance_rule_satisfied(scip, data))
        check(SCIPinterruptSolve(scip));
}

[[nodiscard]] std::optional<RootMasterColumn> try_build_compatible_column(
    const AnnotatedInstance& instance,
    const RootMasterLayout& layout,
    std::span<const int> leaves,
    const std::vector<std::vector<EdgeState>>& edge_states
) {
    try {
        RootMasterColumn column = build_root_master_column(instance, layout, leaves);
        for (int tree = 0; tree < static_cast<int>(column.used_edges.size()); ++tree) {
            for (const int edge : column.used_edges[tree]) {
                if (edge_states[tree][edge] == EdgeState::CUT)
                    return std::nullopt;
            }
        }
        return column;
    } catch (const DisconnectedColumn&) {
        return std::nullopt;
    }
}

void resize_set_packing_memory(BranchAndPriceData& data) {
    data.set_packing_memory.column_history.resize(data.columns.size(), 0.0);
}

[[nodiscard]] bool set_packing_warm_start_contains(
    const SetPackingMemory& memory,
    const int column_id
) {
    return std::ranges::binary_search(memory.best_columns, column_id);
}

[[nodiscard]] double set_packing_dual_cost(
    const RootMasterColumn& column,
    const std::span<const double> row_duals
) {
    if (row_duals.empty())
        return 0.0;
    double cost = 0.0;
    for (const int row : column.row_indices) {
        if (row < static_cast<int>(row_duals.size()))
            cost += std::max(0.0, row_duals[row]);
    }
    return cost;
}

void record_set_packing_memory(
    BranchAndPriceData& data,
    const RootSetPackingSolution& packing
) {
    resize_set_packing_memory(data);
    for (double& value : data.set_packing_memory.column_history)
        value *= constants::scip_root_set_packing_heuristic_history_decay;
    for (const int column_id : packing.columns) {
        if (column_id >= 0 && column_id < static_cast<int>(data.set_packing_memory.column_history.size())) {
            double& value = data.set_packing_memory.column_history[column_id];
            value = std::min(1.0, value + constants::scip_root_set_packing_heuristic_history_increment);
        }
    }
    if (packing.saving > data.set_packing_memory.best_saving) {
        data.set_packing_memory.best_saving = packing.saving;
        data.set_packing_memory.best_columns = packing.columns;
        std::ranges::sort(data.set_packing_memory.best_columns);
    }
}

void include_set_packing_solution_sources(SCIP* scip, SetPackingMemory& memory) {
    SCIP_HEUR* heur = nullptr;
    check(SCIPincludeHeurBasic(
        scip,
        &heur,
        "setpacking",
        "source tag for injected set-packing solutions",
        SCIP_HEURDISPCHAR_TRIVIAL,
        std::numeric_limits<int>::min() / 4,
        -1,
        0,
        -1,
        SCIP_HEURTIMING_BEFORENODE,
        FALSE,
        set_packing_source_heur_exec,
        nullptr
    ));
    SCIPheurMarkExact(heur);
    memory.solution_source = heur;
}

void run_set_packing_heuristic(
    SCIP* scip,
    BranchAndPriceData& data,
    const std::span<const double> row_duals = {}
) {
    resize_set_packing_memory(data);
    const auto edge_states = current_edge_states(scip, data);
    std::vector<std::array<int, 2>> forced_edges;
    for (int tree = 0; tree < static_cast<int>(edge_states.size()); ++tree) {
        for (int edge = 0; edge < static_cast<int>(edge_states[tree].size()); ++edge) {
            if (edge_states[tree][edge] == EdgeState::FORCED)
                forced_edges.push_back({tree, edge});
        }
    }

    std::vector<RootSetPackingColumnView> columns;
    columns.reserve(data.columns.size());
    std::vector<std::vector<int>> forced_rows_by_column;
    forced_rows_by_column.reserve(data.columns.size());
    for (int column_id = 0; column_id < static_cast<int>(data.columns.size()); ++column_id) {
        const Column& column = data.columns[column_id];
        if (!column.active || column.master.leaves.size() < 2)
            continue;
        if (SCIPvarGetUbLocal(column.var) < 0.5)
            continue;
        forced_rows_by_column.emplace_back();
        std::vector<int>& forced_rows = forced_rows_by_column.back();
        for (int row = 0; row < static_cast<int>(forced_edges.size()); ++row) {
            const auto [tree, edge] = forced_edges[static_cast<std::size_t>(row)];
            if (root_master_column_uses_tree_edge(column.master, tree, edge))
                forced_rows.push_back(row);
        }
        columns.push_back(RootSetPackingColumnView{
            .column_id = column_id,
            .leaves = column.master.leaves,
            .row_indices = column.master.row_indices,
            .forced_rows = forced_rows,
            .lp_value = SCIPvarIsInLP(column.var) ? SCIPvarGetLPSol(column.var) : 0.0,
            .dual_cost = set_packing_dual_cost(column.master, row_duals),
            .history_value = data.set_packing_memory.column_history[column_id],
            .warm_start = set_packing_warm_start_contains(data.set_packing_memory, column_id),
            .tie_seed = static_cast<std::uint64_t>(column_id),
        });
    }

    const auto submit_solution = [&](const RootSetPackingSolution& packing) {
        if (packing.saving <= 0)
            return;

        SCIP_SOL* sol = nullptr;
        check(SCIPcreateSol(scip, &sol, data.set_packing_memory.solution_source));
        try {
            int selected_count = 0;
            for (const int column_id : packing.columns) {
                ++selected_count;
                check(SCIPsetSolVal(scip, sol, data.columns[column_id].var, 1.0));
            }

            SCIP_Bool stored = FALSE;
            check(SCIPaddSolFree(scip, &sol, &stored));
            if (stored == TRUE) {
                record_set_packing_memory(data, packing);
                if (logging::enabled(data.log_level, LogLevel::VERBOSE)) {
                    SCIP_NODE* const node = SCIPgetCurrentNode(scip);
                    logging::line(
                        "setpack: call=", data.redcost_pricing_calls,
                        " depth=", node != nullptr ? SCIPnodeGetDepth(node) : -1,
                        " columns=", selected_count,
                        " saving=", packing.saving,
                        " objective=", data.objective_offset + data.layout.leaf_count - packing.saving
                    );
                }
            }
        } catch (...) {
            if (sol != nullptr)
                ignore_unused(SCIPfreeSol(scip, &sol));
            throw;
        }
    };

    RootSetPackingSolution packing = solve_root_set_packing_heuristic(
        data.layout.leaf_count,
        data.layout.row_count,
        static_cast<int>(forced_edges.size()),
        static_cast<int>(data.columns.size()),
        columns);
    submit_solution(packing);

#if MAFFE_COMPETITION_HEURISTIC_BUILD == 0 && MAFFE_COMPETITION_LOWERBOUND_BUILD == 0
    SCIP_NODE* const node = SCIPgetCurrentNode(scip);
    const int depth = node != nullptr ? SCIPnodeGetDepth(node) : -1;
    int* probe_count = nullptr;
    int* last_probe_columns = nullptr;
    int max_probes = 0;
    int min_columns = 0;
    int min_new_columns = 0;
    double time_limit_seconds = 0.0;
    if (depth == 0) {
        probe_count = &data.set_packing_memory.root_submip_probe_count;
        last_probe_columns = &data.set_packing_memory.last_root_submip_probe_columns;
        max_probes = constants::scip_set_packing_submip_max_root_calls;
        min_columns = constants::scip_set_packing_submip_root_min_columns;
        min_new_columns = constants::scip_set_packing_submip_root_min_new_columns;
        time_limit_seconds = constants::scip_set_packing_submip_root_seconds;
    } else if (depth > 0) {
        probe_count = &data.set_packing_memory.node_submip_probe_count;
        last_probe_columns = &data.set_packing_memory.last_node_submip_probe_columns;
        max_probes = constants::scip_set_packing_submip_max_node_calls;
        min_columns = constants::scip_set_packing_submip_node_min_columns;
        min_new_columns = constants::scip_set_packing_submip_node_min_new_columns;
        time_limit_seconds = constants::scip_set_packing_submip_node_seconds;
    }

    if (probe_count != nullptr &&
        *probe_count < max_probes &&
        data.redcost_pricing_calls >= constants::scip_set_packing_submip_min_pricing_calls &&
        static_cast<int>(columns.size()) >= min_columns &&
        static_cast<int>(columns.size()) >= *last_probe_columns + min_new_columns) {
        ++*probe_count;
        *last_probe_columns = static_cast<int>(columns.size());

        std::vector<RootSetPackingColumnView> compact_columns;
        compact_columns.reserve(columns.size());
        std::vector<int> compact_to_column;
        compact_to_column.reserve(columns.size());
        std::vector<int> column_to_compact(data.columns.size(), -1);
        for (const RootSetPackingColumnView& column : columns) {
            const int compact_id = static_cast<int>(compact_columns.size());
            column_to_compact[static_cast<std::size_t>(column.column_id)] = compact_id;
            compact_to_column.push_back(column.column_id);
            compact_columns.push_back(column);
            compact_columns.back().column_id = compact_id;
        }

        std::vector<int> incumbent_columns;
        incumbent_columns.reserve(data.set_packing_memory.best_columns.size());
        for (const int column_id : data.set_packing_memory.best_columns) {
            if (column_id >= 0 && column_id < static_cast<int>(column_to_compact.size())) {
                const int compact_id = column_to_compact[static_cast<std::size_t>(column_id)];
                if (compact_id >= 0)
                    incumbent_columns.push_back(compact_id);
            }
        }

        RootSetPackingSolution exact = solve_root_set_packing_highs(
            data.layout.leaf_count,
            data.layout.row_count,
            static_cast<int>(forced_edges.size()),
            static_cast<int>(compact_columns.size()),
            compact_columns,
            incumbent_columns,
            data.objective_offset,
            time_limit_seconds,
            data.log_level
        );
        for (int& compact_id : exact.columns)
            compact_id = compact_to_column[static_cast<std::size_t>(compact_id)];
        submit_solution(exact);
        if (logging::enabled(data.log_level, LogLevel::VERBOSE)) {
            logging::line(
                "setpack-submip: ",
                depth == 0 ? "root" : "node",
                "-probe=", *probe_count,
                " call=", data.redcost_pricing_calls,
                " depth=", depth,
                " candidates=", compact_columns.size(),
                " columns=", exact.columns.size(),
                " saving=", exact.saving,
                " objective=", data.objective_offset + data.layout.leaf_count - exact.saving
            );
        }
    }
#endif
}

void add_column(
    SCIP* scip,
    BranchAndPriceData& data,
    RootMasterColumn column,
    bool initial
);

[[nodiscard]] MasterRows create_master_rows(
    SCIP* scip,
    const AnnotatedInstance& instance
) {
    MasterRows rows{
        .vertices = std::vector<std::vector<SCIP_CONS*>>(instance.trees.size()),
    };

    auto add_cons = [&](SCIP_CONS* cons) {
        check(SCIPsetConsModifiable(scip, cons, TRUE));
        check(SCIPaddCons(scip, cons));
        return cons;
    };

    for (int i = 0; i < static_cast<int>(instance.trees.size()); ++i) {
        const auto& tree = instance.trees[i];
        rows.vertices[i].resize(tree.vertices(), nullptr);
        for (int u = tree.leaves(); u < tree.vertices(); ++u) {
            SCIP_CONS* cons = nullptr;
            const auto name = std::format("vertex_{}_{}", i, u);
            check(SCIPcreateConsBasicSetpack(
                scip,
                &cons,
                name.c_str(),
                0,
                nullptr
            ));
            cons = add_cons(cons);
            rows.vertices[i][u] = cons;
            check(SCIPcaptureCons(scip, cons));
            check(SCIPreleaseCons(scip, &cons));
        }
    }

    return rows;
}

[[nodiscard]] std::optional<std::vector<int>> compatible_seed_pair(
    const AnnotatedInstance& instance,
    const RootMasterLayout& layout
) {
    if (layout.leaf_count < 2)
        return std::nullopt;

    std::unordered_map<std::vector<int>, int, LeafSetHash> representative;
    std::vector<int> signature;
    signature.reserve(instance.trees.size());
    for (int leaf = 0; leaf < layout.leaf_count; ++leaf) {
        signature.clear();
        bool valid = true;
        for (int tree = 0; tree < static_cast<int>(instance.trees.size()); ++tree) {
            const int component = layout.tree_index[tree].component[leaf];
            if (component < 0) {
                valid = false;
                break;
            }
            signature.push_back(component);
        }
        if (!valid)
            continue;
        if (const auto it = representative.find(signature); it != representative.end())
            return std::vector{it->second, leaf};
        representative.emplace(signature, leaf);
    }
    return std::nullopt;
}

void transform_rows(
    SCIP* scip,
    MasterRows& rows
) {
    for (auto& tree : rows.vertices) {
        for (auto& cons : tree) {
            if (cons == nullptr)
                continue;
            SCIP_CONS* original = cons;
            check(SCIPreleaseCons(scip, &cons));
            check(SCIPgetTransformedCons(scip, original, &cons));
            check(SCIPcaptureCons(scip, cons));
        }
    }
}

void release_rows(
    SCIP* scip,
    MasterRows& rows
) {
    for (auto& tree : rows.vertices) {
        for (auto& cons : tree) {
            if (cons != nullptr)
                check(SCIPreleaseCons(scip, &cons));
        }
    }
}

SCIP_RETCODE create_branch_state_consdata(
    SCIP* scip,
    SCIP_CONSDATA** consdata,
    const int tree,
    std::span<const int> edges,
    const EdgeState state,
    SCIP_NODE* node,
    SCIP_CONS* row
) {
    if (state != EdgeState::CUT && state != EdgeState::FORCED)
        throw std::invalid_argument("invalid branch edge state");
    check(SCIPallocBlockMemory(scip, consdata));
    (*consdata)->tree = tree;
    std::construct_at(&(*consdata)->edges, edges.begin(), edges.end());
    (*consdata)->state = state;
    (*consdata)->npropagatedcols = 0;
    (*consdata)->propagated = false;
    (*consdata)->node = node;
    (*consdata)->row = row;
    if (row != nullptr)
        check(SCIPcaptureCons(scip, row));
    return SCIP_OKAY;
}

void destroy_consdata(
    SCIP* scip,
    SCIP_CONSDATA** consdata
) {
    if ((*consdata)->row != nullptr)
        ignore_unused(SCIPreleaseCons(scip, &(*consdata)->row));
    std::destroy_at(&(*consdata)->edges);
    SCIPfreeBlockMemory(scip, consdata);
}

void invalidate_branch_state_cache(SCIP_CONSHDLRDATA* data) {
    data->cache_node = nullptr;
    data->cache_states.clear();
}

void update_branch_state_cache(
    SCIP* scip,
    SCIP_CONSHDLR* conshdlr,
    SCIP_CONSHDLRDATA* handler_data
) {
    if (handler_data->cache_node == SCIPgetCurrentNode(scip) &&
        !handler_data->cache_states.empty()) {
        return;
    }

    handler_data->cache_node = SCIPgetCurrentNode(scip);
    handler_data->cache_states.assign(handler_data->data->instance.trees.size(), {});
    for (int tree = 0; tree < static_cast<int>(handler_data->data->instance.trees.size()); ++tree) {
        handler_data->cache_states[tree].assign(
            handler_data->data->instance.trees[tree].vertices(),
            EdgeState::UNKNOWN
        );
    }

    SCIP_CONS* const* const conss = SCIPconshdlrGetConss(conshdlr);
    const int nactive = SCIPconshdlrGetNActiveConss(conshdlr);
    for (int c = 0; c < nactive; ++c) {
        const auto* cons_data = SCIPconsGetData(conss[c]);
        for (const int edge : cons_data->edges) {
            auto& cached = handler_data->cache_states[cons_data->tree][edge];
            if (cached != EdgeState::UNKNOWN && cached != cons_data->state)
                throw std::runtime_error("conflicting active branch states");
            cached = cons_data->state;
        }
    }
}

SCIP_DECL_CONSFREE(branch_states_free) {
    ignore_unused(scip);
    delete SCIPconshdlrGetData(conshdlr);
    return SCIP_OKAY;
}

SCIP_DECL_CONSDELETE(branch_states_delete) {
    ignore_unused(conshdlr, cons);
    destroy_consdata(scip, consdata);
    return SCIP_OKAY;
}

SCIP_DECL_CONSTRANS(branch_states_trans) {
    const auto* source = SCIPconsGetData(sourcecons);
    SCIP_CONSDATA* target = nullptr;
    check(create_branch_state_consdata(
        scip,
        &target,
        source->tree,
        source->edges,
        source->state,
        source->node,
        nullptr
    ));
    target->npropagatedcols = source->npropagatedcols;
    target->propagated = source->propagated;
    check(SCIPcreateCons(
        scip,
        targetcons,
        SCIPconsGetName(sourcecons),
        conshdlr,
        target,
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

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
SCIP_DECL_CONSPROP(branch_states_prop) {
    ignore_unused(nusefulconss, nmarkedconss, proptiming);

    auto* handler_data = SCIPconshdlrGetData(conshdlr);
    *result = SCIP_DIDNOTFIND;

    for (int c = 0; c < nconss; ++c) {
        auto* cons_data = SCIPconsGetData(conss[c]);
        if (cons_data->propagated)
            continue;

        int nfixed = 0;
        for (int idx = cons_data->npropagatedcols;
             cons_data->state == EdgeState::CUT &&
                 idx < static_cast<int>(handler_data->data->columns.size());
             ++idx) {
            auto& column = handler_data->data->columns[idx];
            if (!column.active)
                continue;
            if (SCIPvarGetUbLocal(column.var) < 0.5)
                continue;

            bool violates = false;
            for (const int edge : cons_data->edges) {
                if (root_master_column_uses_tree_edge(column.master, cons_data->tree, edge)) {
                    violates = true;
                    break;
                }
            }
            if (!violates)
                continue;

            SCIP_Bool cutoff = FALSE;
            SCIP_Bool fixed = FALSE;
            check(SCIPfixVar(scip, column.var, 0.0, &cutoff, &fixed));
            if (cutoff) {
                *result = SCIP_CUTOFF;
                return SCIP_OKAY;
            }
            if (fixed)
                ++nfixed;
        }

        cons_data->propagated = true;
        cons_data->npropagatedcols = static_cast<int>(handler_data->data->columns.size());
        if (nfixed > 0)
            *result = SCIP_REDUCEDDOM;
    }

    return SCIP_OKAY;
}

SCIP_DECL_CONSACTIVE(branch_states_active) {
    auto* cons_data = SCIPconsGetData(cons);
    auto* handler_data = SCIPconshdlrGetData(conshdlr);
    invalidate_branch_state_cache(handler_data);
    const int ncols = static_cast<int>(handler_data->data->columns.size());
    if (cons_data->npropagatedcols == ncols)
        return SCIP_OKAY;

    cons_data->propagated = false;
    check(SCIPrepropagateNode(scip, cons_data->node));
    return SCIP_OKAY;
}

SCIP_DECL_CONSDEACTIVE(branch_states_deactive) {
    ignore_unused(scip);
    auto* cons_data = SCIPconsGetData(cons);
    auto* handler_data = SCIPconshdlrGetData(conshdlr);
    invalidate_branch_state_cache(handler_data);
    cons_data->npropagatedcols = static_cast<int>(handler_data->data->columns.size());
    return SCIP_OKAY;
}

void include_branch_states(
    SCIP* scip,
    BranchAndPriceData* data
) {
    auto* handler_data = new SCIP_CONSHDLRDATA{
        .data = data,
        .cache_node = nullptr,
        .cache_states = {},
    };
    SCIP_CONSHDLR* conshdlr = nullptr;
    check(SCIPincludeConshdlrBasic(
        scip,
        &conshdlr,
        "maffe_branchstates",
        "local branch edge decisions",
        0,
        -9'999'998,
        1,
        TRUE,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        handler_data
    ));
    check(SCIPsetConshdlrFree(scip, conshdlr, branch_states_free));
    check(SCIPsetConshdlrDelete(scip, conshdlr, branch_states_delete));
    check(SCIPsetConshdlrTrans(scip, conshdlr, branch_states_trans));
    check(SCIPsetConshdlrActive(scip, conshdlr, branch_states_active));
    check(SCIPsetConshdlrDeactive(scip, conshdlr, branch_states_deactive));
    check(SCIPsetConshdlrProp(scip, conshdlr, branch_states_prop, 1, FALSE, SCIP_PROPTIMING_BEFORELP));
}

void add_branch_state_coefs(
    SCIP* scip,
    const Column& column
) {
    SCIP_CONSHDLR* conshdlr = SCIPfindConshdlr(scip, "maffe_branchstates");
    if (conshdlr == nullptr)
        return;

    SCIP_CONS* const* const conss = SCIPconshdlrGetConss(conshdlr);
    const int nconss = SCIPconshdlrGetNConss(conshdlr);
    for (int c = 0; c < nconss; ++c) {
        const auto* cons_data = SCIPconsGetData(conss[c]);
        if (cons_data->state != EdgeState::FORCED || cons_data->row == nullptr)
            continue;
        if (root_master_column_uses_tree_edge(column.master, cons_data->tree, cons_data->edges.front()))
            check(SCIPaddCoefLinear(scip, cons_data->row, column.var, 1.0));
    }
}

void add_branch_state_cons(
    SCIP* scip,
    SCIP_NODE* node,
    const std::string& name,
    const int tree,
    std::span<const int> edges,
    const EdgeState state
) {
    SCIP_CONSHDLR* conshdlr = SCIPfindConshdlr(scip, "maffe_branchstates");
    if (conshdlr == nullptr)
        throw std::runtime_error("missing branch-state handler");

    SCIP_CONS* row = nullptr;
    if (state == EdgeState::FORCED) {
        if (edges.size() != 1)
            throw std::invalid_argument("forced branch state expects exactly one edge");
        check(SCIPcreateConsLinear(
            scip,
            &row,
            (name + "_row").c_str(),
            0,
            nullptr,
            nullptr,
            1.0,
            SCIPinfinity(scip),
            TRUE,
            TRUE,
            TRUE,
            TRUE,
            TRUE,
            TRUE,
            TRUE,
            FALSE,
            FALSE,
            TRUE
        ));
        check(SCIPsetConsModifiable(scip, row, TRUE));
    }

    SCIP_CONSDATA* cons_data = nullptr;
    check(create_branch_state_consdata(scip, &cons_data, tree, edges, state, node, row));

    SCIP_CONS* cons = nullptr;
    check(SCIPcreateCons(
        scip,
        &cons,
        name.c_str(),
        conshdlr,
        cons_data,
        FALSE,
        FALSE,
        FALSE,
        FALSE,
        TRUE,
        TRUE,
        FALSE,
        FALSE,
        FALSE,
        TRUE
    ));

    auto* handler_data = SCIPconshdlrGetData(conshdlr);
    for (const auto& column : handler_data->data->columns) {
        if (!column.active)
            continue;
        if (row != nullptr && root_master_column_uses_tree_edge(column.master, tree, edges.front()))
            check(SCIPaddCoefLinear(scip, row, column.var, 1.0));
    }

    if (row != nullptr)
        check(SCIPaddConsNode(scip, node, row, nullptr));
    check(SCIPaddConsNode(scip, node, cons, nullptr));
    if (row != nullptr)
        check(SCIPreleaseCons(scip, &row));
    check(SCIPreleaseCons(scip, &cons));

    invalidate_branch_state_cache(handler_data);
}

[[nodiscard]] std::vector<double> active_branch_force_weights(
    SCIP* scip,
    const BranchAndPriceData& data,
    const int tree,
    const bool farkas
) {
    std::vector<double> weights(data.instance.trees[tree].vertices(), 0.0);
    SCIP_CONSHDLR* conshdlr = SCIPfindConshdlr(scip, "maffe_branchstates");
    if (conshdlr == nullptr)
        return weights;

    SCIP_CONS* const* const conss = SCIPconshdlrGetConss(conshdlr);
    const int nactive = SCIPconshdlrGetNActiveConss(conshdlr);
    for (int c = 0; c < nactive; ++c) {
        const auto* cons_data = SCIPconsGetData(conss[c]);
        if (cons_data->tree != tree || cons_data->state != EdgeState::FORCED || cons_data->row == nullptr)
            continue;
        const double dual = farkas
            ? static_cast<double>(SCIPgetDualfarkasLinear(scip, cons_data->row))
            : static_cast<double>(SCIPgetDualsolLinear(scip, cons_data->row));
        weights[cons_data->edges.front()] += std::max(dual, 0.0);
    }
    return weights;
}

[[nodiscard]] std::vector<std::vector<EdgeState>> current_edge_states(
    SCIP* scip,
    const BranchAndPriceData& data
) {
    std::vector<std::vector<EdgeState>> states(data.instance.trees.size());
    for (int tree = 0; tree < static_cast<int>(data.instance.trees.size()); ++tree)
        states[tree] = data.instance.trees[tree].edge_state;

    SCIP_CONSHDLR* conshdlr = SCIPfindConshdlr(scip, "maffe_branchstates");
    if (conshdlr == nullptr || SCIPgetStage(scip) < SCIP_STAGE_SOLVING)
        return states;

    auto* handler_data = SCIPconshdlrGetData(conshdlr);
    update_branch_state_cache(scip, conshdlr, handler_data);
    for (int tree = 0; tree < static_cast<int>(states.size()); ++tree) {
        for (int edge = 0; edge < static_cast<int>(states[tree].size()); ++edge) {
            const EdgeState active = handler_data->cache_states[tree][edge];
            if (active == EdgeState::UNKNOWN)
                continue;
            if (states[tree][edge] != EdgeState::UNKNOWN && states[tree][edge] != active)
                throw std::runtime_error("conflicting base and active edge states");
            states[tree][edge] = active;
        }
    }
    return states;
}

[[nodiscard]] std::vector<std::vector<double>> current_vertex_duals(
    SCIP* scip,
    const BranchAndPriceData& data,
    const bool farkas
) {
    std::vector<std::vector<double>> duals(data.instance.trees.size());
    for (int tree = 0; tree < static_cast<int>(data.instance.trees.size()); ++tree) {
        duals[tree].assign(data.instance.trees[tree].vertices(), 0.0);
        for (int u = data.instance.trees[tree].leaves(); u < data.instance.trees[tree].vertices(); ++u) {
            const double dual = farkas
                ? static_cast<double>(SCIPgetDualfarkasSetppc(scip, data.rows.vertices[tree][u]))
                : static_cast<double>(SCIPgetDualsolSetppc(scip, data.rows.vertices[tree][u]));
            duals[tree][u] = snap_pricing_dual(-dual);
        }
    }
    return duals;
}

void refresh_rc_flat_duals(
    const BranchAndPriceData& data,
    PricingContext& context
) {
    context.rc_flat_duals.assign(data.layout.row_count, 0.0);
    for (int tree = 0; tree < static_cast<int>(context.vertex_duals.size()); ++tree) {
        for (int u = data.instance.trees[tree].leaves(); u < data.instance.trees[tree].vertices(); ++u)
            context.rc_flat_duals[data.layout.row_of_vertex[tree][u]] = context.vertex_duals[tree][u];
    }
}

[[nodiscard]] PricingContext build_pricing_context(
    SCIP* scip,
    BranchAndPriceData& data,
    const bool farkas
) {
    const auto setup_start = std::chrono::steady_clock::now();
    PricingContext context;
    context.vertex_duals = current_vertex_duals(scip, data, farkas);
    context.edge_duals.resize(data.instance.trees.size());
    for (int tree = 0; tree < static_cast<int>(data.instance.trees.size()); ++tree) {
        context.edge_duals[tree] = active_branch_force_weights(scip, data, tree, farkas);
        for (double& dual : context.edge_duals[tree])
            dual = snap_pricing_dual(dual);
    }
    refresh_rc_flat_duals(data, context);
    context.edge_states = current_edge_states(scip, data);
    const auto setup_end = std::chrono::steady_clock::now();
    data.pricing_setup_seconds += std::chrono::duration<double>(setup_end - setup_start).count();
    return context;
}

[[nodiscard]] LagrangianResult solve_pricing_lagrangian(
    BranchAndPriceData& data,
    const PricingContext& context
) {
    const auto lagrangian_start = std::chrono::steady_clock::now();
    const auto result = data.lagrangian->solve(
        context.vertex_duals,
        context.edge_duals,
        context.edge_states
    );
    const auto lagrangian_end = std::chrono::steady_clock::now();
    data.pricing_lagrangian_seconds += std::chrono::duration<double>(lagrangian_end - lagrangian_start).count();
    return result;
}

[[nodiscard]] PricingRoundStats add_negative_pricing_columns(
    SCIP* scip,
    BranchAndPriceData& data,
    const LagrangianResult& lagrangian_result,
    const PricingContext& context
) {
    PricingRoundStats stats;
    const auto column_start = std::chrono::steady_clock::now();
    for (const auto& block : lagrangian_result.pricing_blocks()) {
        if (block.size() < 2)
            continue;
        ++stats.candidate_blocks;
        if (const auto existing = data.column_index.find(block); existing != data.column_index.end()) {
            ++stats.early_duplicate_blocks;
            ++stats.duplicate_blocks;
            if (!SCIPvarIsInLP(data.columns[existing->second].var))
                ++stats.duplicate_nonlp_blocks;
            continue;
        }
        auto column = try_build_compatible_column(data.instance, data.layout, block, context.edge_states);
        if (!column.has_value())
            continue;
        double reduced_cost = root_master_reduced_cost(*column, context.rc_flat_duals);
        for (int tree = 0; tree < static_cast<int>(column->used_edges.size()); ++tree) {
            for (const int edge : column->used_edges[tree])
                reduced_cost -= context.edge_duals[tree][edge];
        }
        stats.best_reduced_cost = std::min(stats.best_reduced_cost, reduced_cost);
        if (reduced_cost >= -constants::scip_reduced_cost_tol)
            continue;
        ++stats.negative_blocks;
        if (const auto it = data.column_index.find(column->leaves); it != data.column_index.end()) {
            ++stats.duplicate_blocks;
            if (!SCIPvarIsInLP(data.columns[it->second].var))
                ++stats.duplicate_nonlp_blocks;
            continue;
        }
        add_column(scip, data, std::move(*column), false);
        ++stats.added;
    }
    const auto column_end = std::chrono::steady_clock::now();
    data.pricing_column_seconds += std::chrono::duration<double>(column_end - column_start).count();
    return stats;
}

[[nodiscard]] bool same_dual_shape(
    const std::vector<std::vector<double>>& lhs,
    const std::vector<std::vector<double>>& rhs
) {
    if (lhs.size() != rhs.size())
        return false;
    for (int i = 0; i < static_cast<int>(lhs.size()); ++i) {
        if (lhs[i].size() != rhs[i].size())
            return false;
    }
    return true;
}

void reset_dual_stabilization(
    DualStabilizationState& state,
    const PricingContext& context
) {
    state.initialized = true;
    state.ineffective_rounds = 0;
    state.vertex_center = context.vertex_duals;
    state.edge_center = context.edge_duals;
    state.edge_states = context.edge_states;
}

void blend_dual_center(
    std::vector<std::vector<double>>& center,
    const std::vector<std::vector<double>>& current,
    const double current_weight
) {
    for (int i = 0; i < static_cast<int>(center.size()); ++i) {
        for (int j = 0; j < static_cast<int>(center[i].size()); ++j)
            center[i][j] = (1.0 - current_weight) * center[i][j] + current_weight * current[i][j];
    }
}

[[nodiscard]] PricingContext blended_pricing_context(
    const BranchAndPriceData& data,
    const PricingContext& context,
    const DualStabilizationState& state,
    const double current_weight
) {
    PricingContext stabilized = context;
    for (int tree = 0; tree < static_cast<int>(stabilized.vertex_duals.size()); ++tree) {
        for (int u = 0; u < static_cast<int>(stabilized.vertex_duals[tree].size()); ++u) {
            stabilized.vertex_duals[tree][u] = snap_pricing_dual(
                current_weight * context.vertex_duals[tree][u] +
                (1.0 - current_weight) * state.vertex_center[tree][u]
            );
        }
    }
    for (int tree = 0; tree < static_cast<int>(stabilized.edge_duals.size()); ++tree) {
        for (int edge = 0; edge < static_cast<int>(stabilized.edge_duals[tree].size()); ++edge) {
            stabilized.edge_duals[tree][edge] = snap_pricing_dual(
                current_weight * context.edge_duals[tree][edge] +
                (1.0 - current_weight) * state.edge_center[tree][edge]
            );
        }
    }
    refresh_rc_flat_duals(data, stabilized);
    return stabilized;
}

[[nodiscard]] bool dual_stabilization_shapes_match(
    const DualStabilizationState& state,
    const PricingContext& context
) {
    return same_dual_shape(state.vertex_center, context.vertex_duals) &&
        same_dual_shape(state.edge_center, context.edge_duals) &&
        state.edge_states == context.edge_states;
}

[[nodiscard]] DualStabilizationState* current_dual_stabilization_state(
    SCIP* scip,
    BranchAndPriceData& data
) {
    SCIP_NODE* const node = SCIPgetCurrentNode(scip);
    if (node == nullptr)
        return nullptr;

    if (SCIPnodeGetDepth(node) == 0)
        return &data.root_dual_stabilization;

    const SCIP_Longint node_number = SCIPnodeGetNumber(node);
    const int depth = SCIPnodeGetDepth(node);
    if (data.node_dual_stabilization_node != node ||
        data.node_dual_stabilization_number != node_number ||
        data.node_dual_stabilization_depth != depth) {
        data.node_dual_stabilization_node = node;
        data.node_dual_stabilization_number = node_number;
        data.node_dual_stabilization_depth = depth;
        data.node_dual_stabilization = {};
    }
    return &data.node_dual_stabilization;
}

[[nodiscard]] bool prepare_dual_stabilization(
    DualStabilizationState& state,
    const PricingContext& context,
    const double lp_lagrangian_gap
) {
    if (!state.initialized ||
        !dual_stabilization_shapes_match(state, context)) {
        reset_dual_stabilization(state, context);
        return false;
    }

    if (lp_lagrangian_gap <= constants::scip_dual_stabilization_min_gap)
        return false;

    return state.ineffective_rounds < constants::scip_dual_stabilization_reset_rounds;
}

void record_dual_stabilization_result(
    DualStabilizationState& state,
    const PricingContext& context,
    const PricingRoundStats& stabilized_stats
) {
    if (!state.initialized ||
        !dual_stabilization_shapes_match(state, context)) {
        reset_dual_stabilization(state, context);
        return;
    }

    if (stabilized_stats.added == 0)
        ++state.ineffective_rounds;
    else
        state.ineffective_rounds = 0;

    if (state.ineffective_rounds >= constants::scip_dual_stabilization_reset_rounds)
        reset_dual_stabilization(state, context);
    else {
        blend_dual_center(
            state.vertex_center,
            context.vertex_duals,
            constants::scip_dual_stabilization_update_weight
        );
        blend_dual_center(
            state.edge_center,
            context.edge_duals,
            constants::scip_dual_stabilization_update_weight
        );
    }
}

[[nodiscard]] PricingRoundStats add_stabilized_pricing_columns(
    SCIP* scip,
    BranchAndPriceData& data,
    const PricingContext& context,
    const double effective_global_lower_bound
) {
    if constexpr (constants::scip_dual_stabilization_current_weights.empty())
        return {};

    DualStabilizationState* const state = current_dual_stabilization_state(scip, data);
    if (state == nullptr)
        return {};

    const double lp_objective = SCIPretransformObj(scip, SCIPgetLPObjval(scip));
    const double lp_lagrangian_gap = lp_objective - effective_global_lower_bound;
    if (!state->initialized) {
        reset_dual_stabilization(*state, context);
        return {};
    }

    if (!prepare_dual_stabilization(
        *state,
        context,
        lp_lagrangian_gap
    ))
        return {};

    PricingRoundStats stats;
    for (const double base_weight : constants::scip_dual_stabilization_current_weights) {
        const PricingContext stabilized = blended_pricing_context(
            data,
            context,
            *state,
            base_weight
        );
        const auto stabilized_result = solve_pricing_lagrangian(data, stabilized);
        if (!std::isfinite(stabilized_result.lower_bound))
            break;
        const PricingRoundStats round_stats = add_negative_pricing_columns(
            scip,
            data,
            stabilized_result,
            context
        );
        merge_pricing_stats(stats, round_stats);
    }
    record_dual_stabilization_result(*state, context, stats);
    return stats;
}

[[nodiscard]] std::optional<PricingContext> subgradient_lookahead_context(
    const BranchAndPriceData& data,
    const PricingContext& context,
    const LagrangianResult& lagrangian_result,
    const double lp_lagrangian_gap
) {
    if (lp_lagrangian_gap <= constants::scip_root_pricing_lookahead_min_gap)
        return std::nullopt;

    std::vector<int> row_usage(data.layout.row_count, 0);
    for (const auto& block : lagrangian_result.leaf_partition) {
        if (block.size() < 2)
            continue;
        const auto column = try_build_compatible_column(data.instance, data.layout, block, context.edge_states);
        if (!column.has_value())
            continue;
        for (const int row : column->row_indices)
            ++row_usage[row];
    }

    double norm_squared = 0.0;
    for (const int usage : row_usage) {
        const double subgradient = static_cast<double>(usage - 1);
        norm_squared += subgradient * subgradient;
    }
    if (norm_squared <= 0.0)
        return std::nullopt;

    const double step = std::min(
        constants::scip_root_pricing_lookahead_max_step,
        constants::scip_root_pricing_lookahead_step_scale * lp_lagrangian_gap / norm_squared
    );
    if (step <= 0.0)
        return std::nullopt;

    PricingContext lookahead = context;
    for (int tree = 0; tree < static_cast<int>(lookahead.vertex_duals.size()); ++tree) {
        for (int u = data.instance.trees[tree].leaves(); u < data.instance.trees[tree].vertices(); ++u) {
            const int row = data.layout.row_of_vertex[tree][u];
            const double subgradient = static_cast<double>(row_usage[row] - 1);
            lookahead.vertex_duals[tree][u] = std::max(0.0, lookahead.vertex_duals[tree][u] + step * subgradient);
            lookahead.rc_flat_duals[row] = lookahead.vertex_duals[tree][u];
        }
    }
    return lookahead;
}

[[nodiscard]] std::optional<PricingContext> next_subgradient_lookahead_context(
    const BranchAndPriceData& data,
    const PricingContext& current_context,
    const PricingContext& true_context,
    const LagrangianResult& lookahead_result,
    const double lp_objective,
    const double effective_global_lower_bound
) {
    const double lookahead_lower_bound =
        lookahead_result.lower_bound + static_cast<double>(data.objective_offset);
    const double remaining_gap = lp_objective - std::max(effective_global_lower_bound, lookahead_lower_bound);
    if (remaining_gap <= constants::scip_root_pricing_lookahead_min_gap)
        return std::nullopt;

    auto next = subgradient_lookahead_context(
        data,
        current_context,
        lookahead_result,
        remaining_gap
    );
    if (next.has_value())
        next->edge_states = true_context.edge_states;
    return next;
}

[[nodiscard]] PricingRoundStats add_root_lookahead_pricing_columns(
    SCIP* scip,
    BranchAndPriceData& data,
    const PricingContext& context,
    const LagrangianResult& lagrangian_result,
    const double effective_global_lower_bound
) {
    if constexpr (constants::scip_root_pricing_lookahead_rounds <= 0)
        return {};

    SCIP_NODE* const node = SCIPgetCurrentNode(scip);
    if (node == nullptr || SCIPnodeGetDepth(node) != 0)
        return {};

    const double lp_objective = SCIPretransformObj(scip, SCIPgetLPObjval(scip));
    auto lookahead = subgradient_lookahead_context(
        data,
        context,
        lagrangian_result,
        lp_objective - effective_global_lower_bound
    );
    if (!lookahead.has_value())
        return {};

    PricingRoundStats stats;
    for (int round = 0; round < constants::scip_root_pricing_lookahead_rounds; ++round) {
        const auto lookahead_result = solve_pricing_lagrangian(data, *lookahead);
        if (!std::isfinite(lookahead_result.lower_bound))
            break;
        merge_pricing_stats(stats, add_negative_pricing_columns(scip, data, lookahead_result, context));
        if (round + 1 >= constants::scip_root_pricing_lookahead_rounds)
            break;
        lookahead = next_subgradient_lookahead_context(
            data,
            *lookahead,
            context,
            lookahead_result,
            lp_objective,
            effective_global_lower_bound
        );
        if (!lookahead.has_value())
            break;
    }
    return stats;
}

[[nodiscard]] bool install_initial_solution(
    SCIP* scip,
    const BranchAndPriceData& data,
    const Result& initial_solution
) {
    if (!initial_solution.feasible)
        return false;

    SCIP_SOL* sol = nullptr;
    check(SCIPcreateSol(scip, &sol, nullptr));
    try {
        for (const auto& block : initial_solution.partition) {
            if (block.size() < 2)
                continue;

            std::vector<int> leaves = block;
            std::ranges::sort(leaves);
            const auto it = data.column_index.find(leaves);
            if (it == data.column_index.end()) {
                check(SCIPfreeSol(scip, &sol));
                return false;
            }
            check(SCIPsetSolVal(scip, sol, data.columns[it->second].var, 1.0));
        }

        SCIP_Bool stored = FALSE;
        check(SCIPaddSolFree(scip, &sol, &stored));
        return stored;
    } catch (...) {
        if (sol != nullptr)
            ignore_unused(SCIPfreeSol(scip, &sol));
        throw;
    }
}

void add_column(
    SCIP* scip,
    BranchAndPriceData& data,
    RootMasterColumn master,
    const bool initial
) {
    if (master.leaves.size() < 2)
        return;
    if (data.column_index.contains(master.leaves))
        throw std::runtime_error("duplicate SCIP column insertion");

    const bool probing_only = !initial && SCIPinProbing(scip);
    SCIP_VAR* var = nullptr;
    const std::string name = std::format("col_{}", data.columns.size());
    check(SCIPcreateVar(
        scip,
        &var,
        name.c_str(),
        0.0,
        1.0,
        master.objective,
        SCIP_VARTYPE_BINARY,
        initial ? TRUE : FALSE,
        TRUE,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    ));
    if (probing_only)
        SCIPvarMarkDeletable(var);
    if (initial)
        check(SCIPaddVar(scip, var));
    else
        check(SCIPaddPricedVar(scip, var, 1.0));
    check(SCIPchgVarUbLazy(scip, var, 1.0));

    for (int tree = 0; tree < static_cast<int>(master.used_vertices.size()); ++tree) {
        for (const int u : master.used_vertices[tree])
            check(SCIPaddCoefSetppc(scip, data.rows.vertices[tree][u], var));
    }

    Column column{
        .var = var,
        .master = std::move(master),
        .active = true,
    };
    if (probing_only) {
        SCIP_EVENTHDLR* const eventhdlr = SCIPfindEventhdlr(scip, "maffe_column_delete");
        if (eventhdlr == nullptr)
            throw std::runtime_error("missing column-delete event handler");
        check(SCIPcatchVarEvent(
            scip,
            var,
            SCIP_EVENTTYPE_VARDELETED,
            eventhdlr,
            nullptr,
            &column.delete_filterpos
        ));
    }
    add_branch_state_coefs(scip, column);

    const int index = static_cast<int>(data.columns.size());
    data.column_index.emplace(column.master.leaves, index);
    data.column_var_index.emplace(column.var, index);
    data.columns.push_back(std::move(column));
    check(SCIPreleaseVar(scip, &var));
}

void add_column(
    SCIP* scip,
    BranchAndPriceData& data,
    std::span<const int> leaves,
    const bool initial
) {
    add_column(scip, data, build_root_master_column(data.instance, data.layout, leaves), initial);
}

SCIP_DECL_PRICERFREE(pricer_free) {
    ignore_unused(scip);
    delete SCIPpricerGetData(pricer);
    return SCIP_OKAY;
}

SCIP_DECL_PRICERINIT(pricer_init) {
    auto* data = SCIPpricerGetData(pricer)->data;
    transform_rows(scip, data->rows);
    data->column_var_index.clear();
    for (int i = 0; i < static_cast<int>(data->columns.size()); ++i) {
        auto& column = data->columns[i];
        check(SCIPgetTransformedVar(scip, column.var, &column.var));
        if (column.active)
            data->column_var_index.emplace(column.var, i);
    }
    return SCIP_OKAY;
}

SCIP_DECL_EVENTFREE(event_free) {
    ignore_unused(scip);
    delete SCIPeventhdlrGetData(eventhdlr);
    return SCIP_OKAY;
}

SCIP_DECL_EVENTEXEC(branch_pseudocost_event_exec) {
    ignore_unused(eventdata);
    auto* data = SCIPeventhdlrGetData(eventhdlr)->data;
    const SCIP_EVENTTYPE type = SCIPeventGetType(event);
    SCIP_NODE* node = nullptr;
    if ((type & SCIP_EVENTTYPE_LPSOLVED) != 0)
        node = SCIPgetCurrentNode(scip);
    else
        node = SCIPeventGetNode(event);
    if (node == nullptr)
        return SCIP_OKAY;

    const SCIP_Longint node_number = SCIPnodeGetNumber(node);
    const auto it = data->pending_branch_observations.find(node_number);
    if (it == data->pending_branch_observations.end())
        return SCIP_OKAY;

    if ((type & SCIP_EVENTTYPE_NODEDELETE) != 0) {
        data->pending_branch_observations.erase(it);
        return SCIP_OKAY;
    }
    if ((type & (SCIP_EVENTTYPE_LPSOLVED | SCIP_EVENTTYPE_NODESOLVED)) == 0)
        return SCIP_OKAY;

    const PendingBranchObservation observation = it->second;
    data->pending_branch_observations.erase(it);
    double node_lowerbound = SCIPnodeGetLowerbound(node);
    if ((type & SCIP_EVENTTYPE_LPSOLVED) != 0 && SCIPgetLPSolstat(scip) == SCIP_LPSOLSTAT_OPTIMAL)
        node_lowerbound = std::max(node_lowerbound, static_cast<double>(SCIPgetLPObjval(scip)));
    apply_branch_observation(*data, observation, node_lowerbound);
    return SCIP_OKAY;
}

SCIP_DECL_EVENTINITSOL(branch_pseudocost_event_initsol) {
    auto* data = SCIPeventhdlrGetData(eventhdlr)->data;
    check(SCIPcatchEvent(
        scip,
        SCIP_EVENTTYPE_LPSOLVED | SCIP_EVENTTYPE_NODESOLVED | SCIP_EVENTTYPE_NODEDELETE,
        eventhdlr,
        nullptr,
        &data->branch_pseudocost_filter
    ));
    return SCIP_OKAY;
}

SCIP_DECL_EVENTEXITSOL(branch_pseudocost_event_exitsol) {
    auto* data = SCIPeventhdlrGetData(eventhdlr)->data;
    if (data->branch_pseudocost_filter >= 0) {
        check(SCIPdropEvent(
            scip,
            SCIP_EVENTTYPE_LPSOLVED | SCIP_EVENTTYPE_NODESOLVED | SCIP_EVENTTYPE_NODEDELETE,
            eventhdlr,
            nullptr,
            data->branch_pseudocost_filter
        ));
    }
    data->branch_pseudocost_filter = -1;
    data->pending_branch_observations.clear();
    return SCIP_OKAY;
}

SCIP_DECL_EVENTEXEC(column_delete_event_exec) {
    ignore_unused(eventdata);
    auto* data = SCIPeventhdlrGetData(eventhdlr)->data;
    if ((SCIPeventGetType(event) & SCIP_EVENTTYPE_VARDELETED) == 0)
        return SCIP_OKAY;
    SCIP_VAR* const var = SCIPeventGetVar(event);
    if (var == nullptr)
        return SCIP_OKAY;
    const auto it = data->column_var_index.find(var);
    if (it == data->column_var_index.end())
        return SCIP_OKAY;
    Column& column = data->columns[it->second];
    if (column.delete_filterpos >= 0) {
        check(SCIPdropVarEvent(
            scip,
            var,
            SCIP_EVENTTYPE_VARDELETED,
            eventhdlr,
            nullptr,
            -1
        ));
    }
    deactivate_column(*data, it->second);
    return SCIP_OKAY;
}

SCIP_DECL_EVENTEXITSOL(column_delete_event_exitsol) {
    auto* data = SCIPeventhdlrGetData(eventhdlr)->data;
    for (auto& column : data->columns) {
        if (!column.active || column.delete_filterpos < 0)
            continue;
        check(SCIPdropVarEvent(
            scip,
            column.var,
            SCIP_EVENTTYPE_VARDELETED,
            eventhdlr,
            nullptr,
            column.delete_filterpos
        ));
        column.delete_filterpos = -1;
    }
    return SCIP_OKAY;
}

SCIP_DECL_EVENTEXEC(acceptance_event_exec) {
    ignore_unused(event, eventdata);
    auto* data = SCIPeventhdlrGetData(eventhdlr)->data;
    interrupt_if_acceptance_rule_satisfied(scip, *data);
    return SCIP_OKAY;
}

SCIP_DECL_EVENTINITSOL(acceptance_event_initsol) {
    auto* data = SCIPeventhdlrGetData(eventhdlr)->data;
    if (!acceptance_rule_enabled(*data))
        return SCIP_OKAY;
    check(SCIPcatchEvent(
        scip,
        SCIP_EVENTTYPE_LPSOLVED | SCIP_EVENTTYPE_NODESOLVED | SCIP_EVENTTYPE_BESTSOLFOUND,
        eventhdlr,
        nullptr,
        &data->acceptance_filter
    ));
    return SCIP_OKAY;
}

SCIP_DECL_EVENTEXITSOL(acceptance_event_exitsol) {
    auto* data = SCIPeventhdlrGetData(eventhdlr)->data;
    if (data->acceptance_filter >= 0) {
        check(SCIPdropEvent(
            scip,
            SCIP_EVENTTYPE_LPSOLVED | SCIP_EVENTTYPE_NODESOLVED | SCIP_EVENTTYPE_BESTSOLFOUND,
            eventhdlr,
            nullptr,
            data->acceptance_filter
        ));
    }
    data->acceptance_filter = -1;
    data->acceptance_triggered = false;
    data->acceptance_dualbound = std::numeric_limits<double>::quiet_NaN();
    data->acceptance_primalbound = std::numeric_limits<double>::quiet_NaN();
    return SCIP_OKAY;
}

void assert_redcost_pricing_progress(
    SCIP* scip,
    const PricingRoundStats& stats,
    const double global_lagrangian_lower_bound
) {
    if (stats.added != 0 || SCIPgetNPricevars(scip) > 0)
        return;

    const double lp_objective = SCIPretransformObj(scip, SCIPgetLPObjval(scip));
    if (std::abs(lp_objective - global_lagrangian_lower_bound) <= constants::scip_bound_tol)
        return;

    throw std::runtime_error(std::format(
        "SCIP redcost pricing stalled depth={} lp_obj={} lag_lb={} gap={} added={} pending={} candidates={} "
        "negative={} duplicates={} duplicate_nonlp={} best_rc={}",
        SCIPnodeGetDepth(SCIPgetCurrentNode(scip)),
        lp_objective,
        global_lagrangian_lower_bound,
        lp_objective - global_lagrangian_lower_bound,
        stats.added,
        SCIPgetNPricevars(scip),
        stats.candidate_blocks,
        stats.negative_blocks,
        stats.duplicate_blocks,
        stats.duplicate_nonlp_blocks,
        std::isfinite(stats.best_reduced_cost) ? stats.best_reduced_cost : 0.0
    ));
}

SCIP_DECL_PRICERREDCOST(pricer_redcost) {
    ignore_unused(stopearly);
    auto* data = SCIPpricerGetData(pricer)->data;
    ScopedSeconds timer(data->redcost_pricing_seconds);
    ++data->redcost_pricing_calls;

    const PricingContext context = build_pricing_context(scip, *data, false);
    const auto lagrangian_result = solve_pricing_lagrangian(*data, context);
    if (!std::isfinite(lagrangian_result.lower_bound)) {
        check(SCIPcutoffNode(scip, SCIPgetCurrentNode(scip)));
        *result = SCIP_SUCCESS;
        return SCIP_OKAY;
    }

    const double global_lagrangian_lower_bound =
        lagrangian_result.lower_bound + static_cast<double>(data->objective_offset);
    if (lowerbound != nullptr)
        *lowerbound = std::max(*lowerbound, SCIPtransformObj(scip, global_lagrangian_lower_bound));
    const double effective_global_lower_bound =
        lowerbound != nullptr
            ? SCIPretransformObj(scip, *lowerbound)
            : global_lagrangian_lower_bound;
    if (std::abs(SCIPretransformObj(scip, SCIPgetLPObjval(scip)) - effective_global_lower_bound) <= constants::scip_bound_tol) {
        run_set_packing_heuristic(scip, *data, context.rc_flat_duals);
        *result = SCIP_SUCCESS;
        return SCIP_OKAY;
    }

    PricingRoundStats stats = add_negative_pricing_columns(scip, *data, lagrangian_result, context);
    const PricingRoundStats stabilized_stats = add_stabilized_pricing_columns(
        scip,
        *data,
        context,
        effective_global_lower_bound
    );
    merge_pricing_stats(stats, stabilized_stats);
    const PricingRoundStats lookahead_stats = add_root_lookahead_pricing_columns(
        scip,
        *data,
        context,
        lagrangian_result,
        effective_global_lower_bound
    );
    merge_pricing_stats(stats, lookahead_stats);
    run_set_packing_heuristic(scip, *data, context.rc_flat_duals);
    data->redcost_candidate_blocks += static_cast<std::uint64_t>(stats.candidate_blocks);
    data->redcost_duplicate_blocks += static_cast<std::uint64_t>(stats.duplicate_blocks);
    data->redcost_early_duplicate_blocks += static_cast<std::uint64_t>(stats.early_duplicate_blocks);
    assert_redcost_pricing_progress(
        scip,
        stats,
        effective_global_lower_bound
    );

    *result = SCIP_SUCCESS;
    return SCIP_OKAY;
}

SCIP_DECL_PRICERFARKAS(pricer_farkas) {
    auto* data = SCIPpricerGetData(pricer)->data;
    ScopedSeconds timer(data->farkas_pricing_seconds);
    ++data->farkas_pricing_calls;

    const PricingContext context = build_pricing_context(scip, *data, true);
    const auto lagrangian_result = solve_pricing_lagrangian(*data, context);
    if (!std::isfinite(lagrangian_result.lower_bound)) {
        check(SCIPcutoffNode(scip, SCIPgetCurrentNode(scip)));
        *result = SCIP_SUCCESS;
        return SCIP_OKAY;
    }

    ignore_unused(add_negative_pricing_columns(scip, *data, lagrangian_result, context));

    *result = SCIP_SUCCESS;
    return SCIP_OKAY;
}

[[nodiscard]] std::vector<double> edge_used_mass(
    const BranchAndPriceData& data,
    const int tree
) {
    std::vector<double> mass(data.instance.trees[tree].vertices(), 0.0);
    for (const auto& column : data.columns) {
        if (!column.active)
            continue;
        if (SCIPvarGetUbLocal(column.var) < 0.5)
            continue;
        const double value = SCIPvarGetLPSol(column.var);
        if (value <= constants::scip_branch_tol)
            continue;
        for (const int edge : column.master.used_edges[tree])
            mass[edge] += value;
    }
    return mass;
}

[[nodiscard]] std::optional<double> estimated_branch_score(
    SCIP* scip,
    const EdgePseudocost& pseudocost,
    const double used
) {
    const bool have_cut = pseudocost.cut.distance_sum > constants::scip_branch_tol;
    const bool have_force = pseudocost.force.distance_sum > constants::scip_branch_tol;
    if (!have_cut && !have_force)
        return std::nullopt;

    return SCIPgetBranchScore(
        scip,
        nullptr,
        (have_cut ? pseudocost_value(pseudocost.cut) : 0.0) * used,
        (have_force ? pseudocost_value(pseudocost.force) : 0.0) * (1.0 - used)
    );
}

struct BranchCandidate {
    int tree = -1;
    int edge = -1;
    double used = 0.0;
    double fractionality = 0.0;
    int subtree = 0;
    int balance = 0;
    int depth = 0;
    double pseudocost_score = 0.0;
    bool score_available = false;
};

[[nodiscard]] bool better_balanced_candidate(
    const BranchCandidate& lhs,
    const BranchCandidate& rhs
) {
    return std::tuple(lhs.fractionality * static_cast<double>(lhs.balance), lhs.balance, lhs.fractionality, lhs.subtree, lhs.depth, -lhs.tree, lhs.edge) >
        std::tuple(rhs.fractionality * static_cast<double>(rhs.balance), rhs.balance, rhs.fractionality, rhs.subtree, rhs.depth, -rhs.tree, rhs.edge);
}

[[nodiscard]] bool better_scored_candidate(
    const BranchCandidate& lhs,
    const BranchCandidate& rhs
) {
    return std::tuple(
        lhs.pseudocost_score,
        lhs.fractionality,
        lhs.balance,
        lhs.subtree,
        lhs.depth,
        -lhs.tree,
        lhs.edge
    ) > std::tuple(
        rhs.pseudocost_score,
        rhs.fractionality,
        rhs.balance,
        rhs.subtree,
        rhs.depth,
        -rhs.tree,
        rhs.edge
    );
}

void remember_branch_observation(
    BranchAndPriceData& data,
    SCIP_NODE* const node,
    const int tree,
    const int edge,
    const EdgeState direction,
    const double parent_lowerbound,
    const double distance
) {
    if (node == nullptr || distance <= constants::scip_branch_tol)
        return;
    if (direction != EdgeState::CUT && direction != EdgeState::FORCED)
        throw std::runtime_error("invalid branch observation direction");
    data.pending_branch_observations.insert_or_assign(
        SCIPnodeGetNumber(node),
        PendingBranchObservation{
            .tree = tree,
            .edge = edge,
            .direction = direction,
            .parent_lowerbound = parent_lowerbound,
            .distance = distance,
        }
    );
}

void apply_branch_observation(
    BranchAndPriceData& data,
    const PendingBranchObservation& observation,
    const double node_lowerbound
) {
    if (observation.distance <= constants::scip_branch_tol || !std::isfinite(node_lowerbound))
        return;

    double gain = node_lowerbound - observation.parent_lowerbound;
    if (gain < 0.0)
        gain = 0.0;

    auto& stats = directional_pseudocost(
        data.edge_pseudocost[observation.tree][observation.edge],
        observation.direction
    );
    stats.gain_sum += gain;
    stats.distance_sum += observation.distance;
    ++stats.samples;
}

SCIP_DECL_BRANCHEXECLP(branch_exec_lp) {
    ignore_unused(allowaddcons);
    auto* data = SCIPbranchruleGetData(branchrule)->data;
    if (SCIPgetLPSolstat(scip) != SCIP_LPSOLSTAT_OPTIMAL) {
        *result = SCIP_DIDNOTRUN;
        return SCIP_OKAY;
    }

    const auto edge_states = current_edge_states(scip, *data);
    std::vector<BranchCandidate> candidates;

    const int leaf_count = data->instance.trees.front().leaves();
    for (int tree = 0; tree < static_cast<int>(data->instance.trees.size()); ++tree) {
        const auto used_mass = edge_used_mass(*data, tree);
        for (int edge = 0; edge < data->instance.trees[tree].vertices(); ++edge) {
            if (data->instance.trees[tree].parent[edge] < 0)
                continue;
            if (edge_states[tree][edge] != EdgeState::UNKNOWN)
                continue;

            const double used = used_mass[edge];
            if (used <= constants::scip_branch_tol || used >= 1.0 - constants::scip_branch_tol)
                continue;

            const int subtree = data->subtree_leaf_count[tree][edge];
            candidates.push_back(BranchCandidate{
                .tree = tree,
                .edge = edge,
                .used = used,
                .fractionality = std::min(used, 1.0 - used),
                .subtree = subtree,
                .balance = std::min(subtree, leaf_count - subtree),
                .depth = data->edge_depth[tree][edge],
            });
        }
    }

    if (candidates.empty())
        throw std::runtime_error("edge branching could not find a fractional branch edge");

    SCIP_NODE* const current_node = SCIPgetCurrentNode(scip);
    double parent_lowerbound = current_node != nullptr ? SCIPnodeGetLowerbound(current_node) : -SCIPinfinity(scip);
    parent_lowerbound = std::max(parent_lowerbound, static_cast<double>(SCIPgetLPObjval(scip)));
    for (auto& candidate : candidates) {
        const auto& pseudocost = data->edge_pseudocost[candidate.tree][candidate.edge];
        const bool reliable =
            pseudocost_reliable(directional_pseudocost(pseudocost, EdgeState::CUT)) &&
            pseudocost_reliable(directional_pseudocost(pseudocost, EdgeState::FORCED));
        if (reliable) {
            candidate.pseudocost_score =
                estimated_branch_score(scip, pseudocost, candidate.used).value_or(0.0);
            candidate.score_available = true;
        }
    }

    const BranchCandidate* best = nullptr;
    for (const auto& candidate : candidates) {
        if (!candidate.score_available)
            continue;
        if (best == nullptr || better_scored_candidate(candidate, *best))
            best = &candidate;
    }
    if (best == nullptr) {
        for (const auto& candidate : candidates) {
            if (best == nullptr || better_balanced_candidate(candidate, *best))
                best = &candidate;
        }
    }

    if (best == nullptr || best->edge < 0)
        throw std::runtime_error("edge branching could not find a fractional branch edge");

    SCIP_NODE* cut_child = nullptr;
    SCIP_NODE* force_child = nullptr;
    check(SCIPcreateChild(scip, &cut_child, 0.0, SCIPgetLocalTransEstimate(scip)));
    check(SCIPcreateChild(scip, &force_child, 0.0, SCIPgetLocalTransEstimate(scip)));

    add_branch_state_cons(
        scip,
        cut_child,
        std::format("edgecut_{}_{}", best->tree, best->edge),
        best->tree,
        std::span<const int>(&best->edge, 1),
        EdgeState::CUT
    );
    add_branch_state_cons(
        scip,
        force_child,
        std::format("edgeforce_{}_{}", best->tree, best->edge),
        best->tree,
        std::span<const int>(&best->edge, 1),
        EdgeState::FORCED
    );
    remember_branch_observation(*data, cut_child, best->tree, best->edge, EdgeState::CUT, parent_lowerbound, best->used);
    remember_branch_observation(*data, force_child, best->tree, best->edge, EdgeState::FORCED, parent_lowerbound, 1.0 - best->used);

    *result = SCIP_BRANCHED;
    return SCIP_OKAY;
}

SCIP_DECL_BRANCHFREE(branch_free) {
    ignore_unused(scip);
    delete SCIPbranchruleGetData(branchrule);
    return SCIP_OKAY;
}

[[nodiscard]] Result reconstruct_solution(
    SCIP* scip,
    const BranchAndPriceData& data,
    SCIP_SOL* solution
) {
    if (solution == nullptr)
        throw std::runtime_error("SCIP finished without an incumbent solution");

    const int leaf_count = data.instance.trees.front().leaves();
    std::vector<int> parent(leaf_count);
    std::vector<int> rank(leaf_count, 0);
    std::vector<int> cover(leaf_count, 0);
    for (int leaf = 0; leaf < leaf_count; ++leaf)
        parent[leaf] = leaf;

    const auto find = [&](this auto&& self, const int u) -> int {
        if (parent[u] == u)
            return u;
        parent[u] = self(parent[u]);
        return parent[u];
    };
    const auto join = [&](const int a, const int b) {
        int x = find(a);
        int y = find(b);
        if (x == y)
            return;
        if (rank[x] < rank[y])
            std::swap(x, y);
        parent[y] = x;
        if (rank[x] == rank[y])
            ++rank[x];
    };

    for (const auto& column : data.columns) {
        if (!column.active)
            continue;
        if (SCIPgetSolVal(scip, solution, column.var) <= 0.5)
            continue;
        const auto& leaves = column.master.leaves;
        for (std::size_t i = 1; i < leaves.size(); ++i)
            join(leaves[0], leaves[i]);
        for (const int leaf : leaves)
            ++cover[leaf];
    }
    if (std::ranges::any_of(cover, [](const int count) { return count > 1; }))
        throw std::runtime_error("SCIP incumbent uses overlapping columns");

    std::vector<std::vector<int>> partition(leaf_count);
    for (int leaf = 0; leaf < leaf_count; ++leaf)
        partition[find(leaf)].push_back(leaf);
    std::erase_if(partition, [](const auto& block) { return block.empty(); });

    return Result{
        .partition = std::move(partition),
        .feasible = true,
    };
}

} // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
Result solve_with_scip_branch_and_price(
    const AnnotatedInstance& instance,
    const LogLevel log_level,
    const std::span<const std::vector<int>> seed_columns,
    std::vector<std::vector<int>>* const generated_columns,
    const Result* const initial_solution,
    const bool allow_abort_with_incumbent,
    const int objective_offset,
    const std::optional<double> time_limit_seconds,
    const double acceptable_factor,
    const int acceptable_offset
) {
    scip_error_log::ScopedPrefix scip_error_prefix("scip: ");
    if (!std::isfinite(acceptable_factor) || acceptable_factor < 0.0)
        throw std::invalid_argument("SCIP acceptable_factor must be finite and nonnegative");
    const auto deadline = [&]() -> std::optional<std::chrono::steady_clock::time_point> {
        if (!time_limit_seconds.has_value())
            return std::nullopt;
        if (*time_limit_seconds < 0.0)
            throw std::invalid_argument("SCIP time limit must be nonnegative");
        return std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(*time_limit_seconds));
    }();
    RootMasterLayout layout = build_root_master_layout(instance);
    std::vector<std::vector<int>> initial_column_sets;
    std::unordered_set<std::vector<int>, LeafSetHash> initial_columns;
    const auto add_initial_column = [&](std::span<const int> block) {
        if (block.size() < 2)
            return;
        std::vector<int> leaves(block.begin(), block.end());
        std::ranges::sort(leaves);
        if (initial_columns.emplace(leaves).second)
            initial_column_sets.push_back(std::move(leaves));
    };
    for (const auto& leaves : seed_columns)
        add_initial_column(leaves);
    if (initial_solution != nullptr && initial_solution->feasible) {
        for (const auto& block : initial_solution->partition)
            add_initial_column(block);
    }
    if (initial_column_sets.empty()) {
        std::vector<std::vector<double>> edge_duals(instance.trees.size());
        std::vector<std::vector<EdgeState>> edge_states(instance.trees.size());
        for (int tree = 0; tree < static_cast<int>(instance.trees.size()); ++tree) {
            edge_duals[tree].assign(instance.trees[tree].vertices(), 0.0);
            edge_states[tree] = instance.trees[tree].edge_state;
        }
        auto lagrangian_result = Lagrangian(instance).solve(zero_vertex_duals(instance), edge_duals, edge_states);
        for (const auto& block : lagrangian_result.leaf_partition)
            add_initial_column(block);
        for (const auto& block : lagrangian_result.pricing_blocks())
            add_initial_column(block);
    }
    if (initial_column_sets.empty()) {
        if (const auto pair = compatible_seed_pair(instance, layout); pair.has_value())
            add_initial_column(*pair);
    }
    if (logging::enabled(log_level)) {
        logging::line(std::format(
            "rootlp: skipped seed-columns={}",
            initial_column_sets.size()
        ));
    }
    // With zero non-singleton columns, the master is already exact: every leaf must remain singleton.
    // Returning here also avoids HiGHS-in-SCIP failing on an empty LP with only setppc rows.
    if (initial_column_sets.empty()) {
        if (generated_columns != nullptr)
            generated_columns->clear();
        return detail::singleton_result(instance.trees.empty() ? 0 : instance.trees.front().leaves());
    }

    auto scip = scip_utils::create();
    scip_utils::install_message_handler(scip.get(), logging::enabled(log_level), scip_message_warning);

    check(SCIPincludeDefaultPlugins(scip.get()));
    check(SCIPsetBoolParam(scip.get(), "misc/catchctrlc", FALSE));
    if (logging::enabled(log_level, LogLevel::VERBOSE))
        check(SCIPsetIntParam(scip.get(), "display/freq", constants::scip_verbose_display_frequency));
    check(SCIPsetHeuristics(scip.get(), SCIP_PARAMSETTING_DEFAULT, TRUE));
    check(SCIPsetPresolving(scip.get(), SCIP_PARAMSETTING_OFF, TRUE));
    check(SCIPsetSeparating(scip.get(), SCIP_PARAMSETTING_OFF, TRUE));
    check(SCIPsetBoolParam(scip.get(), "pricing/delvars", TRUE));
    check(SCIPsetBoolParam(scip.get(), "pricing/delvarsroot", TRUE));
    check(SCIPsetIntParam(scip.get(), "constraints/setppc/sepafreq", constants::scip_setppc_separation_frequency));
    check(SCIPsetBoolParam(scip.get(), "constraints/setppc/cliquelifting", FALSE));
    check(SCIPsetIntParam(scip.get(), "separating/maxrounds", constants::scip_separation_max_rounds));
    check(SCIPsetIntParam(scip.get(), "separating/maxroundsroot", constants::scip_root_separation_max_rounds));
    check(SCIPcreateProbBasic(scip.get(), "maffe-branch-and-price"));
    check(SCIPsetObjsense(scip.get(), SCIP_OBJSENSE_MINIMIZE));
    check(SCIPaddOrigObjoffset(
        scip.get(),
        static_cast<SCIP_Real>(objective_offset + instance.trees.front().leaves())
    ));
    check(SCIPsetObjIntegral(scip.get()));
    if (deadline.has_value()) {
        const double remaining = std::chrono::duration<double>(*deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0.0)
            throw std::runtime_error("exact solve timed out before SCIP solve");
        check(SCIPsetRealParam(scip.get(), "limits/time", remaining));
    }

    auto data = std::make_unique<BranchAndPriceData>();
    data->instance = instance;
    data->objective_offset = objective_offset;
    data->acceptable_factor = acceptable_factor;
    data->acceptable_offset = acceptable_offset;
    data->layout = std::move(layout);
    data->rows = create_master_rows(scip.get(), instance);
    data->log_level = log_level;
    data->subtree_leaf_count.resize(instance.trees.size());
    data->edge_depth.resize(instance.trees.size());
    data->edge_pseudocost.resize(instance.trees.size());
    for (int tree = 0; tree < static_cast<int>(instance.trees.size()); ++tree) {
        const auto children = detail::tree_children(instance.trees[tree]);
        data->subtree_leaf_count[tree] = detail::subtree_leaf_counts(instance.trees[tree], children);
        data->edge_depth[tree].assign(instance.trees[tree].vertices(), 0);
        std::vector<int> stack;
        stack.reserve(instance.trees[tree].vertices());
        for (int u = 0; u < instance.trees[tree].vertices(); ++u) {
            if (instance.trees[tree].parent[u] < 0 || instance.trees[tree].edge_state[u] == EdgeState::CUT)
                stack.push_back(u);
        }
        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();
            const auto [left, right] = children[u];
            if (left >= 0) {
                data->edge_depth[tree][left] = data->edge_depth[tree][u] + 1;
                stack.push_back(left);
            }
            if (right >= 0) {
                data->edge_depth[tree][right] = data->edge_depth[tree][u] + 1;
                stack.push_back(right);
            }
        }
    }
    for (int tree = 0; tree < static_cast<int>(instance.trees.size()); ++tree)
        data->edge_pseudocost[tree].assign(instance.trees[tree].vertices(), {});
    data->lagrangian = std::make_unique<Lagrangian>(instance);
    include_set_packing_solution_sources(scip.get(), data->set_packing_memory);
    for (const auto& leaves : initial_column_sets)
        add_column(scip.get(), *data, leaves, true);
    if (initial_solution != nullptr)
        static_cast<void>(install_initial_solution(scip.get(), *data, *initial_solution));

    include_branch_states(scip.get(), data.get());
    for (int i = 0; i < SCIPgetNSepas(scip.get()); ++i)
        SCIPsepaSetFreq(SCIPgetSepas(scip.get())[i], -1);

    auto* pricer_data = new SCIP_PRICERDATA{.data = data.get()};
    SCIP_PRICER* pricer = nullptr;
    check(SCIPincludePricerBasic(
        scip.get(),
        &pricer,
        "maffe_pricer",
        "minimal MAFFE pricer",
        0,
        FALSE,
        pricer_redcost,
        pricer_farkas,
        pricer_data
    ));
    check(SCIPsetPricerFree(scip.get(), pricer, pricer_free));
    check(SCIPsetPricerInit(scip.get(), pricer, pricer_init));
    check(SCIPactivatePricer(scip.get(), pricer));

    auto* branch_event_data = new SCIP_EVENTHDLRDATA{.data = data.get()};
    SCIP_EVENTHDLR* branch_eventhdlr = nullptr;
    check(SCIPincludeEventhdlrBasic(
        scip.get(),
        &branch_eventhdlr,
        "maffe_branch_pscost",
        "update custom edge pseudocosts",
        branch_pseudocost_event_exec,
        branch_event_data
    ));
    check(SCIPsetEventhdlrFree(scip.get(), branch_eventhdlr, event_free));
    check(SCIPsetEventhdlrInitsol(scip.get(), branch_eventhdlr, branch_pseudocost_event_initsol));
    check(SCIPsetEventhdlrExitsol(scip.get(), branch_eventhdlr, branch_pseudocost_event_exitsol));

    auto* column_event_data = new SCIP_EVENTHDLRDATA{.data = data.get()};
    SCIP_EVENTHDLR* column_eventhdlr = nullptr;
    check(SCIPincludeEventhdlrBasic(
        scip.get(),
        &column_eventhdlr,
        "maffe_column_delete",
        "drop deleted generated columns from the MAFFE column index",
        column_delete_event_exec,
        column_event_data
    ));
    check(SCIPsetEventhdlrFree(scip.get(), column_eventhdlr, event_free));
    check(SCIPsetEventhdlrExitsol(scip.get(), column_eventhdlr, column_delete_event_exitsol));

    auto* acceptance_event_data = new SCIP_EVENTHDLRDATA{.data = data.get()};
    SCIP_EVENTHDLR* acceptance_eventhdlr = nullptr;
    check(SCIPincludeEventhdlrBasic(
        scip.get(),
        &acceptance_eventhdlr,
        "maffe_acceptance",
        "interrupt once the incumbent is within floor(a*k)+b",
        acceptance_event_exec,
        acceptance_event_data
    ));
    check(SCIPsetEventhdlrFree(scip.get(), acceptance_eventhdlr, event_free));
    check(SCIPsetEventhdlrInitsol(scip.get(), acceptance_eventhdlr, acceptance_event_initsol));
    check(SCIPsetEventhdlrExitsol(scip.get(), acceptance_eventhdlr, acceptance_event_exitsol));

    auto* branchrule_data = new SCIP_BRANCHRULEDATA{.data = data.get()};
    SCIP_BRANCHRULE* branchrule = nullptr;
    check(SCIPincludeBranchruleBasic(
        scip.get(),
        &branchrule,
        "maffe_tree_edges",
        "branches on a single tree edge",
        50'000,
        -1,
        1.0,
        branchrule_data
    ));
    check(SCIPsetBranchruleFree(scip.get(), branchrule, branch_free));
    check(SCIPsetBranchruleExecLp(scip.get(), branchrule, branch_exec_lp));
    for (int i = 0; i < SCIPgetNBranchrules(scip.get()); ++i) {
        SCIP_BRANCHRULE* other = SCIPgetBranchrules(scip.get())[i];
        if (std::string_view(SCIPbranchruleGetName(other)) == "maffe_tree_edges")
            continue;
        check(SCIPsetBranchruleExecLp(scip.get(), other, nullptr));
        check(SCIPsetBranchruleExecExt(scip.get(), other, nullptr));
        check(SCIPsetBranchruleExecPs(scip.get(), other, nullptr));
        check(SCIPsetBranchrulePriority(scip.get(), other, std::numeric_limits<int>::min() / 4));
    }

    try {
        const auto solve_start = std::chrono::steady_clock::now();
        const ScopedSigintHandler sigint_handler(scip.get());
        check(SCIPsolve(scip.get()));
        const auto solve_end = std::chrono::steady_clock::now();
        const double scip_wall_seconds = std::chrono::duration<double>(solve_end - solve_start).count();
        const double scip_solving_seconds = SCIPgetSolvingTime(scip.get());
        const double pricer_seconds = data->redcost_pricing_seconds + data->farkas_pricing_seconds;
        if (logging::enabled(log_level, LogLevel::VERBOSE)) {
            logging::line(
                "scip-stats: solving-s=", scip_solving_seconds,
                " wall-s=", scip_wall_seconds,
                " pricer-s=", pricer_seconds,
                " scip-other-s=", std::max(0.0, scip_solving_seconds - pricer_seconds),
                " redcost-s=", data->redcost_pricing_seconds,
                " redcost-calls=", data->redcost_pricing_calls,
                " redcost-candidates=", data->redcost_candidate_blocks,
                " redcost-duplicates=", data->redcost_duplicate_blocks,
                " redcost-early-duplicates=", data->redcost_early_duplicate_blocks,
                " farkas-s=", data->farkas_pricing_seconds,
                " farkas-calls=", data->farkas_pricing_calls,
                " pricing-setup-s=", data->pricing_setup_seconds,
                " pricing-lagrangian-s=", data->pricing_lagrangian_seconds,
                " pricing-columns-s=", data->pricing_column_seconds,
                " nodes=", SCIPgetNNodes(scip.get()),
                " lp-iters=", SCIPgetNLPIterations(scip.get())
            );
        }

        const SCIP_STATUS status = SCIPgetStatus(scip.get());
        if (status == SCIP_STATUS_INFEASIBLE) {
            if (generated_columns != nullptr)
                generated_columns->clear();
            release_rows(scip.get(), data->rows);
            scip.reset();
            return Result{
                .partition = {},
                .feasible = false,
            };
        }

        if (generated_columns != nullptr) {
            generated_columns->clear();
            generated_columns->reserve(data->columns.size());
            for (const auto& column : data->columns) {
                if (!column.active)
                    continue;
                generated_columns->push_back(column.master.leaves);
            }
        }

        SCIP_SOL* solution = SCIPgetBestSol(scip.get());
        if (solution == nullptr) {
            SCIP_SOL** sols = SCIPgetSols(scip.get());
            const int nsols = SCIPgetNSols(scip.get());
            if (sols != nullptr && nsols > 0)
                solution = sols[0];
        }
        const bool has_incumbent = solution != nullptr;
        const int nsols = SCIPgetNSols(scip.get());
        if (status != SCIP_STATUS_OPTIMAL) {
            if (!(allow_abort_with_incumbent && interrupted_status(status) && has_incumbent)) {
                throw std::runtime_error(std::format(
                    "SCIP finished without an optimal or usable interrupted solution: status={} "
                    "allow_abort={} nsols={} has_best={}",
                    status_name(status),
                    allow_abort_with_incumbent,
                    nsols,
                    SCIPgetBestSol(scip.get()) != nullptr
                ));
            }
            if (logging::enabled(log_level)) {
                if (data->acceptance_triggered) {
                    logging::line(
                        "accept: status=", status_name(status),
                        " dual=", data->acceptance_dualbound,
                        " primal=", data->acceptance_primalbound,
                        " target<=", acceptance_target(*data, data->acceptance_dualbound)
                    );
                } else {
                    logging::line(
                        "abort: status=", status_name(status),
                        " dual=", SCIPgetDualbound(scip.get()),
                        " primal=", SCIPgetPrimalbound(scip.get()),
                        " gap=", SCIPgetGap(scip.get())
                    );
                }
            }
        }

        const auto result = reconstruct_solution(scip.get(), *data, solution);
        release_rows(scip.get(), data->rows);
        scip.reset();
        return result;
    } catch (...) {
        release_rows(scip.get(), data->rows);
        scip.reset();
        throw;
    }
}
// NOLINTEND(bugprone-easily-swappable-parameters)

} // namespace maffe
