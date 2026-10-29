#include "config.hpp"
#include "maffe_unrooted.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "newick/newick.hpp"
#include "scip/cons_linear.h"
#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "util/constants.hpp"
#include "util/log.hpp"
#include "util/scip_error_log.hpp"
#include "util/scip_utils.hpp"

#if MAFFE_HAVE_GUROBI
#include "gurobi_c.h"
#endif
#include "util/union_find.hpp"

namespace maffe::experimental {
namespace {

using Graph = std::vector<std::vector<int>>;
using Clock = std::chrono::steady_clock;
struct TimedOut {};

struct Budget {
    std::optional<Clock::time_point> deadline;
    void check() const {
        if (deadline && Clock::now() >= *deadline)
            throw TimedOut{};
    }
};

struct Instance {
    std::vector<std::string> labels;
    std::vector<Graph> trees;
};

Instance parse(const std::vector<std::string> &input) {
    if (input.size() < 2)
        throw std::invalid_argument("unrooted mode requires at least two trees");
    Instance result;
    std::map<std::string, int> labels;
    for (const auto &text : input) {
        const auto tree = newick::parse(text);
        std::set<std::string> seen;
        for (int v = 0; v < static_cast<int>(tree.nodes.size()); ++v) {
            const auto &node = tree.nodes[v];
            if (node.children.empty()) {
                if (node.label.empty() || !seen.insert(node.label).second)
                    throw std::invalid_argument("empty or duplicate leaf label");
            } else if (node.children.size() != 2 && !(v == tree.root && node.children.size() == 3)) {
                throw std::invalid_argument("unrooted trees must be binary (root degree two or three)");
            }
        }
        if (result.trees.empty()) {
            result.labels.assign(seen.begin(), seen.end());
            for (int i = 0; i < static_cast<int>(result.labels.size()); ++i)
                labels.emplace(result.labels[i], i);
        } else if (std::vector<std::string>(seen.begin(), seen.end()) != result.labels) {
            throw std::invalid_argument("all trees must have the same leaf set");
        }
        const bool suppress = tree.nodes[tree.root].children.size() == 2;
        std::vector<int> ids(tree.nodes.size(), -1);
        int next = static_cast<int>(labels.size());
        for (int v = 0; v < static_cast<int>(tree.nodes.size()); ++v) {
            if (tree.nodes[v].children.empty())
                ids[v] = labels.at(tree.nodes[v].label);
            else if (!(suppress && v == tree.root))
                ids[v] = next++;
        }
        Graph graph(next);
        const auto connect = [&](int a, int b) {
            graph[a].push_back(b);
            graph[b].push_back(a);
        };
        for (int v = 0; v < static_cast<int>(tree.nodes.size()); ++v) {
            if (suppress && v == tree.root) {
                connect(ids[tree.nodes[v].children[0]], ids[tree.nodes[v].children[1]]);
            } else {
                for (int c : tree.nodes[v].children)
                    connect(ids[v], ids[c]);
            }
        }
        result.trees.push_back(std::move(graph));
    }
    return result;
}

// All-pairs first steps and LCAs under an arbitrary indexing root. Neither the
// median nor a connection's path depends on this indexing root.
struct TreeIndex {
    std::vector<std::vector<int>> next, lca;
    std::vector<int> depth;
    TreeIndex(const Graph &graph, const Budget &budget)
        : next(graph.size(), std::vector<int>(graph.size())), lca(next), depth(graph.size()) {
        std::vector<std::pair<int, int>> stack{{0, -1}};
        while (!stack.empty()) {
            const auto [v, p] = stack.back();
            stack.pop_back();
            for (int w : graph[v])
                if (w != p) {
                    depth[w] = depth[v] + 1;
                    stack.emplace_back(w, v);
                }
        }
        for (int s = 0; s < static_cast<int>(graph.size()); ++s) {
            budget.check();
            std::vector<std::array<int, 4>> todo{{s, -1, s, s}};
            while (!todo.empty()) {
                const auto [v, p, first, top] = todo.back();
                todo.pop_back();
                next[s][v] = first;
                lca[s][v] = top;
                for (int w : graph[v])
                    if (w != p)
                        todo.push_back({w, v, v == s ? w : first, depth[w] < depth[top] ? w : top});
            }
        }
    }
    int median(int a, int b, int c) const {
        int m = lca[a][b];
        for (int v : {lca[a][c], lca[b][c]})
            if (depth[v] > depth[m])
                m = v;
        return m;
    }
};

struct State {
    std::vector<int> vertex;
    std::array<std::vector<int>, 3> ports;
    auto operator<=>(const State &) const = default;
};

struct Model {
    int leaves;
    std::vector<TreeIndex> trees;
    std::vector<State> states;
    std::set<std::pair<int, int>> edges;
    int vertex(int s, int i) const { return s < leaves ? s : states[s - leaves].vertex[i]; }
    int port(int s, int t) const {
        const auto &state = states.at(s - leaves);
        for (int p = 0; p < 3; ++p) {
            bool matches = true;
            for (int i = 0; i < static_cast<int>(trees.size()); ++i)
                matches &= state.ports[p][i] == trees[i].next[vertex(s, i)][vertex(t, i)];
            if (matches)
                return p;
        }
        throw std::logic_error("unrooted connection has inconsistent ports");
    }
};

Model build(const Instance &instance, const Budget &budget) {
    const int n = static_cast<int>(instance.labels.size());
    Model model{.leaves = n, .trees = {}, .states = {}, .edges = {}};
    // Root canonically at leaf zero. Intern unordered pairs of child shapes,
    // with disjoint IDs for leaves and branches, to identify repeated unrooted
    // topologies independently of Newick rooting and internal vertex numbers.
    std::map<std::pair<int, int>, int> branches;
    std::set<int> seen;
    for (const auto &tree : instance.trees) {
        const auto shape = [&](this auto &&self, int v, int parent) -> int {
            budget.check();
            if (v < n)
                return v;
            std::array<int, 2> children;
            int j = 0;
            for (int w : tree[v])
                if (w != parent)
                    children[j++] = self(w, v);
            std::ranges::sort(children);
            return branches.emplace(std::pair{children[0], children[1]},
                                    n + static_cast<int>(branches.size())).first->second;
        };
        const int root = tree[0].empty() ? 0 : shape(tree[0].front(), 0);
        if (seen.insert(root).second)
            model.trees.emplace_back(tree, budget);
    }
    std::map<State, int> states;
    // Colexicographic rank: C(c,3) + C(b,2) + a for a < b < c.
    // Store exactly one state ID per triple, without ordered-map nodes.
    std::vector<std::size_t> pairs(n + 1), offsets(n + 1);
    for (int c = 1; c <= n; ++c) {
        pairs[c] = pairs[c - 1] + c - 1;
        offsets[c] = offsets[c - 1] + pairs[c - 1];
    }
    std::vector<int> triples(offsets[n]);
    const auto state_for = [&](int a, int b, int c) -> int& {
        return triples[offsets[c] + pairs[b] + a];
    };
    const auto connect = [&](int s, int t) { model.edges.emplace(std::min(s, t), std::max(s, t)); };
    for (int a = 0; a < n; ++a) {
        budget.check();
        for (int b = a + 1; b < n; ++b) {
            connect(a, b);
            for (int c = b + 1; c < n; ++c) {
                State state;
                for (const auto &tree : model.trees) {
                    const int m = tree.median(a, b, c);
                    state.vertex.push_back(m);
                    state.ports[0].push_back(tree.next[m][a]);
                    state.ports[1].push_back(tree.next[m][b]);
                    state.ports[2].push_back(tree.next[m][c]);
                }
                std::ranges::sort(state.ports);
                auto [it, inserted] = states.emplace(state, n + static_cast<int>(model.states.size()));
                if (inserted)
                    model.states.push_back(std::move(state));
                const int s = it->second;
                state_for(a, b, c) = s;
                for (int v : {a, b, c})
                    connect(s, v);
            }
        }
    }
    // Every suppressed internal edge has two witnessing leaves on each side.
    // For a shared split ab|cd, states abc and abd coincide, as do acd and
    // bcd. Conversely, equal states for two overlapping triples put their
    // differing leaves in the same port in every tree, proving that split.
    for (int a = 0; a < n; ++a) {
        for (int b = a + 1; b < n; ++b) {
            budget.check();
            for (int c = b + 1; c < n; ++c) {
                const int abc = state_for(a, b, c);
                for (int d = c + 1; d < n; ++d) {
                    const int abd = state_for(a, b, d);
                    const int acd = state_for(a, c, d);
                    const int bcd = state_for(b, c, d);
                    if (abc == abd)
                        connect(abc, acd);
                    else if (abc == acd || abc == bcd)
                        connect(abc, abd);
                }
            }
        }
    }
    return model;
}

void check(SCIP_RETCODE code) {
    if (code != SCIP_OKAY)
        throw std::runtime_error("experimental unrooted SCIP call failed");
}

struct Scip {
    using Var = SCIP_VAR *;
    using Row = SCIP_CONS *;
    SCIP *ptr = nullptr;
    static SCIP_DECL_MESSAGEWARNING(warning) {
        scip_error_log::ignore_unused(messagehdlr, file);
        scip_error_log::log_prefixed_lines("unrooted-scip: ", msg);
    }
    Scip(int leaves, const UnrootedOptions &options) {
        check(SCIPcreate(&ptr));
        try {
            scip_utils::install_message_handler(ptr, logging::enabled(options.log_level), warning);
            check(SCIPincludeDefaultPlugins(ptr));
            check(SCIPsetBoolParam(ptr, "misc/catchctrlc", FALSE));
            if (logging::enabled(options.log_level, LogLevel::VERBOSE))
                check(SCIPsetIntParam(ptr, "display/freq", constants::scip_verbose_display_frequency));
            check(SCIPcreateProbBasic(ptr, "experimental-unrooted-three-port"));
            check(SCIPaddOrigObjoffset(ptr, leaves));
        } catch (...) {
            (void)SCIPfree(&ptr);
            throw;
        }
    }
    Scip(const Scip &) = delete;
    Scip &operator=(const Scip &) = delete;
    ~Scip() {
        if (ptr)
            (void)SCIPfree(&ptr);
    }
    double infinity() const { return SCIPinfinity(ptr); }
    Var variable(const std::string &name, double cost, bool binary = true) {
        SCIP_VAR *v = nullptr;
        check(SCIPcreateVarBasic(ptr, &v, name.c_str(), 0, 1, cost, binary ? SCIP_VARTYPE_BINARY : SCIP_VARTYPE_CONTINUOUS));
        const auto status = SCIPaddVar(ptr, v);
        SCIP_VAR *result = v;
        check(SCIPreleaseVar(ptr, &v));
        check(status);
        return result;
    }
    Row row(const std::string &name, double lower, double upper) {
        SCIP_CONS *row = nullptr;
        check(SCIPcreateConsBasicLinear(ptr, &row, name.c_str(), 0, nullptr, nullptr, lower, upper));
        const auto status = SCIPaddCons(ptr, row);
        SCIP_CONS *result = row;
        check(SCIPreleaseCons(ptr, &row));
        check(status);
        return result;
    }
    void coefficient(Row row, Var var, double value) { check(SCIPaddCoefLinear(ptr, row, var, value)); }
    void optimize(const Budget &budget) {
        SCIP_SOL *seed = nullptr;
        check(SCIPcreateSol(ptr, &seed, nullptr)); // All singletons: x=y=0.
        SCIP_Bool stored = FALSE;
        check(SCIPaddSolFree(ptr, &seed, &stored));
        budget.check();
        if (budget.deadline)
            check(SCIPsetRealParam(ptr, "limits/time", std::max(0.0, std::chrono::duration<double>(*budget.deadline - Clock::now()).count())));
        check(SCIPsolve(ptr));
    }
    bool optimal() const { return SCIPgetStatus(ptr) == SCIP_STATUS_OPTIMAL; }
    bool has_solution() const { return SCIPgetBestSol(ptr) != nullptr; }
    double bound() const { return SCIPgetDualbound(ptr); }
    double objective() const { return SCIPgetSolOrigObj(ptr, SCIPgetBestSol(ptr)); }
    double value(Var v) const { return SCIPgetSolVal(ptr, SCIPgetBestSol(ptr), v); }
};

#if MAFFE_HAVE_GUROBI
struct Gurobi {
    using Var = int;
    using Row = int;
    struct LinearRow {
        std::string name;
        double lower, upper;
        std::vector<int> indices;
        std::vector<double> values;
    };
    GRBenv *env = nullptr;
    GRBmodel *model = nullptr;
    std::vector<LinearRow> rows;
    std::vector<double> solution;
    int variables = 0, status = 0;
    double dual = 1, primal = 0;
    LogLevel log_level;
    void check_grb(int code) const {
        if (code != 0)
            throw std::runtime_error("unrooted Gurobi error " + std::to_string(code) + ": " +
                                     (env ? GRBgeterrormsg(env) : "environment creation failed"));
    }
    static int callback(GRBmodel *, void *data, int where, void *user) {
        const auto &solver = *static_cast<Gurobi *>(user);
        if (where == GRB_CB_MESSAGE && logging::enabled(solver.log_level)) {
            char *message = nullptr;
            if (GRBcbget(data, where, GRB_CB_MSG_STRING, &message) == 0)
                scip_error_log::log_prefixed_lines("unrooted-gurobi: ", message);
        }
        return 0;
    }
    Gurobi(int leaves, const UnrootedOptions &options) : log_level(options.log_level) {
        try {
            check_grb(GRBemptyenv(&env));
            check_grb(GRBsetintparam(env, GRB_INT_PAR_OUTPUTFLAG, logging::enabled(log_level) ? 1 : 0));
            check_grb(GRBsetintparam(env, GRB_INT_PAR_LOGTOCONSOLE, 0));
            check_grb(GRBstartenv(env));
            check_grb(GRBnewmodel(env, &model, "experimental-unrooted-three-port", 0, nullptr, nullptr, nullptr, nullptr, nullptr));
            check_grb(GRBsetdblattr(model, GRB_DBL_ATTR_OBJCON, leaves));
            check_grb(GRBsetcallbackfunc(model, callback, this));
        } catch (...) {
            if (model)
                GRBfreemodel(model);
            if (env)
                GRBfreeenv(env);
            throw;
        }
    }
    Gurobi(const Gurobi &) = delete;
    Gurobi &operator=(const Gurobi &) = delete;
    ~Gurobi() {
        if (model)
            GRBfreemodel(model);
        if (env)
            GRBfreeenv(env);
    }
    double infinity() const { return GRB_INFINITY; }
    Var variable(const std::string &name, double cost, bool binary = true) {
        check_grb(GRBaddvar(model, 0, nullptr, nullptr, cost, 0, 1, binary ? GRB_BINARY : GRB_CONTINUOUS, name.c_str()));
        return variables++;
    }
    Row row(const std::string &name, double lower, double upper) {
        rows.push_back({name, lower, upper, {}, {}});
        return static_cast<int>(rows.size()) - 1;
    }
    void coefficient(Row row, Var var, double value) {
        rows[row].indices.push_back(var);
        rows[row].values.push_back(value);
    }
    void optimize(const Budget &budget) {
        check_grb(GRBupdatemodel(model));
        for (auto &row : rows) {
            budget.check();
            const bool equality = row.lower == row.upper;
            check_grb(GRBaddconstr(model, static_cast<int>(row.indices.size()), row.indices.data(), row.values.data(),
                                   equality ? GRB_EQUAL : GRB_LESS_EQUAL, row.upper, row.name.c_str()));
        }
        rows.clear();
        if (variables > 0) {
            std::vector<double> start(variables, 0);
            check_grb(GRBsetdblattrarray(model, GRB_DBL_ATTR_START, 0, variables, start.data()));
        }
        budget.check();
        if (budget.deadline)
            check_grb(GRBsetdblparam(GRBgetenv(model), GRB_DBL_PAR_TIMELIMIT,
                                     std::max(0.0, std::chrono::duration<double>(*budget.deadline - Clock::now()).count())));
        check_grb(GRBoptimize(model));
        check_grb(GRBgetintattr(model, GRB_INT_ATTR_STATUS, &status));
        int count = 0;
        check_grb(GRBgetintattr(model, GRB_INT_ATTR_SOLCOUNT, &count));
        check_grb(GRBgetdblattr(model, GRB_DBL_ATTR_OBJBOUND, &dual));
        if (count > 0) {
            solution.resize(variables);
            check_grb(GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &primal));
            if (variables > 0)
                check_grb(GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, variables, solution.data()));
            found_solution = true;
        }
    }
    bool found_solution = false;
    bool optimal() const { return status == GRB_OPTIMAL; }
    bool has_solution() const { return found_solution; }
    double bound() const { return dual; }
    double objective() const { return primal; }
    double value(Var v) const { return solution.at(v); }
};
using Solver = Gurobi;
#else
using Solver = Scip;
#endif

using SparseRow = std::vector<std::pair<int, int>>;

template <class Emit>
void difference(const SparseRow &a, const SparseRow &b, Emit emit) {
    auto i = a.begin(), j = b.begin();
    while (i != a.end() || j != b.end()) {
        if (j == b.end() || (i != a.end() && i->first < j->first)) {
            emit(i->first, i->second);
            ++i;
        } else if (i == a.end() || j->first < i->first) {
            emit(j->first, -j->second);
            ++j;
        } else {
            if (i->second != j->second)
                emit(i->first, i->second - j->second);
            ++i;
            ++j;
        }
    }
}

// Each load equals an original capacity expression. Differences cancel shared
// coefficients; [0,1] load bounds retain the original capacities exactly.
template <class Backend>
void encode_capacities(const std::set<std::vector<int>> &capacities, Backend &solver,
                       const std::vector<typename Backend::Var> &variables, const Budget &budget) {
    std::vector<SparseRow> rows;
    std::size_t terms = 0;
    bool binary_coefficients = true;
    for (const auto &support : capacities) {
        SparseRow row;
        for (int j : support) {
            if (!row.empty() && row.back().first == j) {
                ++row.back().second;
                binary_coefficients = false;
            } else
                row.emplace_back(j, 1);
        }
        terms += row.size();
        rows.push_back(std::move(row));
    }
    const int count = static_cast<int>(rows.size());
    if (count == 0)
        return;
    // Dense binary rows admit cheap XOR/popcount distances. Keep this scratch
    // storage proportional to the sparse input; otherwise merge sparse rows.
    const std::size_t words = (variables.size() + 63) / 64;
    std::vector<std::vector<std::uint64_t>> bits;
    if (binary_coefficients && words <= terms / rows.size()) {
        bits.assign(count, std::vector<std::uint64_t>(words));
        for (int r = 0; r < count; ++r)
            for (const auto &[j, _] : rows[r])
                bits[r][j / 64] |= std::uint64_t{1} << (j % 64);
    }
    // Prim's algorithm with an implicit empty root. Connecting to the root
    // emits the original row; other edges cost the difference plus two loads.
    std::vector<int> parent(count, -1), cost;
    for (const auto &row : rows)
        cost.push_back(static_cast<int>(row.size()));
    std::vector<bool> done(count), used(count);
    for (int step = 0; step < count; ++step) {
        budget.check();
        int v = -1;
        for (int j = 0; j < count; ++j)
            if (!done[j] && (v < 0 || cost[j] < cost[v]))
                v = j;
        done[v] = true;
        for (int w = 0; w < count; ++w) {
            if (done[w] || cost[w] <= 2)
                continue;
            int distance = 2;
            if (bits.empty())
                difference(rows[v], rows[w], [&](int, int) { ++distance; });
            else
                for (std::size_t j = 0; j < words && distance < cost[w]; ++j)
                    distance += std::popcount(bits[v][j] ^ bits[w][j]);
            if (distance < cost[w]) {
                cost[w] = distance;
                parent[w] = v;
            }
        }
    }
    for (int p : parent)
        if (p >= 0)
            used[p] = true;
    std::size_t sparse_terms = 0;
    for (int r = 0; r < count; ++r)
        sparse_terms += cost[r] + static_cast<int>(used[r]) - static_cast<int>(parent[r] >= 0);
    // Extra loads can slow presolve on small gains. Use the extended form only
    // when it removes at least two thirds of the capacity coefficients.
    if (sparse_terms > terms / 3) {
        std::ranges::fill(parent, -1);
        std::fill(used.begin(), used.end(), false);
    }
    std::vector<typename Backend::Var> loads(count);
    for (int r = 0; r < count; ++r)
        if (used[r])
            loads[r] = solver.variable("load_" + std::to_string(r), 0, false);
    const SparseRow empty;
    for (int r = 0; r < count; ++r) {
        budget.check();
        // Leaves need no load variable: retain their capacity as an inequality.
        const auto row = solver.row("capacity_" + std::to_string(r), used[r] ? 0 : -solver.infinity(), used[r] ? 0 : 1);
        difference(rows[r], parent[r] < 0 ? empty : rows[parent[r]],
                   [&](int j, int value) { solver.coefficient(row, variables[j], value); });
        if (parent[r] >= 0)
            solver.coefficient(row, loads[parent[r]], 1);
        if (used[r])
            solver.coefficient(row, loads[r], -1);
    }
}

// Keep connection indices stable for reconstruction. An activation whose port
// has just one connection is exactly that connection, including fractionally.
template <class Backend>
std::vector<typename Backend::Var> encode(const Model &model, Backend &solver, const Budget &budget) {
    const int n = model.leaves;
    std::vector<std::array<std::vector<int>, 3>> ports(model.states.size());
    int e = 0;
    for (const auto &[s, t] : model.edges) {
        budget.check();
        if (s >= n)
            ports[s - n][model.port(s, t)].push_back(e);
        if (t >= n)
            ports[t - n][model.port(t, s)].push_back(e);
        ++e;
    }
    // Singleton ports at the same state carry exactly the same flow. Merge
    // their connections transitively before creating any solver variables.
    UnionFind equivalent(static_cast<int>(model.edges.size()));
    for (const auto &state_ports : ports) {
        int first = -1;
        for (const auto &port : state_ports)
            if (port.size() == 1) {
                if (first < 0)
                    first = port.front();
                else
                    equivalent.join(first, port.front());
            }
    }
    std::vector<int> group(model.edges.size()), ids(model.edges.size(), -1);
    std::vector<double> costs;
    for (int j = 0; j < static_cast<int>(group.size()); ++j) {
        const int leader = equivalent.find(j);
        if (ids[leader] < 0) {
            ids[leader] = static_cast<int>(costs.size());
            costs.push_back(0);
        }
        group[j] = ids[leader];
        --costs[group[j]];
    }
    std::vector<int> activation(model.states.size(), -1);
    for (int s = 0; s < static_cast<int>(ports.size()); ++s)
        for (const auto &port : ports[s])
            if (port.size() == 1) {
                activation[s] = group[port.front()];
                ++costs[activation[s]];
                break;
            }
    std::vector<typename Backend::Var> variables;
    for (int j = 0; j < static_cast<int>(costs.size()); ++j) {
        budget.check();
        variables.push_back(solver.variable("connection_" + std::to_string(j), costs[j]));
    }
    std::set<std::vector<std::pair<int, int>>> balances;
    for (int s = 0; s < static_cast<int>(ports.size()); ++s) {
        budget.check();
        if (activation[s] < 0) {
            activation[s] = static_cast<int>(variables.size());
            variables.push_back(solver.variable("state_" + std::to_string(s), 1));
        }
        for (int p = 0; p < 3; ++p) {
            std::map<int, int> terms{{activation[s], -1}};
            for (int j : ports[s][p])
                ++terms[group[j]];
            std::vector<std::pair<int, int>> key;
            for (const auto &[j, value] : terms)
                if (value != 0)
                    key.emplace_back(j, value);
            if (key.empty())
                continue;
            if (key.front().second < 0)
                for (auto &[_, value] : key)
                    value = -value;
            if (!balances.insert(key).second)
                continue;
            auto row = solver.row("port_" + std::to_string(s) + "_" + std::to_string(p), 0, 0);
            for (const auto &[j, value] : key)
                solver.coefficient(row, variables[j], value);
        }
    }
    // Leaf capacity is implied by its adjacent internal vertex and the port
    // balances. For n <= 2 the single possible connection's bound suffices.
    // Construct one tree at a time and retain only distinct internal rows.
    std::set<std::vector<int>> capacities;
    for (int i = 0; i < static_cast<int>(model.trees.size()); ++i) {
        budget.check();
        std::vector<std::vector<int>> rows(model.trees[i].next.size());
        for (int s = 0; s < static_cast<int>(model.states.size()); ++s)
            rows[model.states[s].vertex[i]].push_back(activation[s]);
        int j = 0;
        for (const auto &[s, t] : model.edges) {
            budget.check();
            const int a = model.vertex(s, i), b = model.vertex(t, i);
            for (int v = model.trees[i].next[a][b]; v != b; v = model.trees[i].next[v][b])
                rows[v].push_back(group[j]);
            ++j;
        }
        for (int v = n; v < static_cast<int>(rows.size()); ++v) {
            budget.check();
            auto &indices = rows[v];
            std::ranges::sort(indices);
            if (indices.empty())
                continue;
            capacities.insert(std::move(indices));
        }
    }
    encode_capacities(capacities, solver, variables, budget);
    std::vector<typename Backend::Var> connections;
    connections.reserve(group.size());
    for (int j : group)
        connections.push_back(variables[j]);
    return connections;
}

std::string quote(const std::string &label) {
    if (label.find_first_of(" \t\r\n(),:;[]") == std::string::npos)
        return label;
    return "'" + label + "'";
}

std::string restrict_block(const Instance &instance, const std::vector<int> &block) {
    if (block.size() == 1)
        return quote(instance.labels[block[0]]) + ";";
    Graph graph = instance.trees.front();
    std::vector<bool> keep(graph.size());
    for (int a : block)
        keep[a] = true;
    std::vector<int> degree;
    for (const auto &neighbours : graph)
        degree.push_back(static_cast<int>(neighbours.size()));
    std::vector<int> todo;
    for (int v = 0; v < static_cast<int>(graph.size()); ++v)
        if (degree[v] <= 1 && !keep[v])
            todo.push_back(v);
    while (!todo.empty()) {
        const int v = todo.back();
        todo.pop_back();
        degree[v] = 0;
        for (int w : graph[v])
            if (degree[w] > 0) {
                --degree[w];
                if (degree[w] == 1 && !keep[w])
                    todo.push_back(w);
            }
    }
    const auto branch = [&](this auto &&self, int v, int parent) -> std::string {
        if (keep[v])
            return quote(instance.labels[v]);
        std::vector<std::string> parts;
        for (int w : graph[v])
            if (w != parent && degree[w] > 0)
                parts.push_back(self(w, v));
        if (parts.size() == 1)
            return parts[0];
        if (parts.size() != 2)
            throw std::logic_error("invalid unrooted restriction");
        return "(" + parts[0] + "," + parts[1] + ")";
    };
    // Serialize with an artificial degree-two root on the first leaf's edge.
    const int a = block.front();
    return "(" + quote(instance.labels[a]) + "," + branch(graph[a].front(), a) + ");";
}

void set_partition(UnrootedResult &result, const Instance &instance, const std::vector<std::vector<int>> &blocks) {
    result.partition.clear();
    result.forest.clear();
    for (const auto &block : blocks) {
        std::vector<std::string> labels;
        for (int a : block)
            labels.push_back(instance.labels[a]);
        result.partition.push_back(std::move(labels));
        result.forest.push_back(restrict_block(instance, block));
    }
    result.objective = static_cast<double>(blocks.size());
}

} // namespace

UnrootedResult solve_unrooted(const std::vector<std::string> &input, const UnrootedOptions &options) {
    const auto start = Clock::now();
    if (options.timeout_seconds && (!std::isfinite(*options.timeout_seconds) || *options.timeout_seconds < 0))
        throw std::invalid_argument("time limit must be finite and nonnegative");
    Budget budget;
    if (options.timeout_seconds &&
        *options.timeout_seconds < std::chrono::duration<double>(Clock::time_point::max() - start).count())
        budget.deadline = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(*options.timeout_seconds));
    const Instance instance = parse(input);
    const int n = static_cast<int>(instance.labels.size());
    UnrootedResult result;
    std::vector<std::vector<int>> singletons;
    for (int a = 0; a < n; ++a)
        singletons.push_back({a});
    set_partition(result, instance, singletons);
    if (logging::enabled(options.log_level))
        logging::line("experimental unrooted compact: trees=", input.size(), " leaves=", n, " rooted-reductions=disabled");
    try {
        const auto model = build(instance, budget);
        budget.check();
        Solver solver(n, options);
        const auto connections = encode(model, solver, budget);
        if (logging::enabled(options.log_level))
            logging::line("unrooted model: states=", model.states.size(), " connections=", connections.size());
        solver.optimize(budget);
        result.optimal = solver.optimal();
        result.lower_bound = std::max(1.0, solver.bound());
        if (solver.has_solution()) {
            UnionFind components(n + static_cast<int>(model.states.size()));
            int j = 0;
            for (const auto &[s, t] : model.edges)
                if (solver.value(connections[j++]) > 0.5)
                    components.join(s, t);
            std::map<int, std::vector<int>> groups;
            for (int a = 0; a < n; ++a)
                groups[components.find(a)].push_back(a);
            std::vector<std::vector<int>> blocks;
            for (auto &[_, block] : groups)
                blocks.push_back(std::move(block));
            std::ranges::sort(blocks);
            set_partition(result, instance, blocks);
            if (std::abs(result.objective - solver.objective()) > 1e-5)
                throw std::logic_error("unrooted forest objective does not match model");
        }
    } catch (const TimedOut &) {
        // The singleton incumbent remains valid if model construction times out.
    }
    if (logging::enabled(options.log_level))
        logging::line("unrooted result: objective=", result.objective, " lower-bound=", result.lower_bound,
                      " optimal=", result.optimal, " time-s=", std::chrono::duration<double>(Clock::now() - start).count());
    return result;
}

} // namespace maffe::experimental
