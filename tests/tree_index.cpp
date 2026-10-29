#include "branchandprice/master/root_master.hpp"
#include "heuristic/node_coverage.hpp"
#include "reductions/detail/related_side.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("tree index regression");
}

int naive_lca(const std::vector<int>& parent, int a, int b) {
    std::set<int> ancestors;
    for (; a >= 0; a = parent[a])
        ancestors.insert(a);
    for (; b >= 0; b = parent[b]) {
        if (ancestors.contains(b))
            return b;
    }
    return -1;
}

void check_tree(const maffe::Tree& tree, std::mt19937& random) {
    const auto layout = maffe::build_root_master_layout({{tree}});
    const auto& index = layout.tree_index.front();
    auto parents = tree.parent;
    for (int u = 0; u < tree.vertices(); ++u) {
        if (tree.edge_state[u] == maffe::EdgeState::CUT)
            parents[u] = -1;
    }
    for (int a = 0; a < tree.vertices(); ++a) {
        for (int b = 0; b < tree.vertices(); ++b) {
            const int expected = naive_lca(parents, a, b);
            require(index.lca(a, b) == expected);
            require(index.ancestor(a, b) == (expected == a));
        }
    }
    require(index.virtual_tree({}).nodes.empty());
    std::vector<std::vector<int>> blocks;
    std::vector<int> expected_cover(tree.vertices() + 3, 0);
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<int> leaves;
        const int seed = random() % tree.leaves();
        for (int leaf = 0; leaf < tree.leaves(); ++leaf) {
            if (naive_lca(parents, seed, leaf) >= 0 && (leaf == seed || random() % 2))
                leaves.push_back(leaf);
        }
        int root = leaves.front();
        for (const int leaf : leaves)
            root = naive_lca(parents, root, leaf);
        std::set<int> edges, vertices;
        for (int leaf : leaves) {
            for (int u = leaf; u != root; u = parents[u]) {
                edges.insert(u);
                if (parents[u] >= tree.leaves())
                    vertices.insert(parents[u]);
            }
        }
        blocks.push_back(leaves);
        for (const int vertex : vertices)
            ++expected_cover[vertex];
        const auto column = maffe::build_root_master_column({{tree}}, layout, leaves);
        require(column.used_edges.front() == std::vector<int>(edges.begin(), edges.end()));
        require(column.used_vertices.front() == std::vector<int>(vertices.begin(), vertices.end()));

        // The virtual tree accepts duplicate selections and arbitrary input order.
        leaves.push_back(leaves.front());
        std::shuffle(leaves.begin(), leaves.end(), random);
        const auto virtual_tree = index.virtual_tree(leaves);
        require(virtual_tree.nodes.front() == root);
        require(virtual_tree.parent.front() == -1);
        std::set<int> virtual_edges;
        for (int i = 1; i < static_cast<int>(virtual_tree.nodes.size()); ++i) {
            require(virtual_tree.parent[i] >= 0 && virtual_tree.parent[i] < i);
            const int ancestor = virtual_tree.nodes[virtual_tree.parent[i]];
            for (int u = virtual_tree.nodes[i]; u != ancestor; u = parents[u])
                require(virtual_edges.insert(u).second);
        }
        require(virtual_edges == edges);
    }
    for (int leaf = 1; leaf < tree.leaves(); ++leaf) {
        if (naive_lca(parents, 0, leaf) >= 0)
            continue;
        blocks.push_back({0, leaf}); // Disconnected blocks contribute no coverage.
        bool rejected = false;
        try {
            const std::vector selected{0, leaf};
            (void)index.virtual_tree(selected);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected);
        rejected = false;
        try {
            const std::vector selected{0, leaf};
            (void)maffe::build_root_master_column({{tree}}, layout, selected);
        } catch (const maffe::DisconnectedColumn&) {
            rejected = true;
        }
        require(rejected);
    }
    const maffe::heuristic::NodeCoverage coverage(tree);
    std::vector<int> cover(tree.vertices() + 3, 7);
    coverage.compute(blocks, cover);
    require(cover == expected_cover);
    coverage.compute({}, cover);
    require(std::ranges::all_of(cover, [](int count) { return count == 0; }));
}

} // namespace

int main() {
    // Only the third block can accept the placeholder without crossing another block.
    const maffe::Tree first{{5, 5, 7, 6, 6, 8, 7, 8, -1}, std::vector<maffe::EdgeState>(9)};
    const maffe::Result without{.partition = {{0}, {1}, {3, 2}}};
    const auto extended = maffe::detail::try_extend_with_placeholder({{first, first}}, without);
    require(extended && extended->partition == std::vector<std::vector<int>>{{0}, {1}, {2, 3, 4}});
    require(without.partition == std::vector<std::vector<int>>{{0}, {1}, {3, 2}});
    const maffe::Tree second{{6, 5, 7, 7, 5, 6, 8, 8, -1}, std::vector<maffe::EdgeState>(9)};
    require(!maffe::detail::try_extend_with_placeholder({{first, second}}, without));

    std::mt19937 random(20260913);
    for (int leaves = 1; leaves <= 64; ++leaves) {
        maffe::Tree tree{
            .parent = std::vector<int>(2 * leaves - 1, -1),
            .edge_state = std::vector<maffe::EdgeState>(2 * leaves - 1, maffe::EdgeState::UNKNOWN),
        };
        std::vector<int> active(leaves);
        std::iota(active.begin(), active.end(), 0);
        for (int node = leaves; node < tree.vertices(); ++node) {
            std::shuffle(active.begin(), active.end(), random);
            for (int child = 0; child < 2; ++child) {
                tree.parent[active.back()] = node;
                active.pop_back();
            }
            active.push_back(node);
        }
        check_tree(tree, random);
        for (int u = 0; u < tree.vertices(); ++u) {
            if (random() % 4 == 0)
                tree.edge_state[u] = maffe::EdgeState::CUT;
        }
        check_tree(tree, random);
    }
}
