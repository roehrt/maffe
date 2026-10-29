#include "heuristic/node_coverage.hpp"

#include <algorithm>
#include <cstddef>

namespace maffe::heuristic {

void NodeCoverage::compute(std::span<const std::vector<int>> partition, std::vector<int>& cover) const {
    cover.resize(std::max(cover.size(), index_.tin.size()));
    std::ranges::fill(cover, 0);
    std::vector<std::size_t> seen(index_.tin.size(), partition.size());
    for (std::size_t block = 0; block < partition.size(); ++block) {
        const auto& leaves = partition[block];
        if (leaves.size() < 2)
            continue;
        int root = leaves.front();
        for (const int leaf : leaves) {
            root = index_.lca(root, leaf);
            if (root < 0)
                break;
        }
        if (root < leaves_)
            continue;

        seen[root] = block;
        ++cover[root];
        for (const int leaf : leaves) {
            // Stop at the first already-covered vertex, including the LCA.
            for (int u = index_.parent(leaf); seen[u] != block; u = index_.parent(u)) {
                seen[u] = block;
                ++cover[u];
            }
        }
    }
}

} // namespace maffe::heuristic
