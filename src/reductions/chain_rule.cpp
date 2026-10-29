#include "maffe/common.hpp"
#include "reductions/reductions.hpp"
#include "util/partition_ops.hpp"
#include "util/tree_ops.hpp"

#include <algorithm>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace maffe {
namespace {

struct ChainLiftData {
    std::vector<int> kept_leaves;
    std::vector<int> removed_leaves;
    int anchor_old_leaf = -1;
};

[[nodiscard]] std::optional<std::vector<int>> longest_common_chain(const AnnotatedInstance& instance) {
    if (instance.trees.empty())
        return std::nullopt;

    const int leaves = instance.trees.front().leaves();
    if (leaves <= static_cast<int>(instance.trees.size()) + 1)
        return std::nullopt;

    std::vector<std::vector<std::array<int, 2>>> children;
    children.reserve(instance.trees.size());
    for (const auto& tree : instance.trees)
        children.push_back(detail::tree_children(tree, detail::CutHandling::INCLUDE_CUT));

    std::optional<std::vector<int>> best;
    for (int node = leaves; node < instance.trees.front().vertices(); ++node) {
        const auto [a, b] = children[0][node];
        if (a < 0 || b < 0 || a >= leaves || b >= leaves)
            continue;

        bool common_cherry = true;
        std::vector<int> spine(instance.trees.size());
        for (int i = 0; i < static_cast<int>(instance.trees.size()); ++i) {
            const auto parent_a = instance.trees[i].parent[a];
            if (parent_a < 0 || parent_a != instance.trees[i].parent[b]) {
                common_cherry = false;
                break;
            }
            spine[i] = parent_a;
        }
        if (!common_cherry)
            continue;

        std::vector<int> chain{std::min(a, b), std::max(a, b)};
        std::vector<int> next_spine(instance.trees.size());
        for (;;) {
            std::optional<int> next_leaf;
            bool extend = true;
            for (int i = 0; i < static_cast<int>(instance.trees.size()); ++i) {
                const int parent = instance.trees[i].parent[spine[i]];
                if (parent < 0) {
                    extend = false;
                    break;
                }

                const int sibling = detail::other_child(children[i], parent, spine[i]);
                if (sibling < 0 || sibling >= leaves) {
                    extend = false;
                    break;
                }

                if (!next_leaf)
                    next_leaf = sibling;
                else if (*next_leaf != sibling) {
                    extend = false;
                    break;
                }
                next_spine[i] = parent;
            }
            if (!extend)
                break;
            chain.push_back(*next_leaf);
            spine.swap(next_spine);
        }

        if (chain.size() <= instance.trees.size() + 1)
            continue;
        if (!best || chain.size() > best->size())
            best = std::move(chain);
    }

    return best;
}

[[nodiscard]] Result lift_chain(
    Result result,
    const ChainLiftData& chain
) {
    if (!result.feasible)
        return result;

    bool inserted = false;
    for (auto& block : result.partition) {
        for (int& leaf : block)
            leaf = chain.kept_leaves[leaf];

        if (std::ranges::find(block, chain.anchor_old_leaf) != block.end()) {
            block.insert(block.end(), chain.removed_leaves.begin(), chain.removed_leaves.end());
            inserted = true;
        }
    }

    if (!inserted)
        throw std::runtime_error("chain reduction could not find anchor component");

    detail::sort_partition_blocks(result.partition);
    return result;
}

} // namespace

std::optional<Reduced> try_chain_rule(const AnnotatedInstance& instance) {
    AnnotatedInstance current = instance;
    std::vector<ChainLiftData> lifts;

    for (;;) {
        const auto chain = longest_common_chain(current);
        if (!chain)
            break;

        const int keep = static_cast<int>(current.trees.size()) + 1;
        std::vector<int> leaf_map(current.trees.front().leaves(), -1);
        std::vector<int> kept_leaves;
        std::vector<int> removed_leaves;
        kept_leaves.reserve(current.trees.front().leaves() - (static_cast<int>(chain->size()) - keep));
        removed_leaves.reserve(static_cast<int>(chain->size()) - keep);

        std::vector<char> removed_mask(current.trees.front().leaves(), false);
        for (int i = keep; i < static_cast<int>(chain->size()); ++i) {
            removed_mask[(*chain)[i]] = true;
            removed_leaves.push_back((*chain)[i]);
        }

        int next = 0;
        for (int leaf = 0; leaf < current.trees.front().leaves(); ++leaf) {
            if (removed_mask[leaf])
                continue;
            leaf_map[leaf] = next++;
            kept_leaves.push_back(leaf);
        }

        lifts.push_back({
            .kept_leaves = std::move(kept_leaves),
            .removed_leaves = std::move(removed_leaves),
            .anchor_old_leaf = (*chain)[keep - 1],
        });
        current = detail::restrict_instance_by_leaf_map(current, leaf_map, "chain reduction removed all leaves");
    }

    if (lifts.empty())
        return std::nullopt;

    const int reductions = static_cast<int>(lifts.size());
    return Reduced{
        .instance = std::move(current),
        .lift = [lifts = std::move(lifts)](Result result) {
            for (const auto& chain : lifts | std::views::reverse)
                result = lift_chain(std::move(result), chain);
            return result;
        },
        .reduction_count = reductions,
    };
}

} // namespace maffe
