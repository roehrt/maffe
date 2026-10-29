#pragma once

#include "util/tree_index.hpp"

#include <span>
#include <vector>

namespace maffe::heuristic {

// Reuse the forest index across pricing rounds and candidate columns.
class NodeCoverage {
public:
    explicit NodeCoverage(const Tree& tree) : index_(tree), leaves_(tree.leaves()) {}
    void compute(std::span<const std::vector<int>> partition, std::vector<int>& cover) const;

private:
    detail::TreeIndex index_;
    int leaves_;
};

} // namespace maffe::heuristic
