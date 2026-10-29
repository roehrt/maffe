#pragma once

#include "maffe/common.hpp"
#include "util/tree_index.hpp"

namespace maffe::compact {

class LcaHelper : public detail::TreeIndex {
public:
    explicit LcaHelper(const Tree& tree) : TreeIndex(tree.parent), leaves_(tree.leaves()) {}
    [[nodiscard]] int leaves() const { return leaves_; }

private:
    int leaves_;
};

} // namespace maffe::compact
