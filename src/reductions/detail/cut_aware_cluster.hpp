#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace maffe::detail {

enum class CutAwarePartialKind : std::uint8_t {
    NONE,
    BOTTOM,
    TOP,
};

struct CutAwareRepresentation {
    CutAwarePartialKind partial_kind = CutAwarePartialKind::NONE;
    int partial_component = -1;
    int partial_node = -1;
    std::vector<int> components;
};

struct CutAwareCluster {
    std::vector<int> leaves;
    std::array<CutAwareRepresentation, 2> representation;
};

} // namespace maffe::detail
