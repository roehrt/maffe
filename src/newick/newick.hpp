#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace maffe::newick {

struct Node {
    std::vector<int> children;
    std::string label;
};

struct Tree {
    std::vector<Node> nodes;
    int root = -1;
};

Tree parse(std::string_view text);

} // namespace maffe::newick
