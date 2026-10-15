#pragma once

#include "maffe.hpp"

namespace maffe::experimental {

struct UnrootedOptions {
    std::optional<double> timeout_seconds;
    LogLevel log_level = LogLevel::NORMAL;
};

struct UnrootedResult {
    std::vector<std::vector<std::string>> partition;
    std::vector<std::string> forest;
    double objective = 0.0;
    double lower_bound = 1.0;
    bool optimal = false;
};

// Requires at least two unrooted binary trees with identical, unique leaf labels.
// A degree-two Newick root is suppressed. No rooted reductions are used.
UnrootedResult solve_unrooted(
    const std::vector<std::string> &trees,
    const UnrootedOptions &options = {});

} // namespace maffe::experimental
