#include "maffe.hpp"
#include "config.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "maffe/common.hpp"
#include "reductions/reductions.hpp"
#include "newick/newick.hpp"
#include "util/log.hpp"
#include "util/tree_index.hpp"

namespace maffe {
namespace {
struct ParsedTrees {
    std::vector<newick::Tree> trees;
    std::vector<std::string> leaf_names;
};

std::vector<std::string> leaf_labels(const newick::Tree& tree) {
    std::vector<std::string> labels;
    const auto walk = [&](this auto&& self, const int node) -> void {
        const auto& [children, label] = tree.nodes[node];
        if (children.empty()) {
            if (label.empty())
                throw std::invalid_argument("empty leaf label");
            labels.push_back(label);
            return;
        }

        if (children.size() != 2)
            throw std::invalid_argument("input trees must be rooted and binary");

        self(children[0]);
        self(children[1]);
    };

    walk(tree.root);
    return labels;
}

std::unordered_map<std::string, int> leaf_index(const std::vector<std::string>& labels) {
    std::unordered_map<std::string, int> index;
    index.reserve(labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (!index.emplace(labels[i], static_cast<int>(i)).second)
            throw std::invalid_argument("duplicate leaf label");
    }
    return index;
}

std::string format_seconds(const double seconds) {
    return std::format("{:.3f}s", seconds);
}

Tree annotate_tree(const newick::Tree& input, const std::unordered_map<std::string, int>& leaf_index) {
    Tree tree{
        .parent = std::vector<int>(input.nodes.size(), -2),
        .edge_state = std::vector<EdgeState>(input.nodes.size(), EdgeState::UNKNOWN),
    };
    std::vector<int> index(input.nodes.size(), -1);
    int next = static_cast<int>(leaf_index.size());

    const auto visit = [&](this auto&& self, const int node) -> void {
        const auto& [children, label] = input.nodes[node];
        if (children.empty()) {
            const auto it = leaf_index.find(label);
            if (it == leaf_index.end())
                throw std::invalid_argument("leaf set mismatch between trees");
            index[node] = it->second;
            return;
        }

        if (children.size() != 2)
            throw std::invalid_argument("input trees must be rooted and binary");

        for (const auto child : children)
            self(child);

        const int parent = next++;
        for (const auto child : children)
            tree.parent[index[child]] = parent;
        index[node] = parent;
    };

    visit(input.root);
    tree.parent[index[input.root]] = -1;
    return tree;
}

class RestrictionBuilder {
public:
    explicit RestrictionBuilder(const newick::Tree& source_tree)
        : source(source_tree), index(parents(source_tree)) {
        for (int node = 0; node < static_cast<int>(source.nodes.size()); ++node) {
            const auto& [children, label] = source.nodes[node];
            if (children.empty() && !leaf_node.emplace(label, node).second)
                throw std::invalid_argument("duplicate leaf label");
        }
    }

    [[nodiscard]] std::string restrict_block(const std::vector<std::string>& leaves) const {
        if (leaves.empty())
            throw std::invalid_argument("subtree needs at least one leaf");

        std::vector<int> selected;
        selected.reserve(leaves.size());
        for (const auto& leaf : leaves) {
            const auto it = leaf_node.find(leaf);
            if (it == leaf_node.end())
                throw std::invalid_argument("unknown leaf name");
            selected.push_back(it->second);
        }

        const auto tree = index.virtual_tree(selected);
        std::vector<std::vector<int>> children(tree.nodes.size());
        for (int i = 1; i < static_cast<int>(tree.nodes.size()); ++i)
            children[tree.parent[i]].push_back(i);
        return build_newick(0, tree.nodes, children) + ";";
    }

private:
    [[nodiscard]] static std::vector<int> parents(const newick::Tree& tree) {
        if (tree.root < 0 || tree.root >= static_cast<int>(tree.nodes.size()))
            throw std::invalid_argument("invalid source tree");
        std::vector<int> parent(tree.nodes.size(), -1);
        for (int node = 0; node < static_cast<int>(tree.nodes.size()); ++node) {
            const auto& children = tree.nodes[node].children;
            if (!children.empty() && children.size() != 2)
                throw std::invalid_argument("input trees must be rooted and binary");
            for (const int child : children)
                parent[child] = node;
        }
        return parent;
    }

    [[nodiscard]] std::string build_newick(
        const int node,
        const std::vector<int>& virtual_nodes,
        const std::vector<std::vector<int>>& children
    ) const {
        const auto& local_children = children[node];
        if (local_children.empty())
            return source.nodes[virtual_nodes[node]].label;
        if (local_children.size() == 1)
            return build_newick(local_children.front(), virtual_nodes, children);
        return "(" + build_newick(local_children[0], virtual_nodes, children) + "," +
            build_newick(local_children[1], virtual_nodes, children) + ")";
    }

    const newick::Tree& source;
    std::unordered_map<std::string, int> leaf_node;
    detail::TreeIndex index;
};

ParsedTrees parse_trees(const std::vector<std::string>& input) {
    if (input.empty())
        throw std::invalid_argument("instance must contain at least one tree");

    ParsedTrees parsed;
    parsed.trees.reserve(input.size());
    std::unordered_map<std::string, int> index;
    for (const auto& line : input) {
        auto tree = newick::parse(line);
        auto labels = leaf_labels(tree);
        auto labels_index = leaf_index(labels);

        if (parsed.trees.empty()) {
            if (labels.empty())
                throw std::invalid_argument("trees must contain at least one leaf");
            index = std::move(labels_index);
            parsed.leaf_names = std::move(labels);
        } else if (labels_index.size() != index.size()) {
            throw std::invalid_argument("all trees must have the same leaf set");
        } else {
            for (const auto& leaf : labels) {
                if (!index.contains(leaf))
                    throw std::invalid_argument("all trees must have the same leaf set");
            }
        }

        parsed.trees.push_back(std::move(tree));
    }

    return parsed;
}

} // namespace

std::vector<std::vector<std::string>> solve(const std::vector<std::string>& trees) {
    return solve(trees, {});
}

std::vector<std::vector<std::string>> solve(const std::vector<std::string>& trees, const SolveOptions& options) {
    const ParsedTrees parsed = parse_trees(trees);
    const auto start = std::chrono::steady_clock::now();
    constexpr std::string_view git_hash = MAFFE_GIT_HASH;
    if (logging::enabled(options.log_level)) {
        logging::line(
            "solver: maffe ", MAFFE_PROJECT_VERSION,
            " git=", git_hash.substr(0, std::min<std::size_t>(git_hash.size(), 6)),
            MAFFE_GIT_DIRTY ? "-dirty" : "",
            " scip=", MAFFE_SCIP_VERSION,
            " lps=", MAFFE_SCIP_LPS,
            " highs=", MAFFE_HIGHS_VERSION
        );
        logging::line("instance: trees=", parsed.trees.size(), " leaves=", parsed.leaf_names.size());
        if (options.timeout_seconds.has_value())
            logging::line("limit: time-s=", *options.timeout_seconds);
        if (options.acceptable_factor != 1.0 || options.acceptable_offset != 0) {
            logging::line(
                "accept: factor=", options.acceptable_factor,
                " offset=", options.acceptable_offset
            );
        }
    }

    AnnotatedInstance annotated;
    annotated.trees.reserve(parsed.trees.size());

    const auto index = leaf_index(parsed.leaf_names);
    for (const auto& tree : parsed.trees)
        annotated.trees.push_back(annotate_tree(tree, index));

    const auto partition = solve_annotated(
        std::move(annotated),
        options.timeout_seconds,
        options.acceptable_factor,
        options.acceptable_offset,
        options.log_level,
        options.heuristic_mode
    );

    std::vector<std::vector<std::string>> solution;
    solution.reserve(partition.partition.size());
    for (const auto& block : partition.partition) {
        std::vector<std::string> leaves;
        leaves.reserve(block.size());
        for (const auto leaf : block)
            leaves.push_back(parsed.leaf_names.at(leaf));
        solution.push_back(std::move(leaves));
    }
    if (logging::enabled(options.log_level)) {
        logging::line("result: agreement forest");
        logging::line("result:   components=", solution.size());
        logging::line(
            "result:   time=",
            format_seconds(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count())
        );
    }
    return solution;
}

std::string restrict(const std::vector<std::string>& trees, const std::vector<std::string>& leaves) {
    const ParsedTrees parsed = parse_trees(trees);
    return RestrictionBuilder(parsed.trees.front()).restrict_block(leaves);
}

std::vector<std::string> restrict(
    const std::vector<std::string>& trees,
    const std::vector<std::vector<std::string>>& partition
) {
    const ParsedTrees parsed = parse_trees(trees);
    RestrictionBuilder builder(parsed.trees.front());
    std::vector<std::string> output;
    output.reserve(partition.size());
    for (const auto& leaves : partition)
        output.push_back(builder.restrict_block(leaves));
    return output;
}

} // namespace maffe
