#include "newick/newick.hpp"

#include <cstddef>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace maffe::newick {
namespace {

class Parser {
public:
    explicit Parser(const std::string_view text) : text(text) {}

    Tree parse() {
        Tree tree;
        tree.root = parse_subtree(tree);
        skip_space();
        if (peek() == ';')
            ++pos;
        skip_space();
        if (pos != text.size())
            throw std::invalid_argument("unexpected trailing Newick input");
        return tree;
    }

private:
    int parse_subtree(Tree& tree) {
        skip_space();
        Node node;
        if (peek() == '(') {
            ++pos;
            while (true) {
                node.children.push_back(parse_subtree(tree));
                skip_space();
                if (peek() == ',') {
                    ++pos;
                    continue;
                }
                break;
            }

            expect(')');
        }

        node.label = parse_label();
        if (node.children.empty() && node.label.empty())
            throw std::invalid_argument("missing leaf label in Newick input");
        skip_branch_length();

        tree.nodes.push_back(std::move(node));
        return static_cast<int>(tree.nodes.size()) - 1;
    }

    std::string parse_label() {
        skip_space();
        if (peek() == '\'') {
            ++pos;
            std::string label;
            while (pos < text.size() && text[pos] != '\'')
                label.push_back(text[pos++]);
            expect('\'');
            return label;
        }

        std::string label;
        while (pos < text.size()) {
            const char c = text[pos];
            if (std::isspace(static_cast<unsigned char>(c)) || c == '(' || c == ')' || c == ',' || c == ':' ||
                c == ';')
                break;
            label.push_back(c);
            ++pos;
        }
        skip_space();
        return label;
    }

    void skip_branch_length() {
        skip_space();
        if (peek() != ':')
            return;
        ++pos;
        while (pos < text.size()) {
            const char c = text[pos];
            if (std::isspace(static_cast<unsigned char>(c)) || c == '(' || c == ')' || c == ',' || c == ';')
                break;
            ++pos;
        }
        skip_space();
    }

    void skip_space() {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
            ++pos;
    }

    void expect(const char c) {
        skip_space();
        if (peek() != c)
            throw std::invalid_argument("invalid Newick input");
        ++pos;
    }

    [[nodiscard]] char peek() const {
        return pos < text.size() ? text[pos] : '\0';
    }

    std::string_view text;
    std::size_t pos = 0;
};

} // namespace

Tree parse(const std::string_view text) {
    return Parser(text).parse();
}

} // namespace maffe::newick
