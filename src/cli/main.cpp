#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config.hpp"
#include "maffe.hpp"

namespace {

#ifndef MAFFE_COMPETITION_TIMEOUT_SECONDS
#define MAFFE_COMPETITION_TIMEOUT_SECONDS (-1)
#endif

#ifndef MAFFE_COMPETITION_HEURISTIC_BUILD
#define MAFFE_COMPETITION_HEURISTIC_BUILD 0
#endif

#ifndef MAFFE_COMPETITION_LOWERBOUND_BUILD
#define MAFFE_COMPETITION_LOWERBOUND_BUILD 0
#endif

#if MAFFE_COMPETITION_BUILD
constexpr bool kAcceptanceEnabled =
    MAFFE_COMPETITION_HEURISTIC_BUILD != 0 || MAFFE_COMPETITION_LOWERBOUND_BUILD != 0;
#else
constexpr bool kAcceptanceEnabled = true;
#endif

#if !MAFFE_COMPETITION_BUILD
[[noreturn]] void usage() {
    std::cout << "Usage: ./maffe [-q|--quiet] [-v|--verbose] [-a factor|--accept-factor factor] [-b offset|--accept-offset offset] [input_file]\n";
    std::exit(1);
}
#endif

std::optional<double> competition_timeout() {
#if MAFFE_COMPETITION_BUILD
    if constexpr (MAFFE_COMPETITION_TIMEOUT_SECONDS >= 0)
        return static_cast<double>(MAFFE_COMPETITION_TIMEOUT_SECONDS);
#endif
    return std::nullopt;
}

std::string_view trim(std::string_view line) {
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front())) != 0)
        line.remove_prefix(1);
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back())) != 0)
        line.remove_suffix(1);
    return line;
}

struct ParsedInput {
    std::vector<std::string> trees;
    std::optional<double> acceptable_factor;
    std::optional<int> acceptable_offset;
};

ParsedInput read_input() {
    std::vector<std::string> trees;
    std::optional<double> acceptable_factor;
    std::optional<int> acceptable_offset;

    std::string line;
    while (std::getline(std::cin, line)) {
        const std::string_view trimmed = trim(line);
        if (trimmed.empty())
            continue;

        if (trimmed.front() == '#') {
            if (trimmed.starts_with("#a")) {
                std::istringstream stream{std::string(trimmed)};
                std::string tag;
                double factor = 0.0;
                int offset = 0;
                if (stream >> tag >> factor >> offset; tag == "#a") {
                    acceptable_factor = factor;
                    acceptable_offset = offset;
                }
            }
            continue;
        }

        std::string normalized(trimmed);
        std::erase_if(normalized, [](const unsigned char c) { return std::isspace(c) != 0; });
        if (!normalized.empty())
            trees.push_back(std::move(normalized));
    }

    return ParsedInput{
        .trees = std::move(trees),
        .acceptable_factor = acceptable_factor,
        .acceptable_offset = acceptable_offset,
    };
}

} // namespace

int main(int argc, char** argv) {
#if MAFFE_COMPETITION_BUILD
    std::signal(SIGTERM, SIG_IGN);
#endif

    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::optional<double> acceptable_factor;
    std::optional<int> acceptable_offset;
    maffe::LogLevel log_level = maffe::LogLevel::NORMAL;
#if !MAFFE_COMPETITION_BUILD
    std::optional<std::string> input_path;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "-q" || arg == "--quiet") {
            log_level = maffe::LogLevel::QUIET;
            continue;
        }
        if (arg == "-v" || arg == "--verbose") {
            log_level = maffe::LogLevel::VERBOSE;
            continue;
        }
        if (arg == "-a" || arg == "--accept-factor") {
            if (++i >= argc)
                usage();
            try {
                acceptable_factor = std::stod(argv[i]);
            } catch (...) {
                usage();
            }
            continue;
        }
        if (arg == "-b" || arg == "--accept-offset") {
            if (++i >= argc)
                usage();
            try {
                acceptable_offset = std::stoi(argv[i]);
            } catch (...) {
                usage();
            }
            continue;
        }
        if (input_path.has_value())
            usage();
        input_path = std::string(arg);
    }
    if (input_path.has_value()) {
        if (!std::filesystem::exists(*input_path))
            usage();
        if (freopen(input_path->c_str(), "r", stdin) == nullptr)
            usage();
    }
#else
    (void)argc;
    (void)argv;
#endif

    auto parsed = read_input();
    const auto solution = maffe::solve(parsed.trees, maffe::SolveOptions{
        .timeout_seconds = competition_timeout(),
        .acceptable_factor = kAcceptanceEnabled
            ? acceptable_factor.value_or(parsed.acceptable_factor.value_or(1.0))
            : 1.0,
        .acceptable_offset = kAcceptanceEnabled
            ? acceptable_offset.value_or(parsed.acceptable_offset.value_or(0))
            : 0,
        .log_level = log_level,
        .heuristic_mode = MAFFE_COMPETITION_HEURISTIC_BUILD != 0,
    });
    const auto restricted = maffe::restrict(parsed.trees, solution);
    for (const auto& component : restricted)
        std::cout << component << '\n';

    return 0;
}
