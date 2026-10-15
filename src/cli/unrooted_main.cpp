#include "maffe_unrooted.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

int main(int argc, char **argv) {
    try {
        maffe::experimental::UnrootedOptions options;
        std::string input_path;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: maffe-unrooted [-q|--quiet] [-v|--verbose] [--time-limit seconds] [input_file]\n"
                          << "Experimental compact solver for multiple unrooted binary trees.\n";
                return 0;
            } else if (arg == "-q" || arg == "--quiet") {
                options.log_level = maffe::LogLevel::QUIET;
            } else if (arg == "-v" || arg == "--verbose") {
                options.log_level = maffe::LogLevel::VERBOSE;
            } else if (arg == "--time-limit") {
                if (++i == argc)
                    throw std::invalid_argument("--time-limit requires seconds");
                std::size_t consumed = 0;
                const std::string value(argv[i]);
                options.timeout_seconds = std::stod(value, &consumed);
                if (consumed != value.size())
                    throw std::invalid_argument("invalid time limit");
            } else if (arg.starts_with('-')) {
                throw std::invalid_argument("unknown option: " + std::string(arg));
            } else if (!input_path.empty()) {
                throw std::invalid_argument("only one input file is supported");
            } else {
                input_path = arg;
            }
        }
        std::ifstream file;
        if (!input_path.empty()) {
            file.open(input_path);
            if (!file)
                throw std::runtime_error("cannot open input file: " + input_path);
        }
        std::istream &stream = input_path.empty() ? std::cin : file;
        std::vector<std::string> trees;
        for (std::string line; std::getline(stream, line);) {
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first != std::string::npos && line[first] != '#')
                trees.push_back(std::move(line));
        }
        const auto result = maffe::experimental::solve_unrooted(trees, options);
        for (const auto &component : result.forest)
            std::cout << component << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "maffe-unrooted: " << error.what() << '\n';
        return 1;
    }
}
