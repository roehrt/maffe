#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.hpp"
#include "maffe.hpp"
#include "maffe_unrooted.hpp"

#include <cctype>
#include <format>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace py = pybind11;

namespace {

[[nodiscard]] maffe::LogLevel parse_log_level(const py::handle value, const py::handle enum_type) {
    if (py::isinstance<py::bool_>(value))
        throw py::type_error("log_level must be an int, str, or LogLevel");
    if (py::isinstance<py::str>(value)) {
        auto level = value.cast<std::string>();
        for (char& c : level)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (level == "0" || level == "quiet" || level == "q")
            return maffe::LogLevel::QUIET;
        if (level == "1" || level == "normal" || level == "n")
            return maffe::LogLevel::NORMAL;
        if (level == "2" || level == "verbose" || level == "v")
            return maffe::LogLevel::VERBOSE;
        throw py::value_error("log_level must be quiet/normal/verbose or 0/1/2");
    }
    if (py::isinstance(value, enum_type))
        return value.cast<maffe::LogLevel>();
    if (!py::isinstance<py::int_>(value))
        throw py::type_error("log_level must be an int, str, or LogLevel");
    switch (value.cast<int>()) {
    case 0:
        return maffe::LogLevel::QUIET;
    case 1:
        return maffe::LogLevel::NORMAL;
    case 2:
        return maffe::LogLevel::VERBOSE;
    default:
        throw py::value_error("log_level must be 0, 1, or 2");
    }
}

[[nodiscard]] std::vector<std::vector<std::string>> solve_partition(
    const std::vector<std::string>& trees,
    const double approx_factor,
    const int approx_offset,
    const maffe::LogLevel log_level
) {
    maffe::SolveOptions options;
    options.acceptable_factor = approx_factor;
    options.acceptable_offset = approx_offset;
    options.log_level = log_level;
    return maffe::solve(trees, options);
}

[[nodiscard]] std::vector<std::vector<std::string>> solve_impl(
    const std::vector<std::string>& trees,
    const double approx_factor,
    const int approx_offset,
    const maffe::LogLevel log_level
) {
    py::gil_scoped_release release;
    return solve_partition(trees, approx_factor, approx_offset, log_level);
}

[[nodiscard]] std::string restrict_tree_impl(
    const std::vector<std::string>& trees,
    const std::vector<std::string>& leaves
) {
    return maffe::restrict(trees, leaves);
}

[[nodiscard]] std::vector<std::string> restrict_solution_impl(
    const std::vector<std::string>& trees,
    const std::vector<std::vector<std::string>>& partition
) {
    return maffe::restrict(trees, partition);
}

[[nodiscard]] std::string strip_trailing_semicolon(std::string tree) {
    while (!tree.empty() && std::isspace(static_cast<unsigned char>(tree.back())) != 0)
        tree.pop_back();
    if (!tree.empty() && tree.back() == ';')
        tree.pop_back();
    return tree;
}

[[nodiscard]] std::vector<std::string> rho_extended_trees(
    const std::string& first,
    const std::string& second
) {
    const auto rho = std::format("rho_{:x}", std::hash<std::string>{}(std::format("{}\n{}", first, second)));
    const auto extend = [&](const std::string& tree) {
        return std::format("({},{});", strip_trailing_semicolon(tree), rho);
    };
    return {extend(first), extend(second)};
}

[[nodiscard]] int rspr_dist_impl(
    const std::string& first,
    const std::string& second,
    const double approx_factor,
    const int approx_offset,
    const maffe::LogLevel log_level
) {
    const auto trees = rho_extended_trees(first, second);
    py::gil_scoped_release release;
    return static_cast<int>(solve_partition(trees, approx_factor, approx_offset, log_level).size()) - 1;
}

} // namespace

PYBIND11_MODULE(maffe, m) {
    m.doc() = "Python bindings for maffe";
    m.attr("__version__") = MAFFE_PROJECT_VERSION;
    const auto log_level_type = py::enum_<maffe::LogLevel>(m, "LogLevel")
        .value("QUIET", maffe::LogLevel::QUIET)
        .value("NORMAL", maffe::LogLevel::NORMAL)
        .value("VERBOSE", maffe::LogLevel::VERBOSE);

    m.def(
        "solve",
        [log_level_type](const std::vector<std::string>& trees,
            const double approx_factor,
            const int approx_offset,
            const py::object& log_level) {
            return solve_impl(
                trees,
                approx_factor,
                approx_offset,
                parse_log_level(log_level.ptr(), *log_level_type)
            );
        },
        py::arg("trees"),
        py::kw_only(),
        py::arg("approx_factor") = 1.0,
        py::arg("approx_offset") = 0,
        py::arg("log_level") = py::int_(0)
    );
    m.def(
        "solve_unrooted",
        [log_level_type](const std::vector<std::string>& trees,
            const py::object& log_level) {
            maffe::experimental::UnrootedOptions options;
            options.log_level = parse_log_level(log_level.ptr(), *log_level_type);
            py::gil_scoped_release release;
            return maffe::experimental::solve_unrooted(trees, options).partition;
        },
        py::arg("trees"),
        py::kw_only(),
        py::arg("log_level") = py::int_(0)
    );
    m.def(
        "rspr_dist",
        [log_level_type](const std::string& first,
            const std::string& second,
            const double approx_factor,
            const int approx_offset,
            const py::object& log_level) {
            return rspr_dist_impl(
                first,
                second,
                approx_factor,
                approx_offset,
                parse_log_level(log_level.ptr(), *log_level_type)
            );
        },
        py::arg("first"),
        py::arg("second"),
        py::kw_only(),
        py::arg("approx_factor") = 1.0,
        py::arg("approx_offset") = 0,
        py::arg("log_level") = py::int_(0)
    );
    m.def(
        "restrict_tree",
        &restrict_tree_impl,
        py::arg("trees"),
        py::arg("leaves"),
        py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "restrict_solution",
        &restrict_solution_impl,
        py::arg("trees"),
        py::arg("partition"),
        py::call_guard<py::gil_scoped_release>()
    );
}
