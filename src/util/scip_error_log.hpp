#pragma once

#include "util/log.hpp"

#include <cstdio>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "scip/pub_message.h"
#include "scip/type_stat.h"

namespace maffe::scip_error_log {

template <class... Ts>
constexpr void ignore_unused(const Ts&...) noexcept {}

inline void log_prefixed_lines(const std::string_view prefix, std::string_view remaining) {
    while (!remaining.empty()) {
        const std::size_t newline = remaining.find_first_of("\r\n");
        std::string_view line = newline == std::string_view::npos
            ? remaining
            : remaining.substr(0, newline);
        if (!line.empty())
            logging::line(prefix, line);
        if (newline == std::string_view::npos)
            break;
        remaining.remove_prefix(newline + 1);
        while (!remaining.empty() && (remaining.front() == '\n' || remaining.front() == '\r'))
            remaining.remove_prefix(1);
    }
}

inline void log_prefixed_lines(const std::string_view prefix, const char* const msg) {
    if (msg != nullptr)
        log_prefixed_lines(prefix, std::string_view(msg));
}

[[nodiscard]] inline std::string_view status_name(const SCIP_STATUS status) {
    switch (status) {
    case SCIP_STATUS_UNKNOWN:
        return "unknown";
    case SCIP_STATUS_USERINTERRUPT:
        return "userinterrupt";
    case SCIP_STATUS_NODELIMIT:
        return "nodelimit";
    case SCIP_STATUS_TOTALNODELIMIT:
        return "totalnodelimit";
    case SCIP_STATUS_STALLNODELIMIT:
        return "stallnodelimit";
    case SCIP_STATUS_TIMELIMIT:
        return "timelimit";
    case SCIP_STATUS_MEMLIMIT:
        return "memlimit";
    case SCIP_STATUS_PRIMALLIMIT:
        return "primallimit";
    case SCIP_STATUS_DUALLIMIT:
        return "duallimit";
    case SCIP_STATUS_GAPLIMIT:
        return "gaplimit";
    case SCIP_STATUS_SOLLIMIT:
        return "sollimit";
    case SCIP_STATUS_BESTSOLLIMIT:
        return "bestsollimit";
    case SCIP_STATUS_RESTARTLIMIT:
        return "restartlimit";
    case SCIP_STATUS_OPTIMAL:
        return "optimal";
    case SCIP_STATUS_INFEASIBLE:
        return "infeasible";
    case SCIP_STATUS_UNBOUNDED:
        return "unbounded";
    case SCIP_STATUS_INFORUNBD:
        return "inforunbd";
    case SCIP_STATUS_TERMINATE:
        return "terminate";
    }
    return "invalid";
}

namespace detail {

// One definition avoids duplicate TLS initializers with MinGW.
extern thread_local std::vector<std::string_view> prefix_stack;

inline SCIP_DECL_ERRORPRINTING(error_printing) {
    (void)data;
    (void)file;
    const std::string_view prefix = prefix_stack.empty() ? "scip: " : prefix_stack.back();
    log_prefixed_lines(prefix, msg);
}

} // namespace detail

class ScopedPrefix {
public:
    explicit ScopedPrefix(std::string prefix) : prefix_(std::move(prefix)) {
        detail::prefix_stack.push_back(prefix_);
        SCIPmessageSetErrorPrinting(detail::error_printing, nullptr);
    }

    ScopedPrefix(const ScopedPrefix&) = delete;
    ScopedPrefix& operator=(const ScopedPrefix&) = delete;

    ~ScopedPrefix() {
        detail::prefix_stack.pop_back();
        if (detail::prefix_stack.empty())
            SCIPmessageSetErrorPrintingDefault();
    }

private:
    std::string prefix_;
};

} // namespace maffe::scip_error_log
