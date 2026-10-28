#include "util/scip_error_log.hpp"

namespace maffe::scip_error_log::detail {

thread_local std::vector<std::string_view> prefix_stack;

} // namespace maffe::scip_error_log::detail
