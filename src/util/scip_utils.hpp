#pragma once

#include <memory>
#include <stdexcept>

#include "scip/scip_general.h"
#include "scip/scip_message.h"
#include "scip/pub_message.h"

namespace maffe::scip_utils {

inline void check(const SCIP_RETCODE status) {
    if (status != SCIP_OKAY)
        throw std::runtime_error("SCIP call failed");
}

struct Deleter {
    void operator()(SCIP* scip) const noexcept {
        if (scip != nullptr)
            (void)SCIPfree(&scip);
    }
};

using Ptr = std::unique_ptr<SCIP, Deleter>;

[[nodiscard]] inline Ptr create() {
    SCIP* scip = nullptr;
    check(SCIPcreate(&scip));
    return Ptr(scip);
}

inline void install_message_handler(
    SCIP* scip,
    const bool enabled,
    SCIP_DECL_MESSAGEWARNING((*print))
) {
    SCIP_MESSAGEHDLR* handler = nullptr;
    check(SCIPmessagehdlrCreate(&handler, TRUE, nullptr, !enabled, print, print, print, nullptr, nullptr));
    const auto status = SCIPsetMessagehdlr(scip, handler);
    check(SCIPmessagehdlrRelease(&handler));
    check(status);
}

} // namespace maffe::scip_utils
