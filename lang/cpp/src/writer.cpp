// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/array.hpp"
#include "vortex/common.hpp"
#include "vortex/dtype.hpp"
#include "vortex/error.hpp"
#include "vortex/writer.hpp"
#include "vortex/session.hpp"

#include <vortex.h>

#include <string_view>

namespace vortex {

using detail::Access;
using detail::throw_on_error;
using detail::to_view;

void Writer::Deleter::operator()(vx_writer *ptr) const noexcept {
    vx_writer_free(ptr);
}

Writer Writer::open(const Session &session,
                    std::string_view path,
                    const DataType &dtype,
                    size_t concurrent_array_limit) {
    vx_error *error = nullptr;
    vx_writer *sink = vx_writer_open(Access::c_ptr(session),
                                     to_view(path),
                                     Access::c_ptr(dtype),
                                     concurrent_array_limit,
                                     &error);
    throw_on_error(error);
    return Writer(sink);
}

void Writer::push(const Array &array) {
    if (handle_ == nullptr) {
        throw VortexException("null handle_", ErrorCode::InvalidArgument);
    }
    vx_error *error = nullptr;
    vx_writer_push(handle_.get(), Access::c_ptr(array), &error);
    throw_on_error(error);
}

void Writer::finish() {
    if (handle_ == nullptr) {
        throw VortexException("finish() called twice", ErrorCode::InvalidArgument);
    }
    vx_error *error = nullptr;
    vx_writer_close(handle_.get(), &error);
    handle_.reset();
    throw_on_error(error);
}
} // namespace vortex
