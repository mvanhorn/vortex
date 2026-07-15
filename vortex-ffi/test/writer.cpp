// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <stdexcept>
#include <thread>
#include <vector>

#include <vortex.h>

#include "common.h"
#include "temp_path.hpp"

using Catch::Matchers::ContainsSubstring;

namespace {

const vx_array *pattern_array(uint64_t start, uint64_t len) {
    std::vector<uint64_t> data(len);
    for (uint64_t i = 0; i < len; ++i) {
        data[i] = (start + i) % 997;
    }

    vx_validity validity = {};
    validity.type = VX_VALIDITY_NON_NULLABLE;

    vx_error *error = nullptr;
    const vx_array *array = vx_array_new_primitive(PTYPE_U64, data.data(), len, &validity, &error);
    require_no_error(error);
    return array;
}

TEST_CASE("Push after close", "[writer]") {
    vx_session *session = vx_session_new();
    defer {
        vx_session_free(session);
    };

    TempPath path = temp_vortex_path();
    const vx_dtype *dtype = vx_dtype_new_primitive(PTYPE_U64, false);
    defer {
        vx_dtype_free(dtype);
    };

    vx_error *error = nullptr;
    vx_writer *sink = vx_writer_open(session, vx_view_from_cstr(path.c_str()), dtype, 32, &error);
    require_no_error(error);
    REQUIRE(sink != nullptr);
    defer {
        vx_writer_free(sink);
    };

    const vx_array *array = pattern_array(0, 1);
    defer {
        vx_array_free(array);
    };

    vx_writer_push(sink, array, &error);
    require_no_error(error);

    vx_writer_close(sink, &error);
    require_no_error(error);

    vx_writer_push(sink, array, &error);
    REQUIRE(error != nullptr);
    REQUIRE_THAT(to_string(error), ContainsSubstring("closed"));
    vx_error_free(error);

    vx_writer_close(sink, &error);
    REQUIRE(error != nullptr);
    vx_error_free(error);
}

TEST_CASE("Concurrent push", "[writer]") {
    vx_session *session = vx_session_new();
    defer {
        vx_session_free(session);
    };

    TempPath path = temp_vortex_path();
    const vx_dtype *dtype = vx_dtype_new_primitive(PTYPE_U64, false);
    defer {
        vx_dtype_free(dtype);
    };

    vx_error *error = nullptr;
    vx_writer *sink = vx_writer_open(session, vx_view_from_cstr(path.c_str()), dtype, 32, &error);
    require_no_error(error);
    REQUIRE(sink != nullptr);
    defer {
        vx_writer_free(sink);
    };

    constexpr uint64_t NUM_THREADS = 8;
    constexpr uint64_t CHUNK_LEN = 1000;

    std::vector<std::thread> pool;
    pool.reserve(NUM_THREADS);
    for (uint64_t i = 0; i < NUM_THREADS; ++i) {
        pool.emplace_back([&, i = i] {
            const vx_array *array = pattern_array(i * CHUNK_LEN, CHUNK_LEN);
            vx_error *push_error = nullptr;
            vx_writer_push(sink, array, &push_error);
            vx_array_free(array);
            if (push_error != nullptr) {
                vx_error_free(push_error);
                throw std::runtime_error("push failed");
            }
        });
    }
    for (auto &thread : pool) {
        thread.join();
    }

    vx_writer_close(sink, &error);
    require_no_error(error);

    vx_data_source_options opts = {};
    const vx_view ds_path = vx_view_from_cstr(path.c_str());
    opts.paths = &ds_path;
    opts.paths_len = 1;

    const vx_data_source *ds = vx_data_source_new(session, &opts, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);
    defer {
        vx_data_source_free(ds);
    };

    vx_estimate row_count = {};
    vx_data_source_get_row_count(ds, &row_count);
    REQUIRE(row_count.type == VX_ESTIMATE_EXACT);
    REQUIRE(row_count.estimate == NUM_THREADS * CHUNK_LEN);
}

} // namespace
