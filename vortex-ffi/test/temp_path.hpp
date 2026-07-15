// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <filesystem>
#include <random>

namespace fs = std::filesystem;

struct TempPath : fs::path {
    TempPath() = default;
    explicit TempPath(fs::path p) : fs::path(std::move(p)) {
    }

    TempPath(const TempPath &) = delete;
    TempPath &operator=(const TempPath &) = delete;

    TempPath(TempPath &&other) noexcept : fs::path(std::move(other)) {
    }
    TempPath &operator=(TempPath &&other) noexcept {
        if (this != &other) {
            fs::remove(*this);
            fs::path::operator=(std::move(other));
        }
        return *this;
    }

    ~TempPath() {
        if (!empty()) {
            fs::remove(*this);
        }
    }
};

inline TempPath temp_vortex_path() {
    return TempPath {fs::temp_directory_path() /
                     fs::path("sink-test-" + std::to_string(std::random_device {}()) + ".vortex")};
}
