# SPDX-License-Identifier: CC-BY-4.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# allow linking with vortex_ffi_shared although it's not in current folder
cmake_policy(SET CMP0079 NEW)

foreach(name scan scan_to_arrow dtype write_sample parallel_write)
    add_executable(${name} ${name}.c)
    target_link_libraries(${name} PRIVATE vortex_ffi_shared nanoarrow_shared)
endforeach()
