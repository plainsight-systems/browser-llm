#!/bin/sh
# Removes a CMake build directory that was configured somewhere other than
# where the containerized build will find it.
#
# The problem this solves: the wasm build runs inside the emsdk image with the
# repo mounted at /src, so its cache records /src as the source directory. The
# same build directory configured by a host toolchain records the repo's real
# absolute path instead. CMake stores those paths absolutely and refuses a
# cache that disagrees, so whichever environment ran last locks the other out.
# The error it produces names neither the cause nor the remedy:
#
#   CMake Error: The source "/src/CMakeLists.txt" does not match the source
#   "/Users/.../CMakeLists.txt" used to generate cache.
#
# Removing the stale cache is safe in a way that deleting data is not: a build
# directory is derived, gitignored, and reproducible by the command that is
# about to run anyway. The cost of being wrong is a rebuild.
#
# It is not silent. Reconfiguring is reported, because a build that quietly
# discards state teaches you to ignore it.
set -eu

usage() {
    echo "usage: $0 <build-dir> <expected-source-dir>" >&2
    exit 2
}

[ $# -eq 2 ] || usage

BUILD_DIR="$1"
EXPECTED_SOURCE="$2"
CACHE="${BUILD_DIR}/CMakeCache.txt"

# Never remove a directory that is not a CMake build directory. This is the
# guard's own safety property: without it, a mistyped argument is an rm -rf on
# whatever was named.
if [ ! -f "${CACHE}" ]; then
    exit 0
fi

# Whole-line match, not a substring. "/src" must not accept "/src2".
if grep -qx "CMAKE_HOME_DIRECTORY:INTERNAL=${EXPECTED_SOURCE}" "${CACHE}"; then
    exit 0
fi

FOUND="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${CACHE}" | head -1)"
echo "==> ${BUILD_DIR} was configured for '${FOUND:-<unknown>}', not '${EXPECTED_SOURCE}'."
echo "==> Removing it and reconfiguring; nothing but derived build output is lost."
rm -rf "${BUILD_DIR}"
