#!/bin/sh
# Proves each branch of the container-cache guard actually fires.
#
# A guard nobody has seen fail is not evidence of anything — and this one
# removes directories, so the branch that must be proven hardest is the one
# where it declines to.
set -eu

cd "$(dirname "$0")/.."
GUARD=./tools/ensure_container_cache.sh
WORK="$(mktemp -d)"

cleanup() { rm -rf "${WORK}"; }
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

make_cache_dir() {
    dir="${WORK}/$1"
    rm -rf "${dir}"; mkdir -p "${dir}"
    if [ $# -ge 2 ]; then
        printf 'CMAKE_CACHEFILE_DIR:INTERNAL=%s/build\nCMAKE_HOME_DIRECTORY:INTERNAL=%s\n' "$2" "$2" \
            > "${dir}/CMakeCache.txt"
    fi
    printf 'derived\n' > "${dir}/marker"
    echo "${dir}"
}

# 1. Cache configured for the expected source: must be left alone.
D="$(make_cache_dir matching /src)"
"${GUARD}" "${D}" /src
[ -f "${D}/marker" ] || fail "guard removed a cache configured for the expected source"

# 2. Cache configured elsewhere: must be removed. This is the case that
#    actually happened — a host toolchain build poisoning the container path.
D="$(make_cache_dir foreign /Users/someone/repo)"
"${GUARD}" "${D}" /src >/dev/null
[ -d "${D}" ] && fail "guard kept a cache configured for a different source dir"

# 3. Near-collision. A substring match would accept /src2 as /src, leaving a
#    poisoned cache in place and reproducing the original failure.
D="$(make_cache_dir near_miss /src2)"
"${GUARD}" "${D}" /src >/dev/null
[ -d "${D}" ] && fail "guard accepted /src2 as /src — match is not whole-line"

# 4. A directory with no CMakeCache.txt must never be removed. This is the
#    guard's safety property: a mistyped argument must not become an rm -rf.
D="$(make_cache_dir not_a_build_dir)"
"${GUARD}" "${D}" /src
[ -f "${D}/marker" ] || fail "guard removed a directory that is not a CMake build dir"

# 5. A missing directory is a no-op, not an error — the first build after a
#    clean checkout takes this path.
"${GUARD}" "${WORK}/does_not_exist" /src || fail "guard errored on a missing build dir"

# 6. Idempotent: running twice against a good cache changes nothing.
D="$(make_cache_dir idempotent /src)"
"${GUARD}" "${D}" /src; "${GUARD}" "${D}" /src
[ -f "${D}/marker" ] || fail "guard is not idempotent"

# 7. Wrong argument count must fail loudly rather than guess.
if "${GUARD}" "${WORK}" >/dev/null 2>&1; then
    fail "guard accepted a missing argument"
fi

echo "ensure_container_cache: OK (all seven branches fire)"
