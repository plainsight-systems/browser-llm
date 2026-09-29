# Thin task wrapper. Every target is a one-liner you could run by hand.
#
# On the wasm targets, local and CI build the same thing with the same preset
# and the same pinned toolchain image. They differ trivially in who supplies
# the container: CI's job already runs inside the emsdk image and invokes
# cmake directly, while these targets wrap the identical commands in
# `docker run` to put the same image around them locally.
#
# Two divergences that are NOT trivial, recorded so the line above is not read
# more broadly than it is meant:
#
#   - `make test` configures native-debug; CI (test.yml) tests native-release.
#     Different optimisation levels, so the two can genuinely disagree.
#   - `make check` runs six guards; CI runs three, split across test.yml and
#     pages.yml. The rest are local-only.
#
# Both are stated rather than fixed. Closing them is a CI change, not a
# comment.

EMSDK_IMAGE := emscripten/emsdk:6.0.8
# Where the repo is mounted inside the image. Fixed rather than $(CURDIR)
# on purpose: a constant mount point keeps artifacts identical across
# machines, where the host's own path would bake a home directory into the
# build. The cost is that a host-toolchain build configured in the same
# directory is incompatible -- see tools/ensure_container_cache.sh.
CONTAINER_SRC := /src

.PHONY: test test-native test-web check wasm wasm-diag dist serve serve-dev clean

## Every unit test. No browser, no GPU.
test: test-native test-web

## C++ build and unit tests.
test-native:
	cmake --preset native-debug
	cmake --build --preset native-debug
	ctest --preset native-debug

## JavaScript unit tests, in Node. No dependencies to install.
test-web:
	node --test tests/web/*.test.js

## Structural invariants, plus the tests proving each guard actually fires.
check:
	./tools/check_boundaries.sh
	./tools/check_diagnostics_excluded.sh
	./tests/test_check_boundaries.sh
	./tests/test_codex_review_preflight.sh
	./tests/test_ensure_container_cache.sh
	./tests/test_check_site.sh

## Diagnostic wasm build: same optimisation, instrumentation compiled in.
## Timings from this build are diagnostic and are not quotable as throughput.
wasm-diag:
	./tools/ensure_container_cache.sh build/wasm-diag $(CONTAINER_SRC)
	docker run --rm -v "$(CURDIR)":$(CONTAINER_SRC) -w $(CONTAINER_SRC) $(EMSDK_IMAGE) \
		sh -c "emcmake cmake --preset wasm-diag && cmake --build --preset wasm-diag"

## WebAssembly build, inside the pinned toolchain image.
wasm:
	./tools/ensure_container_cache.sh build/wasm-release $(CONTAINER_SRC)
	docker run --rm -v "$(CURDIR)":$(CONTAINER_SRC) -w $(CONTAINER_SRC) $(EMSDK_IMAGE) \
		sh -c "emcmake cmake --preset wasm-release && cmake --build --preset wasm-release"

## Assemble the deployable static site.
dist: wasm
	./tools/assemble_site.sh
	./tools/check_site.sh

## Serve dist/ locally. The page cannot run from file:// — module loading and
## the model fetch both fail against a null origin.
serve: dist
	cd dist && python3 -m http.server 8080

## Serve a development site that includes web/dev/, for ?fake-runtime.
## Never deployed.
serve-dev: wasm
	./tools/assemble_site.sh --dev
	cd dist-dev && python3 -m http.server 8081

clean:
	rm -rf build dist dist-dev
