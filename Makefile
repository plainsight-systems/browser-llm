# Thin task wrapper. Every target is a one-liner you could run by hand; this
# exists so local and CI invoke identical commands.

EMSDK_IMAGE := emscripten/emsdk:6.0.8
# Where the repo is mounted inside the image. Fixed rather than $(CURDIR)
# on purpose: a constant mount point keeps artifacts identical across
# machines, where the host's own path would bake a home directory into the
# build. The cost is that a host-toolchain build configured in the same
# directory is incompatible -- see tools/ensure_container_cache.sh.
CONTAINER_SRC := /src

.PHONY: test check wasm wasm-diag dist serve clean

## Native build + unit tests. No browser, no GPU.
test:
	cmake --preset native-debug
	cmake --build --preset native-debug
	ctest --preset native-debug

## Structural invariants, plus the tests proving each guard actually fires.
check:
	./tools/check_boundaries.sh
	./tools/check_diagrams.py
	./tools/check_diagnostics_excluded.sh
	./tests/test_check_boundaries.sh
	./tests/test_codex_review_preflight.sh
	./tests/test_ensure_container_cache.sh

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
	mkdir -p dist
	cp web/* dist/
	cp build/wasm-release/browser_llm.mjs build/wasm-release/browser_llm.wasm dist/
	touch dist/.nojekyll

## Serve dist/ locally. The page cannot run from file:// — module loading and
## the model fetch both fail against a null origin.
serve: dist
	cd dist && python3 -m http.server 8080

clean:
	rm -rf build dist
