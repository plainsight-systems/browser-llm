# Thin task wrapper. Every target is a one-liner you could run by hand; this
# exists so local and CI invoke identical commands.

EMSDK_IMAGE := emscripten/emsdk:6.0.8

.PHONY: test check wasm dist serve clean

## Native build + unit tests. No browser, no GPU.
test:
	cmake --preset native-debug
	cmake --build --preset native-debug
	ctest --preset native-debug

## Structural invariants, plus the tests proving each guard actually fires.
check:
	./tools/check_boundaries.sh
	./tools/check_diagrams.py
	./tests/test_check_boundaries.sh
	./tests/test_codex_review_preflight.sh

## WebAssembly build, inside the pinned toolchain image.
wasm:
	docker run --rm -v "$(CURDIR)":/src -w /src $(EMSDK_IMAGE) \
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
