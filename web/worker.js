// Owns the wasm module and the inference loop. Runs off the main thread so the
// page stays responsive.
//
// A plain Web Worker, deliberately: it needs no SharedArrayBuffer, so it works
// on GitHub Pages, which cannot set the COOP/COEP headers that cross-origin
// isolation requires.

import createModule from './browser_llm.mjs';

// The wasm module calls this by name when the self-check resolves.
globalThis.bllmOnResult = (result) => {
  self.postMessage({ type: 'result', result });
};

try {
  const module = await createModule();
  self.postMessage({ type: 'ready' });
  // ?bench runs the readback measurement spike — present only in a diagnostic
  // build. The clean build does not compile it, so the export is absent and we
  // say so rather than failing obscurely.
  if (!self.location.search.includes('bench')) {
    module._bllm_run_self_check();
  } else if (typeof module._bllm_run_readback_bench === 'function') {
    module._bllm_run_readback_bench();
  } else {
    // No early return here: this is module top level, not a function body.
    self.postMessage({
      type: 'result',
      result: {
        ok: false,
        stage: 'request',
        error: 'the readback benchmark is not compiled into this build; ' +
               'configure the wasm-diag preset to run it',
      },
    });
  }
} catch (error) {
  self.postMessage({
    type: 'result',
    result: { ok: false, stage: 'module', error: String(error?.message ?? error) },
  });
}
