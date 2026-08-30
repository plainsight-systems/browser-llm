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
  module._bllm_run_self_check();
} catch (error) {
  self.postMessage({
    type: 'result',
    result: { ok: false, stage: 'module', error: String(error?.message ?? error) },
  });
}
