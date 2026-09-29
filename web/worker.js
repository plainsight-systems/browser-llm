// Owns the wasm module and the inference loop. Runs off the main thread so the
// page stays responsive.
//
// Contract 11, the boundary: this file and src/wasm/bindings.cpp are the only
// two places JavaScript and C++ meet. The crossings are preflight a header
// prefix, load a chunk of the file, generate from a rendered prompt and the
// turn's policy, and cancel. Text comes back one message per token (WASM.2).
//
// A plain Web Worker, deliberately: it needs no SharedArrayBuffer, so it works
// on GitHub Pages, which cannot set the COOP/COEP headers that cross-origin
// isolation requires.

import createModule from './browser_llm.mjs';
import { Notice, Reply, Request } from './protocol.js';

const modulePromise = createModule();

// --- Calls into C++ ---------------------------------------------------------

// C++ answers a call through bllmOnReply, either during the call or later from
// a callback. Each call gets an id so its answer finds it either way.
const pendingCalls = new Map();
let nextCall = 1;
globalThis.bllmOnReply = (call, value) => {
  pendingCalls.get(call)(value);
  pendingCalls.delete(call);
};

function callModule(start) {
  const call = nextCall++;
  return new Promise((resolve) => {
    pendingCalls.set(call, resolve);
    start(call);
  });
}

// Copies `bytes` into the module's memory for the duration of `use`.
async function withBytesInModule(module, bytes, use) {
  const pointer = module._malloc(bytes.byteLength);
  if (pointer === 0) throw new Error(`could not allocate ${bytes.byteLength} bytes in the module`);
  try {
    module.HEAPU8.set(new Uint8Array(bytes), pointer);
    return await use(pointer, bytes.byteLength);
  } finally {
    module._free(pointer);
  }
}

// --- Requests from the page -------------------------------------------------

// One handler per request kind. Each receives the request and a function that
// streams text back, and returns the request's result or throws.
const handlers = {
  async [Request.PREFLIGHT]({ bytes, totalSize }) {
    const module = await modulePromise;
    return withBytesInModule(module, bytes, (pointer, length) =>
      callModule((call) => module._bllm_preflight(call, pointer, length, totalSize)));
  },
};

const fail = (id, stage, message) =>
  self.postMessage({ id, kind: Reply.FAILED, error: { stage, message } });

self.addEventListener('message', async ({ data: request }) => {
  const handler = handlers[request.kind];
  if (handler === undefined) {
    fail(request.id, 'protocol', `unknown request "${request.kind}"`);
    return;
  }
  const streamText = (text) =>
    self.postMessage({ id: request.id, kind: Reply.TOKEN, text });
  try {
    const value = await handler(request, streamText);
    self.postMessage({ id: request.id, kind: Reply.DONE, value });
  } catch (error) {
    fail(request.id, request.kind, String(error?.message ?? error));
  }
});

// --- The device check at startup --------------------------------------------

const postDevice = (device) => self.postMessage({ kind: Notice.DEVICE, device });

// The module reports the device check by calling this by name.
globalThis.bllmOnResult = postDevice;

try {
  const module = await modulePromise;
  // ?bench runs the readback measurement — present only in a diagnostic
  // build, so its absence is reported rather than failing obscurely.
  if (!self.location.search.includes('bench')) {
    module._bllm_run_self_check();
  } else if (typeof module._bllm_run_readback_bench === 'function') {
    module._bllm_run_readback_bench();
  } else {
    postDevice({
      ok: false,
      stage: 'request',
      error: 'the readback benchmark is not compiled into this build; ' +
             'configure the wasm-diag preset to run it',
    });
  }
} catch (error) {
  postDevice({ ok: false, stage: 'module', error: String(error?.message ?? error) });
}
