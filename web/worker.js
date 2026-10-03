// Owns the runtime and answers the page's requests. Runs off the main thread
// so the page stays responsive.
//
// Contract 11, the boundary: this file and src/wasm/bindings.cpp are the only
// two places JavaScript and C++ meet. The crossings are preflight a header
// prefix; begin a load, load a chunk of the file, and finish the load
// (load.js); generate from a rendered prompt and the turn's policy; and
// cancel. Text comes back one message per token (WASM.2).
//
// A plain Web Worker, deliberately: it needs no SharedArrayBuffer, so it works
// on GitHub Pages, which cannot set the COOP/COEP headers that cross-origin
// isolation requires.

import { Notice, Reply, Request } from './protocol.js';

const flags = new URLSearchParams(self.location.search);
const postDevice = (device) => self.postMessage({ kind: Notice.DEVICE, device });

// ?fake-runtime selects the development stand-in, which is absent from the
// deployed site; asking for it there fails at this import, by name.
const runtimeModule = flags.has('fake-runtime') ? './dev/fake_runtime.js' : './wasm_runtime.js';
const runtimePromise = import(runtimeModule)
  .then(({ createRuntime }) => createRuntime({ onDevice: postDevice, runBench: flags.has('bench') }));
runtimePromise.catch((error) =>
  postDevice({ ok: false, stage: 'runtime', error: String(error?.message ?? error) }));

// One handler per request kind. Each receives the runtime, the request, and a
// function that streams text back; it returns the request's result or throws.
const handlers = {
  [Request.PREFLIGHT]: (runtime, { bytes, totalSize }) => runtime.preflight(bytes, totalSize),
  [Request.LOAD_BEGIN]: (runtime, { prefix, totalSize, confirmed, maxChunk }) =>
    runtime.loadBegin({ prefix, totalSize, confirmed, maxChunk }),
  [Request.LOAD_CHUNK]: (runtime, { offset, bytes }) => runtime.loadChunk({ offset, bytes }),
  [Request.LOAD_FINISH]: (runtime) => runtime.loadFinish(),
  [Request.GENERATE]: (runtime, { id, prompt, sampling, seed }, streamText) =>
    runtime.generate({ id, prompt, sampling, seed, onText: streamText }),
  [Request.CANCEL]: (runtime, { target }) => runtime.cancel(target),
};

self.addEventListener('message', async ({ data: request }) => {
  const reply = (message) => self.postMessage({ id: request.id, ...message });
  const handler = handlers[request.kind];
  if (handler === undefined) {
    reply({ kind: Reply.FAILED, error: { stage: 'protocol', message: `unknown request "${request.kind}"` } });
    return;
  }
  try {
    const runtime = await runtimePromise;
    const streamText = (text) => reply({ kind: Reply.TOKEN, text });
    reply({ kind: Reply.DONE, value: await handler(runtime, request, streamText) });
  } catch (error) {
    reply({ kind: Reply.FAILED, error: { stage: request.kind, message: String(error?.message ?? error) } });
  }
});
