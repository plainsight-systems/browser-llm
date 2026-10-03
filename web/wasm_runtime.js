// The runtime: the C++ module, behind the operations the worker offers the
// page. Every call into C++ goes through here.
//
// A load is three operations (src/wasm/bindings.cpp): loadBegin with the
// index prefix preflight read, the file's size and the confirmed duplicates;
// loadChunk for each chunk, in order, copied into the chunk buffer begin
// reported through a fresh view of the heap each time, since growing the
// heap detaches any view kept (WASM.1); and loadFinish. Each resolves with
// the module's answer, or rejects with the failure it names.
//
// A diagnostic module also offers the check of a finished load: checkBegin,
// checkChunk and checkFinish, the same shape, ending with every mismatch.
// `canCheck` says whether this module has them; the clean module does not.

import createModule from './charlotte.mjs';

// Starts the module and the device check, which reports through `onDevice`.
// Resolves with the runtime once the module is instantiated.
export async function createRuntime({ onDevice, runBench }) {
  // The device's granted limits, once the device check reports: what preflight
  // judges fit against. Null when no device was acquired.
  let reportLimits;
  const deviceLimits = new Promise((resolve) => { reportLimits = resolve; });

  // The module reports the device check by calling this by name.
  globalThis.bllmOnResult = (device) => {
    reportLimits(device.ok && device.limits !== undefined ? device.limits : null);
    onDevice(device);
  };
  globalThis.bllmOnReply = answer;

  const module = await createModule();
  startDeviceCheck(module, { onDevice, runBench });
  // The chunk buffer the load in progress copies into, from loadBegin.
  let load = null;
  // The same, for a check in progress; only a diagnostic module checks.
  const canCheck = typeof module._bllm_check_begin === 'function';
  let check = null;

  return {
    preflight: async (bytes, totalSize) => {
      const limits = await deviceLimits;
      return withBytesInModule(module, bytes, (pointer, length) =>
        callModule((call) => module._bllm_preflight(call, pointer, length, totalSize,
          limits?.maxBufferSize ?? 0, limits?.maxStorageBufferBindingSize ?? 0,
          limits?.minStorageBufferOffsetAlignment ?? 0)));
    },

    loadBegin: async ({ prefix, totalSize, confirmed, maxChunk }) => {
      const answer = await withBytesInModule(module, prefix, (pointer, length) =>
        withBytesInModule(module, u32s(confirmed), (ids) =>
          callModule((call) =>
            module._bllm_load_begin(call, pointer, length, totalSize, ids, confirmed.length, maxChunk))));
      load = settled(answer);
      return { chunkBytes: load.chunkBytes };
    },

    loadChunk: async ({ offset, bytes }) => {
      if (load === null) throw new Error('no load is in progress');
      // A fresh view each chunk: growing the heap detaches any view kept.
      module.HEAPU8.set(new Uint8Array(bytes), load.chunkPointer);
      settled(await callModule((call) => module._bllm_load_chunk(call, offset, bytes.byteLength)));
    },

    loadFinish: async () => {
      const answer = await callModule((call) => module._bllm_load_finish(call));
      load = null;
      settled(answer);
    },

    canCheck,

    checkBegin: async ({ maxChunk }) => {
      check = settled(await callModule((call) => module._bllm_check_begin(call, maxChunk)));
      return { chunkBytes: check.chunkBytes };
    },

    checkChunk: async ({ offset, bytes }) => {
      if (check === null) throw new Error('no check is in progress');
      module.HEAPU8.set(new Uint8Array(bytes), check.chunkPointer);
      settled(await callModule((call) => module._bllm_check_chunk(call, offset, bytes.byteLength)));
    },

    checkFinish: async () => {
      const answer = await callModule((call) => module._bllm_check_finish(call));
      check = null;
      return { mismatches: settled(answer).mismatches };
    },

    generate: () => {
      throw new Error('generating text is not implemented in this build');
    },

    // Nothing generates, so there is nothing to stop.
    cancel: () => false,
  };
}

function startDeviceCheck(module, { onDevice, runBench }) {
  if (!runBench) {
    module._bllm_run_self_check();
  } else if (typeof module._bllm_run_readback_bench === 'function') {
    // Present only in a diagnostic build.
    module._bllm_run_readback_bench();
  } else {
    onDevice({
      ok: false,
      stage: 'request',
      error: 'the readback benchmark is not compiled into this build; ' +
             'configure the wasm-diag preset to run it',
    });
  }
}

// C++ answers a call through bllmOnReply, either during the call or later from
// a callback. Each call gets an id so its answer finds it either way.
const pendingCalls = new Map();
let nextCall = 1;

function answer(call, value) {
  pendingCalls.get(call)(value);
  pendingCalls.delete(call);
}

function callModule(start) {
  const call = nextCall++;
  return new Promise((resolve) => {
    pendingCalls.set(call, resolve);
    start(call);
  });
}

// A load answer, or its failure thrown, named.
function settled(answer) {
  if (!answer.ok) throw new Error(answer.subject ? `${answer.error} (${answer.subject})` : answer.error);
  return answer;
}

// Tensor ids as little-endian 32-bit words, for the module.
function u32s(values) {
  const words = new DataView(new ArrayBuffer(Math.max(4, values.length * 4)));
  values.forEach((v, i) => words.setUint32(i * 4, v, true));
  return words.buffer;
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
