// The runtime: the C++ module, behind the operations the worker offers the
// page. Every call into C++ goes through here.

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

  return {
    preflight: async (bytes, totalSize) => {
      const limits = await deviceLimits;
      return withBytesInModule(module, bytes, (pointer, length) =>
        callModule((call) => module._bllm_preflight(call, pointer, length, totalSize,
          limits?.maxBufferSize ?? 0, limits?.maxStorageBufferBindingSize ?? 0,
          limits?.minStorageBufferOffsetAlignment ?? 0)));
    },

    loadChunk: () => {
      throw new Error('loading a model onto the GPU is not implemented in this build');
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
