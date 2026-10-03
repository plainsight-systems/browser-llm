// The messages that cross between the page and the worker. Both sides import
// this file, so every kind is spelled once.
//
// A request carries an id. Every reply to it carries the same id, and the
// replies end with exactly one DONE or FAILED; TOKEN replies may come before.
// Notices carry no id: the worker sends them unprompted.

export const Request = Object.freeze({
  // Read a model file's index from its first bytes and judge it.
  PREFLIGHT: 'preflight',
  // Begin loading a model: its index prefix, size and confirmed duplicates.
  LOAD_BEGIN: 'load-begin',
  // Hand the runtime the next chunk of the model file, in order.
  LOAD_CHUNK: 'load-chunk',
  // End the load: answered once every write is confirmed on the device.
  LOAD_FINISH: 'load-finish',
  // A diagnostic build's check of the loaded model, the same three steps;
  // a build without it answers CHECK_BEGIN with { skipped: true }.
  CHECK_BEGIN: 'check-begin',
  CHECK_CHUNK: 'check-chunk',
  CHECK_FINISH: 'check-finish',
  // Generate a reply to a rendered prompt, streaming its text.
  GENERATE: 'generate',
  // Stop a generation early; `target` is its request id.
  CANCEL: 'cancel',
});

export const Reply = Object.freeze({
  TOKEN: 'token',
  DONE: 'done',
  FAILED: 'failed',
});

export const Notice = Object.freeze({
  // The GPU device was acquired and checked, or could not be.
  DEVICE: 'device',
});
