// The messages that cross between the page and the worker. Both sides import
// this file, so every kind is spelled once.
//
// A request carries an id. Every reply to it carries the same id, and the
// replies end with exactly one DONE or FAILED; TOKEN replies may come before.
// Notices carry no id: the worker sends them unprompted.

export const Request = Object.freeze({
  // Read a model file's index from its first bytes and judge it.
  PREFLIGHT: 'preflight',
  // Hand the runtime the next chunk of the model file, in order.
  LOAD_CHUNK: 'load-chunk',
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
