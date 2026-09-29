// Axis L: how we download. Reads a model file's index before any weight is
// downloaded.
//
// Fetches the front of the file and hands it to the reader. While the reader
// answers that it needs more, fetches only the bytes it lacks and asks again.

import { formatBytes } from './units.js';

// Enough for the index of every model the harness targets in one fetch.
export const FIRST_FETCH_BYTES = 8 * 2 ** 20;
// An index larger than this is not a model this harness can hold.
export const MAX_INDEX_BYTES = 64 * 2 ** 20;

export class PreflightError extends Error {
  constructor(message) {
    super(message);
    this.name = 'PreflightError';
  }
}

// `fetchRange(start, end)` resolves with { bytes, totalSize } for that part of
// the file. `readIndex(bytes, totalSize)` resolves with the reader's answer
// for the file's first bytes. Resolves with the reader's verdict and the
// file's size.
export async function preflight({ fetchRange, readIndex }) {
  let resident = new Uint8Array(0);
  let totalSize = Infinity;
  let wanted = FIRST_FETCH_BYTES;

  for (;;) {
    const piece = await fetchRange(resident.length, Math.min(wanted, totalSize));
    if (piece.bytes.byteLength === 0) throw new PreflightError('the server sent no bytes');
    totalSize = piece.totalSize;
    resident = concat(resident, new Uint8Array(piece.bytes));

    const answer = await readIndex(resident, totalSize);
    switch (answer.status) {
      case 'read':
        return { ...answer, totalSize };
      case 'unreadable':
        throw new PreflightError(`not a readable GGUF file: ${answer.error}`);
      case 'need-bytes':
        if (answer.bytesNeeded > MAX_INDEX_BYTES) {
          throw new PreflightError(`the file's index is larger than ${formatBytes(MAX_INDEX_BYTES)}`);
        }
        wanted = Math.max(answer.bytesNeeded, resident.length * 2);
        break;
      default:
        throw new PreflightError(`unexpected answer from the reader: ${answer.status}`);
    }
  }
}

function concat(a, b) {
  const joined = new Uint8Array(a.length + b.length);
  joined.set(a);
  joined.set(b, a.length);
  return joined;
}
