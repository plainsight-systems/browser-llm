import { test } from 'node:test';
import assert from 'node:assert/strict';

import { confirmDuplicates, equalBuffers } from '../../web/duplicates.js';

// A file of a header, a tensor, a gap, and a second tensor that is a copy of
// the first except where `differAt` changes one byte.
function file({ length = 1000, differAt = -1 } = {}) {
  const bytes = new Uint8Array(24 + length + 7 + length);
  for (let i = 0; i < length; i++) {
    bytes[24 + i] = (i * 37) % 251;
    bytes[24 + length + 7 + i] = (i * 37) % 251;
  }
  if (differAt >= 0) bytes[24 + length + 7 + differAt] ^= 0xff;
  return { blob: new Blob([bytes]), candidate: { tensor: 5, copies: 1, offset: 24 + length + 7, copiesOffset: 24, length } };
}

// Records every slice read, to see how much is held and where reading stops.
function watched(blob) {
  const reads = [];
  return {
    size: blob.size,
    reads,
    slice(start, end) {
      reads.push(end - start);
      return blob.slice(start, end);
    },
  };
}

test('a candidate whose every byte matches is confirmed', async () => {
  const { blob, candidate } = file();
  assert.deepEqual(await confirmDuplicates({ file: blob, candidates: [candidate], sliceBytes: 300 }), [5]);
});

test('one differing byte, even in the last slice, leaves a candidate unconfirmed', async () => {
  for (const differAt of [0, 299, 300, 999]) {
    const { blob, candidate } = file({ differAt });
    assert.deepEqual(await confirmDuplicates({ file: blob, candidates: [candidate], sliceBytes: 300 }), [],
      `byte ${differAt}`);
  }
});

test('slices are read in matching pairs no larger than the slice, and reading stops at the first that differs', async () => {
  const { blob, candidate } = file({ differAt: 350 });
  const f = watched(blob);
  await confirmDuplicates({ file: f, candidates: [candidate], sliceBytes: 300 });
  assert.deepEqual(f.reads, [300, 300, 300, 300]);   // the second pair differs; the rest are never read
});

test('only the candidates that match are named, in order', async () => {
  const same = file();
  const bytes = new Uint8Array(await same.blob.arrayBuffer());
  const other = { ...same.candidate, tensor: 9, offset: 0, length: 24 };   // the header against the tensor
  assert.deepEqual(await confirmDuplicates({ file: new Blob([bytes]), candidates: [other, same.candidate] }), [5]);
});

test('a candidate that reads past the file is a defect, rejected by name', async () => {
  const { blob, candidate } = file();
  await assert.rejects(confirmDuplicates({ file: blob, candidates: [{ ...candidate, length: candidate.length + 1 }] }),
    /candidate 5 reads past the end/);
});

test('an aborted comparison stops with the abort', async () => {
  const { blob, candidate } = file();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(confirmDuplicates({ file: blob, candidates: [candidate], signal: controller.signal }),
    { name: 'AbortError' });
});

test('buffers compare by every byte, word-aligned or not', () => {
  for (const n of [0, 1, 3, 4, 5, 8, 11]) {
    const a = Uint8Array.from({ length: n }, (_, i) => i + 1);
    assert.equal(equalBuffers(a.buffer, a.slice().buffer), true, `${n} bytes`);
    for (let i = 0; i < n; i++) {
      const b = a.slice();
      b[i] ^= 1;
      assert.equal(equalBuffers(a.buffer, b.buffer), false, `${n} bytes, byte ${i}`);
    }
  }
  assert.equal(equalBuffers(new ArrayBuffer(4), new ArrayBuffer(5)), false);
});
