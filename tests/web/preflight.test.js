import { test } from 'node:test';
import assert from 'node:assert/strict';

import { FIRST_FETCH_BYTES, MAX_INDEX_BYTES, PreflightError, preflight } from '../../web/preflight.js';

// A file of `totalSize` bytes whose index ends at `indexEnd`, served by a
// fake fetchRange that records every range asked for, and read by a fake
// reader that behaves like the real one: it reads, or asks for `indexEnd`.
function fakeFile({ totalSize, indexEnd }) {
  const fetched = [];
  return {
    fetched,
    fetchRange: async (start, end) => {
      fetched.push([start, end]);
      return { bytes: new ArrayBuffer(Math.min(end, totalSize) - start), totalSize };
    },
    readIndex: async (bytes) => (bytes.length >= indexEnd
      ? { status: 'read', accepted: false, rejections: [] }
      : { status: 'need-bytes', bytesNeeded: indexEnd }),
  };
}

test('a small index is read from the first fetch alone', async () => {
  const file = fakeFile({ totalSize: 500e6, indexEnd: 6e6 });
  const verdict = await preflight(file);
  assert.equal(verdict.status, 'read');
  assert.equal(verdict.totalSize, 500e6);
  assert.deepEqual(file.fetched, [[0, FIRST_FETCH_BYTES]]);
});

test('a larger index fetches only the bytes it lacks, contiguously', async () => {
  const file = fakeFile({ totalSize: 500e6, indexEnd: 20e6 });
  await preflight(file);
  assert.equal(file.fetched[0][0], 0);
  for (let i = 1; i < file.fetched.length; i++) {
    assert.equal(file.fetched[i][0], file.fetched[i - 1][1], 'each fetch starts where the last ended');
  }
});

test('a file smaller than the first fetch is read whole, without asking past its end', async () => {
  const file = fakeFile({ totalSize: 1000, indexEnd: 900 });
  await preflight(file);
  assert.deepEqual(file.fetched, [[0, FIRST_FETCH_BYTES]]);
});

test('an index larger than the limit is refused by name', async () => {
  const file = fakeFile({ totalSize: 500e6, indexEnd: MAX_INDEX_BYTES + 1 });
  await assert.rejects(preflight(file), (error) =>
    error instanceof PreflightError && /index is larger than/.test(error.message));
});

test('an unreadable file is reported with the reader\'s reason', async () => {
  await assert.rejects(preflight({
    fetchRange: async () => ({ bytes: new ArrayBuffer(4), totalSize: 4 }),
    readIndex: async () => ({ status: 'unreadable', error: 'not a GGUF file: magic mismatch' }),
  }), /magic mismatch/);
});

test('a server that sends nothing ends the check instead of looping', async () => {
  await assert.rejects(preflight({
    fetchRange: async () => ({ bytes: new ArrayBuffer(0), totalSize: 100 }),
    readIndex: async () => assert.fail('nothing to read'),
  }), /sent no bytes/);
});
