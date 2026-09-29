import { test } from 'node:test';
import assert from 'node:assert/strict';

import { loadModel } from '../../web/load.js';

test('the file is handed over in order, in chunks that cover it exactly', async () => {
  const bytes = Uint8Array.from({ length: 1000 }, (_, i) => i % 251);
  const file = new Blob([bytes]);
  const received = [];
  await loadModel({ file, chunkBytes: 300,
    send: async ({ offset, bytes: chunk, totalSize }) => {
      assert.equal(totalSize, 1000);
      received.push([offset, new Uint8Array(chunk)]);
    } });
  assert.deepEqual(received.map(([offset]) => offset), [0, 300, 600, 900]);
  const joined = new Uint8Array(1000);
  for (const [offset, chunk] of received) joined.set(chunk, offset);
  assert.deepEqual(joined, bytes);
});

test('a chunk is not sent until the runtime has taken the one before', async () => {
  let inFlight = 0;
  let most = 0;
  await loadModel({ file: new Blob([new Uint8Array(1000)]), chunkBytes: 100,
    send: async () => {
      most = Math.max(most, ++inFlight);
      await new Promise((resolve) => setTimeout(resolve, 1));
      inFlight--;
    } });
  assert.equal(most, 1);
});

test('a runtime that refuses a chunk stops the load with its reason', async () => {
  await assert.rejects(loadModel({ file: new Blob([new Uint8Array(10)]),
    send: async () => { throw new Error('loading is not implemented'); } }), /not implemented/);
});
