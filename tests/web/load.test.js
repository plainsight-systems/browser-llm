import { test } from 'node:test';
import assert from 'node:assert/strict';

import { loadModel } from '../../web/load.js';

// A runtime that records every call in order.
function runtime({ refuse } = {}) {
  const calls = [];
  return {
    calls,
    begin: async ({ maxChunk }) => { calls.push(['begin', maxChunk]); },
    send: async ({ offset, bytes }) => {
      calls.push(['chunk', offset, new Uint8Array(bytes)]);
      if (refuse !== undefined && offset >= refuse) throw new Error('the device refused it');
    },
    finish: async () => { calls.push(['finish']); return 'finished'; },
  };
}

test('the file is handed over in order, in chunks that cover it exactly, between begin and finish', async () => {
  const bytes = Uint8Array.from({ length: 1000 }, (_, i) => i % 251);
  const r = runtime();
  const answer = await loadModel({ file: new Blob([bytes]), chunkBytes: 300, ...r });
  assert.equal(answer, 'finished');
  assert.deepEqual(r.calls[0], ['begin', 300]);
  assert.deepEqual(r.calls.at(-1), ['finish']);
  const chunks = r.calls.slice(1, -1);
  assert.deepEqual(chunks.map(([, offset]) => offset), [0, 300, 600, 900]);
  const joined = new Uint8Array(1000);
  for (const [, offset, chunk] of chunks) joined.set(chunk, offset);
  assert.deepEqual(joined, bytes);
});

test('a chunk is not sent until the runtime has taken the one before', async () => {
  let inFlight = 0;
  let most = 0;
  await loadModel({ ...runtime(), file: new Blob([new Uint8Array(1000)]), chunkBytes: 100,
    send: async () => {
      most = Math.max(most, ++inFlight);
      await new Promise((resolve) => setTimeout(resolve, 1));
      inFlight--;
    } });
  assert.equal(most, 1);
});

test('a runtime that refuses a chunk stops the load with its reason, and is not asked to finish', async () => {
  const r = runtime({ refuse: 200 });
  await assert.rejects(loadModel({ file: new Blob([new Uint8Array(1000)]), chunkBytes: 100, ...r }),
    /refused it/);
  assert.notDeepEqual(r.calls.at(-1), ['finish']);   // it settled the load itself
});

test('a load the page aborts is still finished, so the runtime takes the next', async () => {
  const controller = new AbortController();
  const r = runtime();
  const send = r.send;
  r.send = async (chunk) => {
    await send(chunk);
    if (chunk.offset === 200) controller.abort();
  };
  await assert.rejects(loadModel({ file: new Blob([new Uint8Array(1000)]), chunkBytes: 100,
    signal: controller.signal, ...r }), { name: 'AbortError' });
  assert.deepEqual(r.calls.at(-1), ['finish']);
});

test('a begin the runtime refuses sends nothing', async () => {
  const r = runtime();
  r.begin = async () => { throw new Error('no checked GPU device to load onto'); };
  await assert.rejects(loadModel({ file: new Blob([new Uint8Array(10)]), ...r }), /no checked GPU/);
  assert.equal(r.calls.length, 0);
});
