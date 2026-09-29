import { after, before, test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';

import { FetchError, fetchRange, parseContentRange } from '../../web/fetch.js';

// A real HTTP server, so fetchRange runs against real Range semantics rather
// than a stubbed fetch.
const FILE = Buffer.from(Array.from({ length: 1000 }, (_, i) => i % 256));
let base;
let server;

before(async () => {
  server = createServer((req, res) => {
    if (req.url === '/gated') { res.writeHead(401).end(); return; }
    if (req.url === '/missing') { res.writeHead(404).end(); return; }
    if (req.url === '/no-ranges') { res.writeHead(200).end(FILE); return; }
    const [, first, last] = /bytes=(\d+)-(\d+)/.exec(req.headers.range);
    const start = Number(first);
    const end = Math.min(Number(last), FILE.length - 1);
    res.writeHead(206, { 'Content-Range': `bytes ${start}-${end}/${FILE.length}` });
    res.end(FILE.subarray(start, end + 1));
  });
  await new Promise((resolve) => server.listen(0, resolve));
  base = `http://127.0.0.1:${server.address().port}`;
});

after(() => server.close());

test('parseContentRange reads a well-formed header and rejects anything else', () => {
  assert.deepEqual(parseContentRange('bytes 0-15/382156480'), { start: 0, end: 15, total: 382156480 });
  assert.equal(parseContentRange('bytes */1000'), null);
  assert.equal(parseContentRange(null), null);
});

test('fetchRange returns exactly the bytes asked for and the file size', async () => {
  const { bytes, totalSize } = await fetchRange(`${base}/file`, 100, 110);
  assert.equal(totalSize, 1000);
  assert.deepEqual([...new Uint8Array(bytes)], [...FILE.subarray(100, 110)]);
});

test('a range past the end of the file returns what the file has', async () => {
  const { bytes } = await fetchRange(`${base}/file`, 990, 2000);
  assert.equal(bytes.byteLength, 10);
});

test('failures are named by reason', async () => {
  const reasonOf = (path) => fetchRange(`${base}${path}`, 0, 10).then(
    () => assert.fail('expected a failure'),
    (error) => { assert.ok(error instanceof FetchError); return error.reason; });
  assert.equal(await reasonOf('/gated'), 'needs-sign-in');
  assert.equal(await reasonOf('/missing'), 'not-found');
  assert.equal(await reasonOf('/no-ranges'), 'no-range-support');
});

test('an unreachable server is a network failure', async () => {
  await assert.rejects(fetchRange('http://127.0.0.1:1/file', 0, 10),
    (error) => error instanceof FetchError && error.reason === 'network');
});
