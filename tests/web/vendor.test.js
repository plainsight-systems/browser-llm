import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

// Third-party files are served exactly as published. A change to one is a
// dependency change, and must change the manifest with it.
test('every vendored file matches the hash its manifest records', async () => {
  const directory = new URL('../../web/vendor/', import.meta.url);
  const manifest = JSON.parse(await readFile(new URL('manifest.json', directory)));
  for (const [file, entry] of Object.entries(manifest)) {
    if (file.startsWith('$')) continue;
    const hash = createHash('sha256').update(await readFile(new URL(file, directory))).digest('hex');
    assert.equal(hash, entry.sha256, file);
  }
});
