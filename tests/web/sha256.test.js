import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash, randomBytes } from 'node:crypto';

import { Sha256 } from '../../web/sha256.js';

const reference = (bytes) => createHash('sha256').update(bytes).digest('hex');

test('matches the standard test vectors', () => {
  assert.equal(new Sha256().hex(), 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
  assert.equal(new Sha256().update(new TextEncoder().encode('abc')).hex(),
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
});

test('matches Node for every length around the padding boundaries', () => {
  for (let length = 0; length <= 200; length++) {
    const bytes = randomBytes(length);
    assert.equal(new Sha256().update(bytes).hex(), reference(bytes), `length ${length}`);
  }
});

test('the digest does not depend on how the input is split', () => {
  const bytes = randomBytes(100_000);
  const whole = reference(bytes);
  for (const chunk of [1, 7, 63, 64, 65, 4096, 99_999]) {
    const hash = new Sha256();
    for (let i = 0; i < bytes.length; i += chunk) hash.update(bytes.subarray(i, i + chunk));
    assert.equal(hash.hex(), whole, `chunks of ${chunk}`);
  }
});
