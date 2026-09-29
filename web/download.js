// Axis L: how we download. Streams a model file into the cache, hashing it
// as it arrives, and publishes it only when it is complete and matches.

import { Sha256 } from './sha256.js';
import { formatBytes } from './units.js';

export class DownloadError extends Error {
  constructor(message) {
    super(message);
    this.name = 'DownloadError';
  }
}

// Where a model lives in the cache. A curated model is named by its hash, so
// the same file is cached once whichever URL served it; a pasted URL is
// named by the URL.
export function cacheKey(model) {
  if (model.sha256 !== undefined) return `sha256-${model.sha256}`;
  return `url-${new Sha256().update(new TextEncoder().encode(model.url)).hex().slice(0, 32)}`;
}

// `openDownload(url, { signal })` resolves with { totalSize, chunks };
// `cache.writer(key)` with a writer. `onProgress(received, total)` follows the
// download; `total` is null when neither the list nor the server says.
export async function downloadModel({ model, openDownload, cache, onProgress, signal }) {
  const { totalSize, chunks } = await openDownload(model.url, { signal });
  if (model.sizeBytes !== undefined && totalSize !== null && totalSize !== model.sizeBytes) {
    throw new DownloadError(`the server's file is ${formatBytes(totalSize)}, ` +
                            `but the model list says ${formatBytes(model.sizeBytes)}`);
  }
  const expected = model.sizeBytes ?? totalSize;

  const writer = await cache.writer(cacheKey(model));
  try {
    const hash = new Sha256();
    let received = 0;
    for await (const chunk of chunks) {
      signal?.throwIfAborted();
      hash.update(chunk);
      await writer.write(chunk);
      received += chunk.length;
      onProgress?.(received, expected);
    }
    if (expected !== null && received !== expected) {
      throw new DownloadError(`the download ended after ${formatBytes(received)} of ${formatBytes(expected)}`);
    }
    const sha256 = hash.hex();
    if (model.sha256 !== undefined && sha256 !== model.sha256) {
      throw new DownloadError('the downloaded file does not match its published SHA-256');
    }
    await writer.commit({ name: model.name, url: model.url, sizeBytes: received, sha256 });
  } catch (error) {
    await writer.discard();
    throw error;
  }
}
