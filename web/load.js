// Axis L. Reads a cached model file in chunks and hands each to the runtime,
// in order, one at a time — so the page never holds more than one chunk.

export const LOAD_CHUNK_BYTES = 16 * 2 ** 20;

// `send({ offset, bytes, totalSize })` delivers one chunk and resolves when
// the runtime has taken it. Resolves with the last chunk's answer.
export async function loadModel({ file, send, onProgress, signal, chunkBytes = LOAD_CHUNK_BYTES }) {
  let answer;
  for (let offset = 0; offset < file.size; offset += chunkBytes) {
    signal?.throwIfAborted();
    const bytes = await file.slice(offset, offset + chunkBytes).arrayBuffer();
    answer = await send({ offset, bytes, totalSize: file.size });
    onProgress?.(Math.min(offset + chunkBytes, file.size), file.size);
  }
  return answer;
}
