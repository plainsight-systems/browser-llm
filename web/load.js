// Axis L. Reads a cached model file in chunks and hands each to the runtime,
// in order, one at a time — so the page never holds more than one chunk.
//
// A load is three crossings into the module (WASM.2: a phase per crossing):
//
//   1. begin: the index prefix preflight read, the file's size, and the
//      duplicates confirmDuplicates confirmed (duplicates.js). The module
//      plans the routes and creates the buffers, and answers once the device
//      has confirmed it holds them, or names what it could not hold.
//   2. a chunk, LOAD_CHUNK_BYTES at a time, in order: copied straight into a
//      chunk buffer the module allocated once at begin, by pointer and length.
//      The view over the heap is made afresh for each chunk and never kept,
//      since growing the heap detaches it (WASM.1). The module answers when it
//      may take the next — after the queue has finished the chunk before, so
//      reading chunk n + 1 from the cache overlaps the GPU copying chunk n
//      (src/core/residency/upload.h).
//   3. finish: answered once every write has completed, or with the failure
//      that stopped them.
//
// Optimization (browser): each chunk is read from the cache into one
// ArrayBuffer and copied once into the heap; it is never held twice, and the
// heap does not grow with the file (WASM.1, WASM.9).
//
// Guidelines (the C++ performance guidelines; their WASM and GPU entries
// govern the page as much as the module):
//   WASM.1 Size linear memory to the real high-water mark; never hold a view
//          across a call that can grow the heap — one chunk buffer, a fresh
//          view per chunk.
//   WASM.2 Batch work across the JS boundary — three kinds of crossing, one
//          chunk per crossing, by pointer and length.
//   WASM.9 Stream in bounded chunks — the page holds one chunk.
//   GPU.7  Pipeline CPU and GPU work — reading the next chunk overlaps the
//          GPU copying the last.

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
