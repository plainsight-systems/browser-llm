// Axis L: how a cached model is read.
//
// Confirms which candidate duplicates really are copies, by comparing their
// bytes in the cached file, before upload begins. Preflight names each
// candidate and the tensor it may copy, with both byte ranges; a candidate is
// confirmed only if every byte matches. Upload then shares the confirmed ones
// and routes the rest like any tensor (src/core/residency/routes.h).
//
// The comparison reads the cache, not the device: the file is already on disk
// whole, so it costs two sequential reads of the tensor and no GPU readback
// (GPU.1). It reads both ranges in matching 16 MiB slices and drops each pair
// once compared, so the page holds two slices whatever the tensor's size
// (WASM.9), and it stops at the first slice that differs.
//
// Guidelines (the C++ performance guidelines; their WASM and GPU entries
// govern the page as much as the module):
//   GPU.1  Keep data on the device; budget every round trip — the comparison
//          reads the cache, never the device.
//   WASM.9 Stream in bounded chunks — two 16 MiB slices at a time, whatever
//          the tensor.
//
//   confirmDuplicates({ file, candidates, signal })
//     candidates: [{ tensor, copies, offset, copiesOffset, length }]
//     resolves with the `tensor` of each candidate whose bytes match; a
//     range that runs past the file is a defect in the report, and rejects
//     naming the tensor — it is never quietly left unconfirmed.

export const COMPARE_SLICE_BYTES = 16 * 2 ** 20;

export async function confirmDuplicates({ file, candidates, signal, sliceBytes = COMPARE_SLICE_BYTES }) {
  const confirmed = [];
  for (const candidate of candidates) {
    if (await sameBytes(file, candidate, sliceBytes, signal)) confirmed.push(candidate.tensor);
  }
  return confirmed;
}

async function sameBytes(file, { tensor, offset, copiesOffset, length }, sliceBytes, signal) {
  if (offset + length > file.size || copiesOffset + length > file.size) {
    throw new RangeError(`duplicate candidate ${tensor} reads past the end of the file`);
  }
  for (let at = 0; at < length; at += sliceBytes) {
    signal?.throwIfAborted();
    const n = Math.min(sliceBytes, length - at);
    const [a, b] = await Promise.all([
      file.slice(offset + at, offset + at + n).arrayBuffer(),
      file.slice(copiesOffset + at, copiesOffset + at + n).arrayBuffer(),
    ]);
    if (!equalBuffers(a, b)) return false;
  }
  return true;
}

// Optimization (browser): compares four bytes a step through word views,
// then the last few one at a time; an ArrayBuffer always starts on a word.
export function equalBuffers(a, b) {
  if (a.byteLength !== b.byteLength) return false;
  const words = Math.floor(a.byteLength / 4);
  const wa = new Uint32Array(a, 0, words);
  const wb = new Uint32Array(b, 0, words);
  for (let i = 0; i < words; i++) {
    if (wa[i] !== wb[i]) return false;
  }
  const ba = new Uint8Array(a);
  const bb = new Uint8Array(b);
  for (let i = words * 4; i < a.byteLength; i++) {
    if (ba[i] !== bb[i]) return false;
  }
  return true;
}
