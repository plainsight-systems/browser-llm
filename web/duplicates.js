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
//     resolves with the `tensor` of each candidate whose bytes match
