# Load performance audit: the floor, the path, and the rule we were missing

**Date:** 2026-10-03.
**Context:** Qwen3 0.6B (382 MB, Q4_0 / Q4_1 / Q6_K / F32), loaded from the
browser's cache onto the GPU. Apple M3 Max, Chrome 152, release module, warm
(the file in the operating system's cache), not cross-origin isolated (a
100 µs clock, ample at these durations). Best or median of three to five
runs, as marked. Cold loads were not measured: the browser cannot flush the
operating system's cache.

## Method

GDSA.6: write down the bytes each stage moves, measure each stage's ceiling
on its own on this machine, and compare the whole with that floor. The load
path, per 16 MiB chunk: read the cached file, hand it to the module, rearrange
each piece's blocks into streams, `writeBuffer` each stream, wait for the
previous chunk's queue work.

## Each stage alone

| Stage | Measured |
|---|---|
| Cache read, blob, one at a time | 122–128 ms (3.1 GB/s) |
| Cache read, blob, two / three / four in flight | 77 / 48 / 41 ms |
| Cache read, `FileSystemSyncAccessHandle` in a worker, into wasm memory | 23–26 ms (16 GB/s) |
| memcpy, native | 63 GB/s |
| Rearrangement, native, runtime-width copies (Q4_0) | 83 ms per 360 MiB (4.5 GB/s) |
| Rearrangement, native, fixed-width copies (Q4_0) | 11.9 ms per 360 MiB (31.8 GB/s) |
| Rearrangement, wasm in Node, fixed-width copies (Q4_0) | about 17 ms per 360 MiB |
| `writeBuffer` + GPU, 23 writes of 16 MiB, paced | 41–77 ms |
| The same bytes as 25 writes a chunk | 365–407 ms |

## The path, before and after

| | Load (warm, end to end) |
|---|---|
| Before the audit | about 298 ms |
| Fixed-width rearrangement (95a1c62) | about 230 ms |
| Worker reads the cache into the module (a70fb55) | 241 ms median of five, against 257 |

The reads gained less than their stage measurement promised: the page's reads
had overlapped the GPU process working through the previous chunk's writes.
What binds the load now is the number of `writeBuffer` calls. The module
issues 567 a load, one per stream of every piece; merged where they touch or
are separated only by the plan's alignment padding they are 78, and replayed
both ways on a clean page the write phase halves. A `writeBuffer` call costs
about half a millisecond in Chrome at these sizes, whatever its length.

## Why the audit had to be asked for

Each optimization in the load was designed by reasoning and labelled — one
write per stream rather than per block, F32 written straight from the chunk,
two chunks in flight — and each was right locally. None was checked against
the path's floor, because the design never stated one:

- **No accounting of the whole path.** GDSA.6 asks for the bytes each stage
  moves and a comparison with memcpy. The headers cited it for one stage at
  most, and no document added the stages up, so a stage running at a
  fourteenth of memcpy went unseen.
- **Crossings were counted on one side only.** WASM.2 prices every call out
  of wasm into a web API. The headers counted the page's crossings into the
  module (46 a load) and never the module's calls into WebGPU (567).
- **Measurement was deferred, then dropped on a guess.** The design said
  "measured where it can be", the harness that would have measured it was
  postponed, and I then recommended dropping it because the load "isn't
  where the performance story is" — a judgement about importance, made
  without the numbers GPU.10 asks for before it.
- **Smoke timings stood in for evidence.** 0.33 s and 1.2 s looked fast, and
  nothing said fast compared with what.

## The rule

**A data path is not done until it has been measured against its floor.**
Before a path's design is accepted, its header accounts every stage: the
bytes it moves, the calls it makes across every boundary, including from the
module into browser APIs, and the ceiling that stage reaches alone on the
target. Once the path runs, it is measured end to end against the sum or the
overlap of those ceilings, and any stage well off its own is either fixed or
named as the remaining gap. A labelled optimization claims its effect
against that floor, not against intuition.

## Levers not taken here

- Merging writes (above): the measured lever, a change to the piece writer's
  output contract.
- Rearranging on the GPU instead: one raw write a chunk and a compute pass;
  not pursued while merging is the simpler fix.
- Wasm SIMD: no measurable effect on rearrangement, in Node or Chrome.
