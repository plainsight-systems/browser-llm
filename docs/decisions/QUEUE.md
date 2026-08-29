# Work Queue

This file tracks active and accepted work.

## Active

- **Repo governance bootstrap (2026-08-28).** Governance surface created per
  `repo_creation_runbook.md`. No packet: this is a decision-doc bootstrap where
  the decision conversation is the work. Implementation complete; awaiting human
  acceptance.

## Ready

- **BLLM-001: Harness architecture decision packet.** Not yet written. Must be
  scoped with a human before any implementation, because it settles the Open
  Questions in `MEMORY.md` (target model, quantization, blessed browser/GPU
  targets, C++/JS boundary, WebGPU access path, performance budget). Change
  class: architectural. Requires a C++ architecture note and a C++ performance
  note.

## Accepted

- None.

## Parking Lot

- **Publish the inherited C++ governance, and pick licenses.** Deferred until
  this repo gets a remote; nothing depends on it before then. Six of the eight
  inherited docs are publication candidates, `plainsight_policies.md` is not,
  and `engineering_philosophies.md` is already public in redacted form. Org
  precedent is CC BY 4.0 for prose and Apache-2.0 for code
  (`plainsight-systems/cpp-perf-guidelines`). Note the governance repo currently
  has no LICENSE file at all. Open question: whether the public subset extends
  `plainsight-systems/.github` or gets a dedicated repo.
