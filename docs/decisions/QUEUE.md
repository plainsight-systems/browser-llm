# Work Queue

This file tracks active and accepted work.

## Active

- **BLLM-001: Repo skeleton and build system.** Packet:
  `packets/2026-08-29-repo-skeleton-and-build-system.md`. Status
  `changes_requested` after independent codex review
  (`research/2026-08-29-repo-skeleton-and-build-system-codex-review.md`).
  Implementation is complete and deployed; the review's P1 and P2 findings
  have been worked. Second review completed 2026-08-30 with a full guideline
  audit (no environment failure) and again returned `changes_requested`;
  `research/2026-08-30-bllm-001-codex-review-2.md`. Its P1 and the mechanical
  P2/P3 findings are fixed. Ownership of the device now flows through
  `run_self_check` rather than being re-read from module state, which closed
  the singleton half of the review's callback finding.

  Remaining inside this packet, from its own review: no timeout on the async
  callbacks, so a callback that never fires wedges the module permanently.
  Being worked now. Review findings stay with the packet that produced them
  rather than becoming a second packet held open alongside it.

## Accepted

- None.

## Parking Lot

- None.
