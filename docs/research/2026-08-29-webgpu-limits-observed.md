# Observed WebGPU limits vs. the spec guarantees

**Date:** 2026-08-29
**Context:** BLLM-001 self-check, Chrome on Apple silicon (vendor `apple`,
architecture `metal-3`).

## Finding

During design discussion the buffer limits were stated as hard caps:
`maxStorageBufferBindingSize` 128MB and `maxBufferSize` 256MB. That was wrong
as stated. Those are the **guaranteed minimums** every conformant
implementation must support, not what a given adapter reports.

Measured on this device:

| Limit | Guaranteed minimum | Observed here |
|---|---|---|
| `maxBufferSize` | 256 MiB | 4096 MiB |
| `maxStorageBufferBindingSize` | 128 MiB | 4096 MiB |
| `maxComputeWorkgroupsPerDimension` | 65,535 | 65,535 |
| `maxComputeInvocationsPerWorkgroup` | 256 | 1,024 |

## Why it matters

The earlier conclusion — that a several-hundred-megabyte model must be split
across many bound buffers *regardless* of device — does not hold universally.
On a capable adapter the weights could plausibly live in far fewer, larger
buffers.

This does not make buffer packing unnecessary. It changes it from an
unavoidable constraint into a **portability decision**:

- Designing to the 128MB floor means the harness runs anywhere WebGPU runs.
- Designing to observed limits means better residency on capable hardware and
  outright failure on conformant-but-minimal devices.

Two of the four limits on this device are at exactly the guaranteed minimum,
which is a useful reminder that the floor is not hypothetical.

## Consequence for BLLM-002

Weight residency must be planned against a **declared floor**, and the floor is
a product decision, not a discovered fact. The harness should query limits at
startup and fail explicitly when a device is below the declared floor, rather
than assuming capacity and faulting mid-load.

Any performance baseline must record the adapter and its limits, since the same
code will make different residency choices on different devices.
