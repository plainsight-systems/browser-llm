# WebGPU limits: adapter maxima are not device limits

**Date:** 2026-08-29, **corrected 2026-08-30**
**Context:** BLLM-001 self-check, Chrome on Apple silicon (`apple` / `metal-3`).

## Correction

The first version of this note claimed that observed limits "far exceed the
spec minimums" and concluded that weight residency was therefore a portability
decision rather than a hard constraint. **That was wrong, and it was wrong in a
way that mattered**: it read `wgpuAdapterGetLimits` as if it described the
device that had been acquired.

It does not. WebGPU grants a device the **default** limits unless better ones
are named in `requiredLimits` at acquisition, and validation enforces the
*device's* limits, not the adapter's maxima.

## Measured, both sides

Same machine, same run, after querying `wgpuDeviceGetLimits` separately:

| Limit | Spec default | Device (enforced) | Adapter (could grant) |
|---|---|---|---|
| `maxBufferSize` | 256 MiB | **256 MiB** | 4096 MiB |
| `maxStorageBufferBindingSize` | 128 MiB | **128 MiB** | 4096 MiB |
| `maxComputeInvocationsPerWorkgroup` | 256 | **256** | 1024 |
| `maxComputeWorkgroupsPerDimension` | 65,535 | 65,535 | 65,535 |

The acquired device sits exactly on the spec defaults, because this project
requested no better limits. The 4 GiB figure is capability the adapter would
grant *if asked* — and was never asked for.

## What this means

The original design conclusion was right after all: a several-hundred-megabyte
model **must** be split across many bound buffers, because 128 MiB is the
binding size actually in force.

The real decision for BLLM-002 is different from either previous framing:

- **Ask for nothing** — run on the defaults, work everywhere WebGPU works, and
  accept 128 MiB storage-buffer bindings as a hard design constraint.
- **Ask for more via `requiredLimits`** — better residency on capable hardware,
  but acquisition *fails* on any device that cannot grant it. That is a
  portability floor being declared, and it must be declared deliberately.

Either way the floor is a product decision, and the harness must request what
it needs at acquisition and fail explicitly when it is unavailable, rather than
discovering the shortfall mid-load.

## Lesson

Two queries that look interchangeable were not, and the difference silently
inverted a design conclusion. It survived a self-review and was caught only by
independent review. Capability reported by a component is not capacity granted
to you — check the thing you were actually given.
