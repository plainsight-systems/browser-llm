# Measurement without lying: build configurations for performance work

**Date:** 2026-08-31

Method notes from a prior in-house C++ inference harness, generalised. The
problem this solves: instrumentation that is present when you measure
throughput makes the throughput number wrong, and instrumentation that is
absent when you diagnose makes the diagnosis impossible.

Their answer is three configurations and one compile-time switch, and the part
that is easy to get wrong is which axis the switch sits on.

## The shape

| Configuration | Optimisation | Instrumentation | Purpose |
|---|---|---|---|
| **clean** | Release | **compiled out** | throughput numbers you are allowed to quote |
| **diagnostic** | **Release** | compiled in | finding out *why*, knowingly perturbed |
| **debug** | Debug | compiled in | correctness, not performance at all |

**The diagnostic build is Release, not Debug.** This is the whole trick. A
Debug build cannot diagnose a Release performance problem — the optimiser is
off, so the code being measured is not the code that ships. The diagnostic
configuration is the *same optimised build* with measurement hooks compiled in,
accepting that the hooks perturb what they observe.

So the axis is **instrumentation, not optimisation**. Reaching for a Debug
build to investigate a performance problem is the mistake this arrangement
prevents.

## The compile-time gate

```cpp
#ifndef X_TRACE_ENABLED
#define X_TRACE_ENABLED 0      // absent define == free, never accidentally on
#endif

constexpr bool compiled_enabled() noexcept { return X_TRACE_ENABLED != 0; }
```

Two details worth copying:

**Default off, in the header.** A translation unit that never receives the
build-system define still compiles, and compiles to the free version. Enabling
instrumentation is opt-in at the configuration level; forgetting cannot silently
cost you.

**The disabled branch is inline no-op functions, not macros expanding to
nothing.** Every parameter is still named and `(void)`-cast:

```cpp
inline void counter(std::string_view name, std::int64_t value,
                    std::string_view unit) noexcept {
    (void)name; (void)value; (void)unit;
}
```

This matters more than it looks. With macros-to-nothing, a call site can
reference a variable that no longer exists, or pass the wrong type, and nothing
fails until somebody enables tracing months later — at which point the
diagnostic build is broken exactly when it is needed. With inline no-ops the
call sites are **still type-checked in every build**, so instrumentation cannot
rot. The optimiser removes them entirely.

`constexpr bool compiled_enabled()` lets code branch at compile time on whether
instrumentation exists, without `#ifdef` at the call site.

The scoped timer is non-copyable *and* non-movable in **both** branches, so the
enabled and disabled versions have identical semantics at every call site.

Their trace header also sits deliberately outside the public include directory —
it is an internal observability hook, not part of the product's API.

## The measurement-methodology half

Instrumentation hygiene is necessary and not sufficient. The same body of work
recorded a case where a benchmark sweep reported materially lower throughput
than the baseline, and the cause was neither the code nor the instrumentation:
**sustained load and benchmark ordering** had induced a machine state that the
harness's own environmental probe reported as nominal.

Two lessons transfer regardless of platform:

- A cell measured after heavy preceding load is not comparable to the same cell
  measured cold. Ordering is part of the measurement, so record it.
- An environmental probe reporting "nominal" is evidence about the probe, not
  proof about the machine. If a number moves and the code did not, suspect the
  measurement before the source.

## What this means for browser-llm

We have the same problem with different mechanics.

**Configurations.** `wasm-release` is our clean build. We need a `wasm-diag`
alongside it: same optimisation, instrumentation compiled in. Native builds get
the same switch, so kernel diagnostics are available without a browser.

**The GPU dimension has no CPU analogue.** WebGPU timestamp queries require the
`timestamp-query` feature to be *requested at device acquisition*. So the choice
is made before any measurement, in `Device::request` — which means the
diagnostic configuration changes how the device itself is created, not merely
what the host records. That must be explicit: a device acquired with timestamp
queries enabled is not the device the clean build measures.

**Two artifacts, never one run.** Peak-heap instrumentation and clean
upload-throughput cannot come from the same execution; the first perturbs the
second. Each measurement record should name which configuration produced it,
and a number from the diagnostic build must never be quoted as throughput.

**We already have an instance of this to fix.** `readback_bench` is diagnostic
instrumentation compiled into the production wasm module and reached via
`?bench`. Under this pattern it belongs in the diagnostic configuration,
compiled out of the clean build. Its measurements remain valid — they were
taken to characterise the round trip, which is diagnosis, not a throughput
claim — but it should not be shipping in the module the page loads.

## Relation to the performance corpus

`TLM.1` (compile telemetry out by default) is the compile-time gate. `TLM.2`
(channelize — separate cost models per telemetry kind) is why heap tracking and
timing are independent switches rather than one flag. `TLM.6` (a self-test is a
correctness gate, not a benchmark) is why no number from an instrumented run is
quotable as throughput.

The corpus supplies the rules; the three-configuration arrangement above is one
concrete way to satisfy them, and the Release-diagnostic build is the part the
rules do not spell out.
