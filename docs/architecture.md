# Architecture

**As built on 2026-08-30.** Nothing here is planned or aspirational — every box
exists, every arrow is a call in the current tree. Planned structure lives in
[`README.md`](../README.md#repository-structure); this file is the present tense.

Scope today: the toolchain reaches the GPU and runs one compute shader. No
model, no inference.

---

## 1. Module boundaries

Dependency direction is one-way and mechanically enforced by
[`tools/check_boundaries.sh`](../tools/check_boundaries.sh).

```mermaid
flowchart TD
    subgraph WEB["web/ — presentation"]
        idx["index.html"]
        app["app.js<br/>render, GPU capability gate"]
        wrk["worker.js<br/>owns the wasm module"]
    end

    subgraph WASM["src/wasm/ — platform wrapper"]
        bind["bindings.cpp<br/>ONLY Emscripten-aware TU"]
    end

    subgraph CORE["src/core/ — platform-neutral C++"]
        dev["gpu/device<br/>acquisition, limits"]
        chk["gpu/self_check<br/>dispatch + verify"]
        dsp["gpu/dispatch_math<br/>pure arithmetic"]
        hnd["gpu/unique_handle<br/>+ wgpu_handles"]
    end

    subgraph EXT["external"]
        wgpu["webgpu.h<br/>via emdawnwebgpu"]
    end

    idx --> app
    app --> wrk
    wrk --> bind
    bind --> dev
    bind --> chk
    chk --> dsp
    chk --> hnd
    dev --> hnd
    dev --> wgpu
    chk --> wgpu

    classDef forbidden stroke-dasharray: 5 5
    CORE:::forbidden
```

**Enforced invariants** — each proven to fire by
[`tests/test_check_boundaries.sh`](../tests/test_check_boundaries.sh):

| Rule | Meaning |
|---|---|
| No `emscripten.h`, `EM_JS`, `EM_ASM`, `EMSCRIPTEN_` under `src/core/` | core compiles natively unchanged |
| No include from `core/` into `wasm/` or `web/` | arrows point inward only |
| Exactly `src/wasm/bindings.cpp` may be Emscripten-aware | exact path allowlist, not a count |

---

## 2. Build targets

The same core sources feed two targets. `device` and `self_check` compile only
under Emscripten today, because natively `webgpu.h` means linking Dawn — that is
deferred to the first model kernel (BLLM-003).

```mermaid
flowchart LR
    subgraph SRC["sources"]
        s_dsp["dispatch_math.cpp"]
        s_hnd["unique_handle.h"]
        s_dev["device.cpp"]
        s_chk["self_check.cpp"]
        s_wgsl["shaders/*.wgsl"]
        s_bind["wasm/bindings.cpp"]
    end

    gen["cmake codegen<br/>shaders_generated.h"]

    subgraph NAT["native config"]
        t_core_n["browser_llm_core"]
        t_test["browser_llm_tests<br/>doctest, pinned by commit"]
    end

    subgraph WSM["wasm config"]
        t_core_w["browser_llm_core"]
        t_mod["browser_llm.mjs + .wasm"]
    end

    s_wgsl --> gen
    gen --> t_core_n
    gen --> t_core_w
    s_dsp --> t_core_n
    s_hnd --> t_core_n
    s_dsp --> t_core_w
    s_hnd --> t_core_w
    s_dev --> t_core_w
    s_chk --> t_core_w
    s_bind --> t_mod
    t_core_n --> t_test
    t_core_w --> t_mod
```

---

## 3. Runtime call sequence

One pass, from page load to rendered result. Two asynchronous hops sit inside
C++ — the build has ASYNCIFY off, so these are continuations, not blocking waits.

```mermaid
sequenceDiagram
    autonumber
    participant P as app.js
    participant W as worker.js
    participant B as bindings.cpp
    participant D as core/device
    participant S as core/self_check
    participant G as WebGPU / GPU

    P->>P: navigator.gpu present?
    P->>W: new Worker(type module)
    W->>W: await createModule()
    W-->>P: postMessage ready
    W->>B: _bllm_run_self_check()

    B->>D: Device::request(on_device)
    D->>G: wgpuCreateInstance
    D->>G: wgpuInstanceRequestAdapter
    G-->>D: adapter callback
    D->>G: wgpuAdapterGetInfo / GetLimits
    D->>G: wgpuAdapterRequestDevice(requiredLimits = adapter maxima)
    G-->>D: device callback
    D->>G: wgpuDeviceGetLimits (what was granted)
    D-->>B: on_device(unique_ptr Device)

    B->>S: run_self_check(device, 4096)
    S->>S: dispatch_count vs granted limits
    S->>G: create buffers, write lhs and rhs
    S->>G: shader module, pipeline, bind group
    S->>G: encode dispatch, copy to readback, submit
    S->>G: wgpuBufferMapAsync
    G-->>S: map callback
    S->>S: compare every element vs CPU
    S-->>B: on_self_check(result)

    B->>W: bllm_deliver -> globalThis.bllmOnResult
    W-->>P: postMessage result
    P->>P: render adapter, limits, status
```

---

## 4. Device acquisition — control flow

Every exit is explicit. There is no path that returns a default-looking value
on failure.

```mermaid
stateDiagram-v2
    [*] --> CreateInstance
    CreateInstance --> RequestAdapter: instance ok
    CreateInstance --> Failed: null instance

    RequestAdapter --> QueryAdapter: adapter callback ok
    RequestAdapter --> Failed: status not success

    QueryAdapter --> RequestDevice: limits queried
    QueryAdapter --> Failed: GetLimits failed

    RequestDevice --> ReadGranted: device callback ok
    RequestDevice --> Failed: status not success

    ReadGranted --> Ready: DeviceGetLimits ok
    ReadGranted --> Failed: DeviceGetLimits failed

    Ready --> [*]: unique_ptr Device to caller
    Failed --> [*]: nullptr plus error string

    note right of QueryAdapter
        GetInfo failure is recorded as
        queried=false, not fatal.
        GetLimits failure IS fatal.
    end note
```

---

## 5. Self-check — control flow

```mermaid
flowchart TD
    c_start(["run_self_check"]) --> c_disp{"dispatch_count<br/>vs granted limits"}
    c_disp -->|"not Ok"| c_err(["report error"])
    c_disp -->|"Ok"| c_buf["create 4 buffers"]
    c_buf --> c_null{"any null?<br/>CreateBuffer is nullable"}
    c_null -->|"yes"| c_err
    c_null -->|"no"| c_enc["upload, encode dispatch,<br/>copy to readback, submit"]
    c_enc --> c_map["wgpuBufferMapAsync"]
    c_map --> c_ok{"map succeeded?"}
    c_ok -->|"no"| c_err
    c_ok -->|"yes"| c_cmp["compare every element<br/>against CPU result"]
    c_cmp --> c_res(["report mismatches"])

    c_note["all handles are UniqueHandle:<br/>every early exit above releases<br/>without a manual call"]
    c_enc -.-> c_note
```

---

## 6. Ownership

| Resource | Owner | Released by |
|---|---|---|
| WebGPU handles | `UniqueHandle<H, ReleaseFn>` | destructor, exactly once |
| `Device` | caller, via `std::unique_ptr` | caller |
| `PendingDeviceRequest` | itself, across the async hop | `delete this` on either exit |
| `ReadbackContext` | itself, across the map hop | `delete this` in `finish()` |
| module run state | function-local statics in `bindings.cpp` | process teardown |

The last three rows are **open findings**, tracked as BLLM-005: manual ownership
sitting outside the `UniqueHandle` system, and singleton state that
construct-on-first-use does not eliminate.

---

## 7. What has no automated test

Stated because the gaps matter more than the coverage.

| Covered natively | Not covered |
|---|---|
| `dispatch_math` arithmetic and bounds | device acquisition |
| `UniqueHandle` ownership semantics | granted-limit behavior |
| shader embedding, byte-identical | adapter metadata failure |
| boundary rules fire | device loss |
| review preflight refuses | GPU execution, browser page contract |

The right column needs native Dawn (BLLM-003) or a browser test. Until then the
GPU path is verified only by the readback assertion on the page itself.
