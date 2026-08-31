# Architecture

Module boundaries and the call tree, as built. Present tense only — planned
structure lives in [`README.md`](../README.md#repository-structure).

## Module boundaries

```mermaid
flowchart TD
    subgraph WEB["web/ — presentation"]
        idx["index.html"]
        app["app.js"]
        wrk["worker.js"]
    end

    subgraph WASM["src/wasm/ — platform wrapper"]
        bind["bindings.cpp"]
    end

    subgraph CORE["src/core/ — platform-neutral C++"]
        dev["gpu/device"]
        chk["gpu/self_check"]
        dsp["gpu/dispatch_math"]
        hnd["gpu/unique_handle<br/>gpu/wgpu_handles"]
    end

    wgpu["webgpu.h<br/>emdawnwebgpu"]

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
```

Arrows point inward only. Enforced by
[`tools/check_boundaries.sh`](../tools/check_boundaries.sh):

- No `emscripten.h`, `EM_JS`, `EM_ASM` or `EMSCRIPTEN_` under `src/core/`.
- No include from `core/` into `wasm/` or `web/`.
- `src/wasm/bindings.cpp` is the only Emscripten-aware translation unit.

## Call tree

```mermaid
flowchart TD
    t_entry["worker.js: _bllm_run_self_check()"]
    t_req["Device::request()"]
    t_inst["wgpuCreateInstance"]
    t_ad["wgpuInstanceRequestAdapter ⟶ callback"]
    t_info["wgpuAdapterGetInfo / GetLimits"]
    t_devreq["wgpuAdapterRequestDevice(requiredLimits) ⟶ callback"]
    t_grant["wgpuDeviceGetQueue / GetLimits"]
    t_ondev["on_device(unique_ptr&lt;Device&gt;)"]
    t_chk["run_self_check()"]
    t_dsp["dispatch_count()"]
    t_bufs["create buffers, upload"]
    t_pipe["shader module, pipeline, bind group"]
    t_sub["encode dispatch, copy, submit"]
    t_map["wgpuBufferMapAsync ⟶ callback"]
    t_cmp["compare vs CPU result"]
    t_onchk["on_self_check(result)"]
    t_out["bllm_deliver ⟶ bllmOnResult ⟶ postMessage"]

    t_entry --> t_req
    t_req --> t_inst
    t_req --> t_ad
    t_ad --> t_info
    t_ad --> t_devreq
    t_devreq --> t_grant
    t_devreq --> t_ondev
    t_ondev --> t_chk
    t_chk --> t_dsp
    t_chk --> t_bufs
    t_chk --> t_pipe
    t_chk --> t_sub
    t_chk --> t_map
    t_map --> t_cmp
    t_map --> t_onchk
    t_onchk --> t_out
```

`⟶ callback` marks the two asynchronous hops. The build has ASYNCIFY off, so
these are continuations rather than blocking waits, and every failure exits
through the same reporting path as success.
