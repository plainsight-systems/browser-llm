#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/index.h"
#include "core/policy/policy.h"
#include "core/residency/plan.h"

namespace bllm::preflight {

// Axis I: applies the capability table; it holds no criteria of its own.
//
// How far this build can take a model, judged from its tensor index before
// any weight byte is fetched. A model passes through ordered stages, each
// needing every stage before it:
//
//   Read       the file is GGUF and its index reads (decided by the reader;
//              a file that does not read has no verdict)
//   Download   the file can be fetched and cached — every readable file can
//   Describe   its architecture is implemented and can describe this file
//   Fit        its residency plan fits the granted limits and the budget
//   Upload     its weights are written to the GPU and read back identical
//   Run        it generates: graph, every weight format, tokenizer
//
// The verdict lists blockers, each naming the stage it stops and why: the
// architecture, each unsupported format with how many tensors use it and the
// first, the tokenizer or pre-tokenizer, or a stage this build does not
// implement. Every check that can run does, so the verdict says everything a
// model still needs, not only what stops the next stage. The stage reached is
// derived from the blockers, never stored beside them.

enum class Stage {
    Read,
    Download,
    Describe,
    Fit,
    Upload,
    Run,
};

[[nodiscard]] constexpr std::string_view to_string(Stage stage) noexcept {
    switch (stage) {
        case Stage::Read: return "read";
        case Stage::Download: return "download";
        case Stage::Describe: return "describe";
        case Stage::Fit: return "fit";
        case Stage::Upload: return "upload";
        case Stage::Run: return "run";
    }
    return "unknown";
}

// The furthest stage this build implements. Every later stage is blocked, by
// name, whatever the model: a verdict never claims a stage that does not
// exist. Advanced when the next stage is built.
inline constexpr Stage kImplementedThrough = Stage::Describe;

// What stops `stage`. Never Read or Download: those depend only on the
// reader.
struct Blocker {
    Stage stage;
    std::string detail;
};

struct Verdict {
    std::vector<Blocker> blockers;

    // The furthest stage reached: the one before the earliest stage blocked.
    [[nodiscard]] Stage reached() const noexcept {
        Stage earliest = Stage::Run;
        bool blocked = false;
        for (const Blocker& b : blockers) {
            if (!blocked || b.stage < earliest) earliest = b.stage;
            blocked = true;
        }
        if (!blocked) return Stage::Run;
        return static_cast<Stage>(static_cast<int>(earliest) - 1);
    }
};

[[nodiscard]] Verdict preflight(const gguf::TensorIndex& index,
                                const residency::DeviceLimits& limits,
                                const policy::LoadPolicy& policy);

}  // namespace bllm::preflight
