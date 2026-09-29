#pragma once

#include <cstdint>

namespace bllm::kernels {

// Contract 7: kernel launch.
//
// Every kernel binds the same way, so one piece of code launches all of them
// and changing a kernel never changes how the others are called. In bind
// group 0:
//
//   binding 0   the step's parameters — position and token count — shared by
//               every launch and written once per step
//   binding 1   this launch's constants, written once at load
//   binding 2…  the kernel's weights, in the order it declares them, then its
//               activations
//
//   - Bind groups are built at load, one per launch in the graph. No bind
//     group, pipeline or buffer is created per token, so a token costs one
//     uniform write and its dispatches.
//   - The regime is chosen per step from its token count.
//   - Dispatch geometry is arithmetic in core/gpu/dispatch_math. Workgroup
//     size belongs to each kernel: the right value differs per kernel, and a
//     shared constant would couple them.

enum class Regime {
    // One token: matrix times vector, bound by weight bandwidth.
    Decode,
    // Many tokens: matrix times block, bound by arithmetic.
    Prefill,
};

inline constexpr std::uint32_t kBindGroup = 0;
inline constexpr std::uint32_t kStepBinding = 0;
inline constexpr std::uint32_t kLaunchBinding = 1;
inline constexpr std::uint32_t kFirstWeightBinding = 2;

[[nodiscard]] Regime regime_for(std::uint32_t tokens_in_step) noexcept;

}  // namespace bllm::kernels
