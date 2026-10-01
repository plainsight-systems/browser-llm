#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "core/arch/architecture.h"
#include "core/gguf/index.h"
#include "core/model/model_description.h"

namespace bllm::arch {

// Axis C: changes with GGUF's naming convention for transformer weights.
//
// What describing shares across architectures. GGUF files written by
// llama.cpp's converters name hyperparameters "<arch>.<key>" and a layer's
// weights "blk.<layer>.<suffix>", and a weight in a given role always has the
// same shape in terms of the hyperparameters. An architecture supplies only
// what differs: its name, the roles its layers have and what it calls them,
// and anything its layers do differently from one another.

// The numbers every architecture here declares.
struct Hyperparameters {
    std::uint32_t block_count;
    std::uint32_t context_length;
    std::uint32_t embedding_length;
    std::uint32_t feed_forward_length;
    std::uint32_t head_count;
    std::uint32_t head_count_kv;
    std::uint32_t head_dimension;
    float norm_epsilon;
    float rope_base;
};

// What an architecture calls a layer's weight in a role: blk.<i>.<suffix>.
struct RoleName {
    model::Role role;
    std::string_view suffix;
};

// Reads "<arch>.<key>" for every hyperparameter. The head dimension is
// <arch>.attention.key_length, or the embedding width over the head count when
// the file does not say. Fails on a value no architecture could run with.
[[nodiscard]] DescribeResult read_hyperparameters(const gguf::TensorIndex& index,
                                                  std::string_view arch, Hyperparameters& out);

// Fills `out` from the hyperparameters and the file's tensors: the vocabulary,
// the global weights, and every layer's weights by role, each checked against
// the shape its role requires. Every layer attends over the full context with
// the file's rotary base; an architecture whose layers differ adjusts them
// afterwards.
[[nodiscard]] DescribeResult describe_layers(const gguf::TensorIndex& index,
                                             const Hyperparameters& hp,
                                             std::span<const RoleName> roles,
                                             model::ModelDescription& out);

// Reads an optional "<arch>.<key>": `fallback` when the file omits it, an
// error when it declares it with another type.
[[nodiscard]] DescribeResult read_u32_or(const gguf::TensorIndex& index, std::string_view arch,
                                         std::string_view key, std::uint32_t fallback,
                                         std::uint32_t& out);
[[nodiscard]] DescribeResult read_f32_or(const gguf::TensorIndex& index, std::string_view arch,
                                         std::string_view key, float fallback, float& out);

}  // namespace bllm::arch
