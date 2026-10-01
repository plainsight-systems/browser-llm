// Compile check only: every platform-neutral contract and module header builds
// natively, together, with nothing else included first. The headers that hold
// GPU objects (residency/upload.h) build in the wasm configuration only.

#include "core/arch/architecture.h"
#include "core/cache/kv.h"
#include "core/cache/prefix.h"
#include "core/capability/capability.h"
#include "core/formats/format.h"
#include "core/gguf/arrays.h"
#include "core/gguf/index.h"
#include "core/kernels/interface.h"
#include "core/model/model_description.h"
#include "core/policy/policy.h"
#include "core/preflight/preflight.h"
#include "core/residency/plan.h"
#include "core/residency/weight_view.h"
#include "core/runtime/runtime.h"
#include "core/sampler/sampler.h"
#include "core/tokenizer/pretokenize.h"
#include "core/tokenizer/tokenizer.h"
