#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace bllm::gguf {

// A random-access source of file bytes that knows how long the file is.
//
// The reader validates every offset and length against `size()` — the
// authoritative total — not against whatever window happens to be available.
// That is the whole point of this interface: a parser holding one span over
// the bytes would either require the entire 420 MB file resident, or be unable
// to validate a tensor region that lies beyond the mapped prefix.
//
// Two implementations exist: an in-memory one for tests and fixtures, and a
// chunked one over fetch/OPFS in the browser. The seam is real, not
// speculative.
class ByteSource {
public:
    virtual ~ByteSource() = default;

    ByteSource() = default;
    ByteSource(const ByteSource&) = delete;
    ByteSource& operator=(const ByteSource&) = delete;
    ByteSource(ByteSource&&) = delete;
    ByteSource& operator=(ByteSource&&) = delete;

    // Authoritative length of the whole file, regardless of what is resident.
    [[nodiscard]] virtual std::uint64_t size() const noexcept = 0;

    // Fills `out` from `offset`. Returns false — without writing anything the
    // caller should trust — if the range is not wholly inside the file or the
    // bytes could not be obtained. Implementations must never read past
    // `size()`.
    [[nodiscard]] virtual bool read(std::uint64_t offset,
                                    std::span<std::byte> out) noexcept = 0;
};

// In-memory source over a caller-owned buffer. Used by tests and fixtures; the
// browser uses a chunked source instead.
class MemoryByteSource final : public ByteSource {
public:
    explicit MemoryByteSource(std::span<const std::byte> bytes) noexcept
        : bytes_(bytes) {}

    [[nodiscard]] std::uint64_t size() const noexcept override {
        return static_cast<std::uint64_t>(bytes_.size());
    }

    [[nodiscard]] bool read(std::uint64_t offset,
                            std::span<std::byte> out) noexcept override {
        // Checked against the total, and written so the addition cannot wrap:
        // offset + out.size() could overflow for a hostile offset.
        const auto want = static_cast<std::uint64_t>(out.size());
        if (offset > size() || want > size() - offset) {
            return false;
        }
        if (want != 0) {
            std::memcpy(out.data(), bytes_.data() + offset,
                        static_cast<std::size_t>(want));
        }
        return true;
    }

private:
    std::span<const std::byte> bytes_;
};

}  // namespace bllm::gguf
