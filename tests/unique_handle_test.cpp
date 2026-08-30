#include <doctest/doctest.h>

#include <vector>

#include "core/gpu/unique_handle.h"

// Pins the ownership invariant that manual wgpuXRelease calls violated. The
// template is deliberately free of webgpu.h so this runs natively, with no
// GPU and no browser.
namespace {

using Handle = int*;

std::vector<Handle> g_released;
void fake_release(Handle h) { g_released.push_back(h); }

using TestHandle = bllm::gpu::UniqueHandle<Handle, fake_release>;

int a = 1;
int b = 2;

struct Fixture {
    Fixture() { g_released.clear(); }
};

}  // namespace

TEST_CASE_FIXTURE(Fixture, "a default-constructed handle owns nothing and releases nothing") {
    { TestHandle h; CHECK_FALSE(static_cast<bool>(h)); }
    CHECK(g_released.empty());
}

TEST_CASE_FIXTURE(Fixture, "destruction releases exactly once") {
    { TestHandle h(&a); CHECK(static_cast<bool>(h)); }
    REQUIRE(g_released.size() == 1);
    CHECK(g_released[0] == &a);
}

TEST_CASE_FIXTURE(Fixture, "move transfers ownership and does not double release") {
    {
        TestHandle first(&a);
        TestHandle second(std::move(first));
        CHECK_FALSE(static_cast<bool>(first));
        CHECK(second.get() == &a);
        CHECK(g_released.empty());   // nothing released by the move itself
    }
    CHECK(g_released.size() == 1);
}

TEST_CASE_FIXTURE(Fixture, "move assignment releases the target's previous handle") {
    {
        TestHandle target(&a);
        TestHandle source(&b);
        target = std::move(source);
        REQUIRE(g_released.size() == 1);
        CHECK(g_released[0] == &a);   // the overwritten handle, released once
        CHECK(target.get() == &b);
    }
    REQUIRE(g_released.size() == 2);
    CHECK(g_released[1] == &b);
}

TEST_CASE_FIXTURE(Fixture, "self move assignment does not release") {
    {
        TestHandle h(&a);
        auto& alias = h;
        h = std::move(alias);
        CHECK(h.get() == &a);
        CHECK(g_released.empty());
    }
    CHECK(g_released.size() == 1);
}

TEST_CASE_FIXTURE(Fixture, "reset releases the old handle and adopts the new one") {
    {
        TestHandle h(&a);
        h.reset(&b);
        REQUIRE(g_released.size() == 1);
        CHECK(g_released[0] == &a);
        CHECK(h.get() == &b);
    }
    CHECK(g_released.size() == 2);
}

TEST_CASE_FIXTURE(Fixture, "release relinquishes ownership without releasing") {
    {
        TestHandle h(&a);
        const Handle raw = h.release();
        CHECK(raw == &a);
        CHECK_FALSE(static_cast<bool>(h));
    }
    CHECK(g_released.empty());   // caller now owns it; nothing was released
}
