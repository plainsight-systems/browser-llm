#include <doctest/doctest.h>

#include "core/run_guard.h"

using bllm::RunGuard;

TEST_CASE("a fresh guard is idle and begin yields a usable generation") {
    RunGuard g;
    CHECK_FALSE(g.active());
    const auto gen = g.begin();
    CHECK(gen != RunGuard::kNoRun);
    CHECK(g.active());
}

TEST_CASE("a second begin while a run is live is refused") {
    RunGuard g;
    const auto first = g.begin();
    CHECK(first != RunGuard::kNoRun);
    CHECK(g.begin() == RunGuard::kNoRun);
    CHECK(g.begin() == RunGuard::kNoRun);
    // The refusals must not have disturbed the live run.
    CHECK(g.complete(first));
}

TEST_CASE("completing the live generation ends the run") {
    RunGuard g;
    const auto gen = g.begin();
    CHECK(g.complete(gen));
    CHECK_FALSE(g.active());
}

TEST_CASE("completing twice reports false the second time") {
    // This is what stops a run being reported to the page twice.
    RunGuard g;
    const auto gen = g.begin();
    CHECK(g.complete(gen));
    CHECK_FALSE(g.complete(gen));
}

TEST_CASE("a stale generation cannot complete a later run") {
    // The timeout fired for run 1 and reported failure. Run 2 started. Run 1's
    // real callback now arrives late — it must not close run 2.
    RunGuard g;
    const auto stale = g.begin();
    REQUIRE(g.complete(stale));

    const auto live = g.begin();
    CHECK(live != stale);
    CHECK_FALSE(g.complete(stale));
    CHECK(g.active());        // run 2 is untouched
    CHECK(g.complete(live));
}

TEST_CASE("first past the post wins, whether that is the callback or the timeout") {
    RunGuard g;
    const auto gen = g.begin();

    // Whoever calls first owns the run; the other learns it is stale and
    // stays silent rather than reporting a contradictory second result.
    CHECK(g.complete(gen));
    CHECK_FALSE(g.complete(gen));
}

TEST_CASE("kNoRun never completes anything") {
    RunGuard g;
    CHECK_FALSE(g.complete(RunGuard::kNoRun));
    g.begin();
    CHECK_FALSE(g.complete(RunGuard::kNoRun));
    CHECK(g.active());
}

TEST_CASE("generations are distinct across successive runs") {
    RunGuard g;
    const auto a = g.begin();
    REQUIRE(g.complete(a));
    const auto b = g.begin();
    REQUIRE(g.complete(b));
    const auto c = g.begin();
    CHECK(a != b);
    CHECK(b != c);
    CHECK(a != c);
}
