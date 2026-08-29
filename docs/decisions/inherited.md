# Inherited Governance

This repo inherits governance from `plainsight-systems-governance`, which is
currently a **private** repository. Rather than commit symlinks that would
dangle in every clone, this file is the canonical list of what is inherited and
what an outside reader can actually reach.

## Publicly readable today

| Document | Location |
|---|---|
| Engineering philosophies | <https://github.com/plainsight-systems/.github/blob/main/engineering_philosophies.md> |

That public copy is a redaction: it omits a brand-specific section that is
internal by decision, and is otherwise the document this repo is held to.

Related public material from the same org, useful context but not the governance
docs themselves:

- C++ performance guidelines corpus (CC BY 4.0 / Apache-2.0) —
  <https://github.com/plainsight-systems/cpp-perf-guidelines>
- MCP servers exposing that corpus to agents —
  <https://github.com/plainsight-systems/mcp-servers>

## Inherited but not published

These bind this repo and are not currently readable outside the org. They are
listed so the governance surface is honest rather than implied.

| Document | Status |
|---|---|
| `cpp_architecture_playbook.md` | Not published. Candidate for publication — generic C++ guidance citing public sources. |
| `cpp_architecture_review.md` | Not published. Candidate for publication — generic reviewer checklist. |
| `cpp_performance_playbook.md` | Not published. Candidate for publication. Distinct from the public corpus above: this is the governance layer (budgets, measurement-first rules, acceptance criteria). |
| `cpp_performance_review.md` | Not published. Candidate for publication — generic reviewer checklist. |
| `product_memory_workflow.md` | Not published. Candidate, minor redaction needed. |
| `repo_creation_runbook.md` | Not published. Candidate, needs a public-repo branch in its symlink guidance. |
| `plainsight_policies.md` | **Internal.** Brand and entity boundaries, security-update and end-of-life policy. Not a public artifact. |

Whether to publish the candidates, and under what license, is deliberately
deferred until this repo gets a remote. See the parking lot in `QUEUE.md`.

## Local setup

If you have `plainsight-systems-governance` checked out at
`repositories/plainsight-systems-governance`, recreate the local symlinks:

```sh
./scripts/link-governance.sh
```

The links are gitignored. They are a local convenience for agents and editors,
not part of the repo.
