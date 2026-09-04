# GridFM PR Triage & Work Status

_Last updated: 2026-09-04 — snapshot for durability (pushed to remote so it survives a local crash/disk failure)._

This is a working-state dump across **gridfm-datakit** and **gridfm-graphkit**: what was done, what merged, what's still open, and the recommended next actions.

---

## 1. Completed this session

### gridfm-graphkit #115 — Docker image + docs (MERGED)
- Added a self-contained **CPU-only** `Containerfile` installing latest `gridfm-datakit` + `gridfm-graphkit` from PyPI.
- Key pins baked in (the interlocking constraints that make resolution land on the latest releases):
  - `torch<2.13` (required by graphkit 0.9.0)
  - `torch-scatter`/`torch-sparse` built against the exact installed torch ABI (avoids `undefined symbol`)
  - `juliapkg<0.1.24`, `juliacall<0.9.35`
  - Julia toolchain baked via `gridfm_datakit setup_pm`; `MLFLOW_ALLOW_FILE_STORE=true` set. Image ~4 GB.
- Also added: `.dockerignore`, `.devcontainer/devcontainer.json` (VS Code Dev Containers), `docs/install/docker.md` (build → smoke test → datagen→train hello-world → dev container), mkdocs nav entry, README pointer, and tiny case14 hello-world configs.
- **CI fix:** the graphkit test suite globs `examples/config/*.yaml` and validates each as a *graphkit* config. The datagen hello-world file is a *datakit* config (different schema) → moved it to `examples/config/datakit/hello_world_datagen_case14.yaml` (outside the non-recursive glob). pytests green after that.
- `pip-audit` remains red repo-wide on `mlflow 3.15.2 / CVE-2026-71211` — a base-dependency advisory, NOT introduced by this PR; suppression is handled main-side.
- Merged as squash commit `6be23bd`. Approved by Alban.
- Docs URL (after Pages rebuild): https://gridfm.github.io/gridfm-graphkit/install/docker/

### gridfm-datakit — merged a clean batch (6 PRs), in this order
`#79 → #77 → #78 → #81 → #82 → #83` — **all MERGED**.
- #79 — Add timeout & retries to PGLib download
- #77 — Fix PGLib sweep never testing OPF
- #78 — Remove pathlib/numba, depend on pyyaml
- #81 — Stop shadowing the `any` builtin in network
- #82 — Fix validate failing after a 2nd generation run
- #83 — Cache downloaded grids outside the installed package (was `CONFLICTING`; resolved automatically once #79 landed)
- All by `yemine0x01` (member), 7/7 CI passing.

---

## 2. Repo merge policies discovered (important operational notes)

### gridfm-datakit `main` branch protection
- `strict: true` (branches must be **up-to-date** before merge) → **each merge re-stales every other open PR** (`BEHIND`); must `gh pr update-branch` then merge, sequentially.
- Required status checks: **only `DCO`** (pytests etc. are NOT required).
- Required reviews: **none** (`required_pull_request_reviews: null`).
- `enforce_admins: true` (admins are NOT exempt from up-to-date requirement).
- Allowed merge method: **squash only** (`allow_merge_commit:false`, `allow_rebase_merge:false`).
- Auto-merge: **disabled** (`allow_auto_merge:false`) — so no queue; manual update→merge loop required.

### gridfm-graphkit `main` (from prior memory + observed)
- Squash-only; ruleset needs **1 approving review** (admin-bypass exists); DCO required; strict/up-to-date; no auto-merge.

---

## 3. Open PRs — full triage (as of 2026-09-04)

Generated via the headless `claude` CLI on **npzrl080.zurich.ibm.com** (see §5). Last activity on every PR was by `romeokienzler` (maintainer).

### gridfm-datakit — remaining open (after the merged batch above)

| PR | Title | Author | CI | State | Interaction | Recommendation |
|---|---|---|---|---|---|---|
| #40 | add precomputed_profile | hithuv (contrib) | 2✅/2❌ | BEHIND | waiting on author | Rebase + fix failing `pre-commit-run` & `security-test` |
| #44 | Update GH Actions for PyPI release | romeokienzler (maint) | 4✅ | open | waiting on review | Self-authored & green — needs 2nd maintainer review |
| #49 | Fix PV→PQ bus type conversion (bus row order) | yemine0x01 (member) | 7✅ | open | waiting on review | Green — route review to Alban, then merge |
| #53 | Update Python to 3.12.12 in CI | romeokienzler (maint) | 4✅/1❌/1⊘ | open | waiting on author | Fix failing `pre-commit-run` (pytests cancelled) |
| #64 | powsybl pf params for no single slack | tengxiangren (contrib) | 7✅ | open | waiting on review | Green — review, then merge |
| #71 | sync gen.vg to OPF bus voltage in pf_preproc | rosielickorish (contrib) | 1✅ | BEHIND | waiting on author | Rebase; only 1 check — confirm full CI runs |
| #74 | Add citation section to README | albanpuech (maint) | 6✅ | open | waiting on review | Green docs — quick review, then merge |
| #76 | Fix bandit scan skipping whole package | yemine0x01 (member) | 7✅ | open | waiting on review | Green but was BLOCKED — approve, then merge |
| #80 | Bound CI runs and fix pip cache key | yemine0x01 (member) | 7✅ | open | waiting on review | Green but was BLOCKED — approve, then merge |

_Note: #76/#80 showed `BLOCKED` earlier; datakit main has no required review, so re-verify current mergeStateStatus — they may just need `update-branch`._

### gridfm-graphkit — open (all green, gated on required review)

| PR | Title | Author | CI | Review | Recommendation |
|---|---|---|---|---|---|
| #99 | fix unit-consistency | naomi-simumba (member) | 11✅ | REVIEW_REQUIRED | Approve, then squash-merge |
| #105 | Restore masked bus/branch limit cols in PF output | naomi-simumba (member) | 11✅ | REVIEW_REQUIRED | Approve, then squash-merge |
| #110 | Add linear warmup for physics loss weight | guangzhao27 (contrib) | 11✅ | REVIEW_REQUIRED | Green & rebased — approve, then merge |

---

## 4. Pending decisions / next actions

- **AWAITING USER DECISION:** graphkit #99, #105, #110 are green but require an approving review. User was asked whether to (a) approve + squash-merge under their account, or (b) review/route to Alban themselves. **No approval submitted yet** — do not self-approve without confirmation.
- datakit review-gated greens ready once reviewed: #49, #64, #74 (and re-check #76, #80).
- datakit author-blocked (bounce back): #40 (rebase + 2 failing checks), #53 (failing CI, maintainer's own PR), #71 (rebase + CI looks incomplete).
- Remember: datakit strict mode → update-branch immediately before each merge; squash only.

## 5. Environment / how the triage was produced

- The PR triage table was generated by running the **headless Claude Code CLI** (`claude -p ... --dangerously-skip-permissions`) on **npzrl080.zurich.ibm.com** (SSH host `npzrl080.zurich.ibm.com`, user `rkie`).
- `gh` on np is **not** authenticated; local `gh` (this workstation) is. Token was forwarded to the np session via stdin into `GH_TOKEN` (never written to disk/argv/history).
- A transient `api.github.com` connection drop killed a monitoring loop mid-run once, but merges had already committed — always re-verify actual PR `state` after a network error rather than trusting loop output.
