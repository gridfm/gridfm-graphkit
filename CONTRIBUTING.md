# Contributing to GridFM

Thank you for your interest in contributing to GridFM. This document explains our contribution process and procedures:

* [How to Contribute a Bug Fix or Change](#How-to-Contribute-a-Bug-Fix-or-Change)
* [Running the Integration Tests](#Running-the-Integration-Tests)
* [Development Workflow](#Development-Workflow)
* [Coding Style](#Coding-Style)

For a description of the roles and responsibilities of the various members of the GridFM community, see the [governance policies], and for further details, see the project's [Technical Charter]. Briefly, Contributors are anyone who submits content to the project, Committers review and approve such submissions, and the Technical Steering Committee provides general project oversight.

If you just need help or have a question, refer to [SUPPORT.md](SUPPORT.md).

## How to Contribute a Bug Fix or Change

To contribute code to the project, first read over the [governance policies] page to understand the roles involved.

Each contribution must meet the [PEP 8] and include..

* Tests and documentation to explain the functionality.
* Any new files have [copyright and license headers]
* A [Developer Certificate of Origin signoff].
* Submitted to the project as a pull request.

GridFM is licensed under the [Apache 2.0 license]. Contributions should abide by that standard license.

Project committers will review the contribution in a timely manner, and advise of any changes needed to merge the request.

## Running the Integration Tests

The integration tests in `integrationtests/` assert that training metrics fall
within calibrated bounds. These bounds are **machine-specific** (they depend on
your CPU/GPU, CUDA, and library versions), so the baseline
(`integrationtests/calibration_baseline.json`) is **not committed** — it is
git-ignored and you calibrate your own on your machine first.

**Prerequisite.** The tests train through MLflow; if you don't have an MLflow
tracking server, opt into the local file store first:

```bash
export MLFLOW_ALLOW_FILE_STORE=true
```

**Calibrate before you change any code.** Run the calibration on a clean
checkout so the recorded bounds reflect the current behaviour, then make your
changes and run the tests to detect any drift they introduce:

1. On an unchanged checkout, calibrate a baseline on this machine:

   ```bash
   pytest integrationtests --calibrate -s
   ```

   This runs training 5 times and writes per-metric bounds (plus a per-test
   environment fingerprint) to `integrationtests/calibration_baseline.json`.

2. Make your code changes.

3. Run the integration tests (no `--calibrate`) to assert against the baseline
   you calibrated in step 1:

   ```bash
   pytest integrationtests -s
   ```

If you calibrate *after* changing the code, the baseline will simply encode your
changed behaviour and the tests can no longer catch regressions — always
calibrate on the unchanged code first.

### Calibration options

Passed to `pytest`:

| Flag | When omitted | Meaning |
|------|--------------|---------|
| `--calibrate [N]` | assert mode (train once, check bounds) | `--calibrate` alone → 5 calibration runs; `--calibrate N` → `N` runs. Both write bounds and skip assertions. |
| `--ci C` | `0.995` | Confidence level for the Student-t interval (calibration mode only). |
| `--pad P` | `0.01` | Relative floor on each bound's half-width (`P * |mean|`) during calibration. Absorbs residual same-machine jitter; metrics whose mean is exactly `0` stay exactly `(0, 0)`. |

### If the environment fingerprint doesn't match

Calibrated bounds depend on your hardware (CPU/GPU), CUDA/cuDNN, and library
versions (notably the PyTorch version), so each test's bounds are stamped with
their own environment fingerprint under ``fingerprints`` in the baseline. When
you run the assertion tests, a fingerprint that differs from the one recorded
for that test emits a **loud warning but does not fail the run** — it lists
every differing field.

A mismatch means the recorded bounds may not hold in your current environment
(for example, after a PyTorch upgrade the metrics can shift). When you see this
warning, **recalibrate on the unchanged code in your current environment before
trusting the assertions**:

```bash
export MLFLOW_ALLOW_FILE_STORE=true
pytest integrationtests --calibrate -s
```


[PEP 8]: https://peps.python.org/pep-0008/
[Apache 2.0 license]: LICENSE
[governance policies]: GOVERNANCE.md
[Technical Charter]: https://github.com/lf-energy/foundation/blob/main/project_charters/gridfm_charter.pdf
[copyright and license headers]: https://github.com/lf-energy/tac/blob/main/process/contribution_guidelines.md#license
[Developer Certificate of Origin signoff]: https://github.com/lf-energy/tac/blob/main/process/contribution_guidelines.md#contribution-sign-off
