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

# Contribution Checklist ✅

Before opening a PR, make sure you complete all steps:

### 1. Development setup
- [ ] Install dev and test dependencies:
  ```bash
  pip install -e ".[dev,test]"
  ```

* [ ] Install `torch-scatter` and `torch-sparse` separately (the correct wheel depends on your PyTorch and CUDA versions):
  ```bash
  TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.__version__ + ('+cpu' if torch.version.cuda is None else ''))")
  pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html
  pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html
  ```

  If `data.pyg.org` has no prebuilt wheel for your PyTorch version, build from source instead (this
  compiles against your installed torch, so it works with any version):
  ```bash
  pip install torch-scatter torch-sparse --no-build-isolation
  ```
  Building from source requires a C++ compiler (and a matching CUDA toolkit with
  `nvcc` for GPU builds).

* [ ] Install the git hooks (this repo runs pre-commit hooks at the **pre-push** stage):
  ```bash
  pre-commit install
  ```

* [ ] Create a new branch for your feature from `main`.


### 2. Code quality

* [ ] Write clear, readable code with self-explanatory variable names.
* [ ] Comment code where needed.
* [ ] Prefer the clearest implementation if multiple options have similar complexity.

### 3. Code style & documentation

* [ ] Add Google-style Python docstrings for all functions/classes.
* [ ] **Double-check that no debug prints or temporary files are left in the code.**
* [ ] Update documentation and `README.md` whenever relevant.

### 4. Configuration & dependencies

* [ ] Add any new dependencies to `pyproject.toml`.
* [ ] If introducing new parameters, update all relevant YAML files:

  * `examples/config`
  * `tests/config`

### 5. Testing
* [ ] Ask your favorite code assistant to identify bugs and edge cases.
* [ ] Add unit tests covering:

  * Core functionality of your changes.
  * Potential edge cases.
* [ ] Run all tests:

  ```bash
  MLFLOW_ALLOW_FILE_STORE=true pytest tests/
  ```
* [ ] Run integration tests, see [Running the Integration Tests](#running-the-integration-tests).

### 6. Pre-commit & linting

The hooks run automatically on `git push`. To run them manually before pushing:

* [ ] Run the hooks on all files:

  ```bash
  pre-commit run --all-files
  ```
* [ ] Fix any issues.
* [ ] Re-run the hooks after any code changes.

### 7. Documentation build

* [ ] Build docs and check rendering:

  ```bash
  mkdocs build
  mkdocs serve
  ```

### 8. Pull request

* [ ] **Rebase your branch onto the latest `main` before opening a PR.**
* [ ] **Sign off all commits** with the Developer Certificate of Origin (`git commit -s`). See [Developer Certificate of Origin signoff].
* [ ] Open a PR with a short description of your changes.
* [ ] Ensure code, tests, and documentation are clear and complete.
