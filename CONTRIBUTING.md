# Contributing to GridFM

Thank you for your interest in contributing to GridFM. This document explains our contribution process and procedures:

* [How to Contribute a Bug Fix or Change](#How-to-Contribute-a-Bug-Fix-or-Change)
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
your CPU/GPU, CUDA, and library versions), so they are not committed for you —
you must calibrate them on your own machine first.

**Calibrate before you change any code.** Run the calibration on a clean
checkout so the recorded bounds reflect the current behaviour, then make your
changes and run the tests to detect any drift they introduce:

1. On an unchanged checkout, calibrate a baseline on this machine:

   ```bash
   pytest integrationtests --calibrate 5 -s
   ```

   This runs the training a few times and writes per-metric bounds (plus an
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


[PEP 8]: https://peps.python.org/pep-0008/
[Apache 2.0 license]: LICENSE
[governance policies]: GOVERNANCE.md
[Technical Charter]: https://github.com/lf-energy/foundation/blob/main/project_charters/gridfm_charter.pdf
[copyright and license headers]: https://github.com/lf-energy/tac/blob/main/process/contribution_guidelines.md#license
[Developer Certificate of Origin signoff]: https://github.com/lf-energy/tac/blob/main/process/contribution_guidelines.md#contribution-sign-off
