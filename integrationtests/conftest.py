import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--calibrate",
        type=int,
        nargs="?",
        const=5,
        default=0,
        help="Calibrate metric bounds instead of asserting. Omit for normal assert mode "
        "(default 0). --calibrate alone runs 5 calibration passes; --calibrate N runs N. "
        "Example: pytest integrationtests --calibrate -s",
    )
    parser.addoption(
        "--ci",
        type=float,
        default=0.995,
        help="Confidence interval level for calibration stats (default 0.995). "
        "Example: pytest integrationtests --calibrate -s --ci 0.995",
    )
    parser.addoption(
        "--pad",
        type=float,
        default=0.01,
        help="Relative padding added to each calibrated bound as a floor on the "
        "margin of error (default 0.01 = 1%%). Absorbs residual same-machine "
        "jitter; metrics whose mean is 0 stay exactly (0, 0).",
    )
    parser.addoption(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs per run for the integration tests (default 20). "
        "Lower values speed up the assert gate; the calibration baseline must be "
        "recalibrated with the same value it is asserted against.",
    )
    parser.addoption(
        "--retries",
        type=int,
        default=5,
        help="Max extra training attempts when a metric falls outside its "
        "calibrated bounds (default 5). Each retry is a full training run, so a "
        "flaky assertion can multiply wall-clock by up to --retries+1. Set to 0 "
        "to fail fast on the first out-of-bounds run.",
    )
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Drop deterministic-mode knobs (--deterministic warn and "
        "CUBLAS_WORKSPACE_CONFIG) to allow faster nondeterministic GPU kernels. "
        "Only use when the calibrated bounds are wide enough to tolerate "
        "nondeterministic numerics; the baseline should be calibrated with --fast too.",
    )
    parser.addoption(
        "--preprocess-check",
        choices=["on", "off"],
        default="on",
        help="Re-run the raw->processed preprocessing from the downloaded raw/ "
        "tables and assert the regenerated tensors match the pre-generated "
        "processed/ tensors shipped in the zip (default on). This exercises the "
        "real preprocessing pipeline end to end. Pass 'off' for a lightweight run "
        "(e.g. CI) that trains directly on the shipped processed tensors.",
    )


@pytest.fixture
def calibrate_runs(request):
    """Number of calibration runs via --calibrate (Omitted = assert mode; flag alone = 5)."""
    return request.config.getoption("--calibrate")


@pytest.fixture
def ci_level(request):
    """Student-t confidence level for calibrated bounds via --ci (0.995 if omitted)."""
    return request.config.getoption("--ci")


@pytest.fixture
def calibrate_pad(request):
    """Relative half-width floor for calibrated bounds via --pad (0.01 if omitted)."""
    return request.config.getoption("--pad")


@pytest.fixture
def train_epochs(request):
    """Training epochs per run via --epochs (20 if omitted)."""
    return request.config.getoption("--epochs")


@pytest.fixture
def max_retries(request):
    """Max out-of-bounds retry attempts via --retries (5 if omitted)."""
    return request.config.getoption("--retries")


@pytest.fixture
def fast_mode(request):
    """Whether to drop deterministic knobs for faster kernels via --fast (False if omitted)."""
    return request.config.getoption("--fast")


@pytest.fixture
def preprocess_check(request):
    """Whether to re-run preprocessing from raw and compare to shipped tensors via
    --preprocess-check ('on' if omitted; pass 'off' for a lightweight CI run)."""
    return request.config.getoption("--preprocess-check") == "on"
