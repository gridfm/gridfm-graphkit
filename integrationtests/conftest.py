import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--calibrate",
        type=int,
        default=0,
        help="Run training N times to collect metric mean/std for range calibration. "
        "Skips metric range assertions. Example: pytest --calibrate 5",
    )
    parser.addoption(
        "--ci",
        type=float,
        default=0.995,
        help="Confidence interval level for calibration stats (default 0.995). "
        "Example: pytest --calibrate 5 -s --ci 0.995",
    )
    parser.addoption(
        "--pad",
        type=float,
        default=0.01,
        help="Relative padding added to each calibrated bound as a floor on the "
        "margin of error (default 0.01 = 1%%). Absorbs residual same-machine "
        "jitter; metrics whose mean is 0 stay exactly (0, 0).",
    )


@pytest.fixture
def calibrate_runs(request):
    """Number of calibration runs requested via --calibrate (0 = normal test mode)."""
    return request.config.getoption("--calibrate")


@pytest.fixture
def ci_level(request):
    """Confidence interval level requested via --ci (default 0.995)."""
    return request.config.getoption("--ci")


@pytest.fixture
def calibrate_pad(request):
    """Relative padding for calibrated bounds requested via --pad (default 0.01)."""
    return request.config.getoption("--pad")
