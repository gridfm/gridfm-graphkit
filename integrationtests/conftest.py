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


@pytest.fixture
def calibrate_runs(request):
    """Number of calibration runs requested via --calibrate (0 = normal test mode)."""
    return request.config.getoption("--calibrate")


@pytest.fixture
def ci_level(request):
    """Confidence interval level requested via --ci (default 0.995)."""
    return request.config.getoption("--ci")

