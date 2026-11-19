"""make_figures.py on a small MLflow folder laid out like the paper evaluation runs."""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

SCRIPT = "scripts/state_estimation/make_figures.py"
FEATURES = ["Vm", "Va", "Pg", "Qg"]
GRIDS = ["case14", "case30", "case57", "case118"]
SWEEP = [f"swype_{m}_{o}" for m in [0.1, 0.2, 0.4] for o in [0.0, 0.1, 0.2]] + [
    "swype_0.4_0.3",
]
UNKNOWN = [f"unk_pert_perturbation_{s}_0" for s in [0.0, 0.1, 0.3]]


def write_experiment(root, exp_id, name):
    os.makedirs(root / exp_id, exist_ok=True)
    (root / exp_id / "meta.yaml").write_text(
        yaml.safe_dump({"name": name, "lifecycle_stage": "active"}),
    )


def write_run(root, exp_id, run_id, run_name, sets, start_time=1, parameters=False):
    """sets: {label: (median error for every feature, WLS convergence rate)}."""
    run = root / exp_id / run_id
    test = run / "artifacts" / "test"
    os.makedirs(test, exist_ok=True)
    meta = {"run_name": run_name, "lifecycle_stage": "active", "start_time": start_time}
    (run / "meta.yaml").write_text(yaml.safe_dump(meta))
    for label, (median, converged) in sets.items():
        df = pd.DataFrame(
            {
                "mean_grid": median,
                "std_grid": 0.1,
                "median_grid": median,
                "converged": converged,
            },
            index=FEATURES,
        )
        df.to_csv(test / f"case14_ieee_metrics_{label}.csv")
    if parameters:
        pe = pd.DataFrame(
            {
                "relmean_grid": ["tensor(0.0571)", "tensor(0.0490)"],
                "relmedian_grid": ["tensor(0.0560)", "tensor(0.0480)"],
                "relstd_grid": ["tensor(0.0130)", "tensor(0.0110)"],
                "meas_relmean_grid": ["tensor(0.1000)", "tensor(0.1000)"],
            },
            index=["r", "x"],
        )
        pe.to_csv(test / "case14_ieee_parameters_cnc.csv")


@pytest.fixture
def paper_runs(tmp_path):
    root = tmp_path / "mlruns"
    for g_idx, grid in enumerate(GRIDS):
        exp_id = str(100 + g_idx)
        write_experiment(root, exp_id, f"se_eval_{grid}_ieee")
        for r_idx, name in enumerate(SWEEP + UNKNOWN):
            # WLS error 4x the neural one. nc error 1.1x. 90% convergence.
            sets = {"c": (1.0, 0.9), "pp": (4.0, 0.9), "nc": (1.1, 0.9)}
            write_run(root, exp_id, f"r{r_idx}", f"{name}.yaml", sets)
    # An older run with the same name and different numbers must be ignored.
    write_run(
        root,
        "100",
        "old",
        "swype_0.1_0.0.yaml",
        {"c": (1.0, 0.5), "pp": (100.0, 0.5)},
        start_time=0,
    )

    write_experiment(root, "300", "se_eval_pe")
    for grid in GRIDS:
        write_run(
            root,
            "300",
            grid,
            f"{grid}seed0",
            {"cnc": (1.0, 1.0)},
            parameters=True,
        )
    return root


def test_make_figures_reads_paper_runs(paper_runs, tmp_path):
    out = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable, SCRIPT,
            "--mlflow_dir", str(paper_runs), "--out_dir", str(out),
            "--experiments", "case14=se_eval_case14_ieee", "case30=se_eval_case30_ieee",
            "case57=se_eval_case57_ieee", "case118=se_eval_case118_ieee",
            "--pe_experiment", "se_eval_pe",
        ],
        check=True,
    )  # fmt: skip
    expected = {
        "fig5_base_case.pdf",
        "fig6_unknown_perturbation.pdf",
        "fig19_non_observable.pdf",
        "fig5.csv",
        "fig6.csv",
        "fig19.csv",
        "parameter_estimation.csv",
    }
    assert set(os.listdir(out)) == expected

    fig5 = pd.read_csv(out / "fig5.csv", header=[0, 1], index_col=0)
    assert fig5.shape == (9, 12)
    assert list(fig5.index)[:3] == ["8, 0.0", "8, 0.1", "8, 0.2"]
    assert np.allclose(fig5["Vm"].values, 4.0)
    assert np.allclose(fig5["converged"].values, 90.0)

    fig19 = pd.read_csv(out / "fig19.csv", header=[0, 1], index_col=0)
    assert np.allclose(fig19["Vm"].values, 1.1)

    fig6 = pd.read_csv(out / "fig6.csv", header=[0, 1], index_col=0)
    assert list(fig6.index.astype(str)) == ["0.1", "0.3"]

    pe = pd.read_csv(out / "parameter_estimation.csv", index_col=0)
    assert np.isclose(pe.loc["case14seed0", "r_rel_error"], 0.0571)
    assert np.isclose(pe.loc["case14seed0", "r_input_rel_error"], 0.1)
    assert np.isclose(pe.loc["case14seed0", "r_reduction"], 1 - 0.0571 / 0.1)


def test_make_figures_from_saved_results(tmp_path):
    """The figures build from results/se_evaluation_results.csv."""
    out = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--results",
            "scripts/state_estimation/results/se_evaluation_results.csv",
            "--out_dir",
            str(out),
        ],
        check=True,
    )
    for name in ["fig5", "fig6", "fig19"]:
        table = pd.read_csv(out / f"{name}.csv", header=[0, 1], index_col=0)
        assert not table.isna().any().any(), name
    pe = pd.read_csv(out / "parameter_estimation.csv", index_col=0)
    assert len(pe) == 4
