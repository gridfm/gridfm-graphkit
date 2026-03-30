
from __future__ import annotations
import os
import sys
from typing import Any, Dict, List
import yaml
import itertools

SCHEMA = {
    "model12": bool,
    "randomMask": bool,
    "removeNan": bool,
    "graphWise": bool,
    "nodeWise": bool,
    "topologyOnly": bool,
    "error_based": bool,
    "ones": bool,
    "generate": bool,
    "ablation": bool,
    "modelComparison": bool,
    "leaveOneOut": bool,
    "featureNames": list,  # list[str]
}


def _validate_fixed(fixed: Dict[str, Any]) -> None:
    for key, typ in SCHEMA.items():
        if key not in fixed:
            raise ValueError(f"Missing fixed key: '{key}'")
        val = fixed[key]
        if typ is list:
            if not isinstance(val, list):
                raise TypeError(f"'{key}' must be a list; got {type(val).__name__}")
            for i, el in enumerate(val):
                if not isinstance(el, str):
                    raise TypeError(f"'{key}[{i}]' must be str; got {type(el).__name__}")
        else:
            if not isinstance(val, typ):
                raise TypeError(f"'{key}' must be {typ.__name__}; got {type(val).__name__}")

def _resolve_runtime_args() -> Tuple[str]:
    """
    Decide which experiment YAML to load at runtime.
    Precedence:
      CLI: --file <path>
      ENV: EXP_FILE
      Default: exp_iterate_ones_graphwise.yaml
    """
    args = sys.argv[1:]
    def argval(flag: str, default: str) -> str:
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args):
                return args[i + 1]
            raise SystemExit(f"Usage: {flag} <value>")
        return default
    path = argval("--file", os.getenv("EXP_FILE", "exp_iterate_ones_graphwise.yaml"))
    path = 'ExpConfigs/'+path
    if not 'yaml' in path:
        path = path+'.yaml'
    return (path,)

def _expand_matrix(fixed: Dict[str, Any], sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand cartesian product of sweep values on top of fixed.
    Supports optional: sweep.exclude (list of dicts), sweep.replicates (int).
    """
    exclude = sweep.get("exclude", []) or []
    replicates = int(sweep.get("replicates", 1))
    # Collect keys and value lists (only keys that actually have a list of values)
    keys = [k for k, v in sweep.items() if k not in ("exclude", "replicates") and isinstance(v, list)]
    values_lists = [sweep[k] for k in keys]

    runs: List[Dict[str, Any]] = []
    for combo in itertools.product(*values_lists):
        overrides = dict(zip(keys, combo))
        # skip excluded combos
        if any(all(overrides.get(k) == ex.get(k) for k in ex.keys()) for ex in exclude):
            continue
        for r in range(replicates):
            run_cfg = {**fixed, **overrides}
            run_cfg["_name"] = _name_from_overrides(overrides, index=len(runs), replicate=r)
            run_cfg["_index"] = len(runs)
            run_cfg["_replicate"] = r
            runs.append(run_cfg)
    return runs

def _expand_combinations(fixed: Dict[str, Any], combos: List[Dict[str, Any]], replicates: int = 1) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for i, overrides in enumerate(combos):
        for r in range(replicates):
            run_cfg = {**fixed, **overrides}
            run_cfg["_name"] = _name_from_overrides(overrides, index=i, replicate=r)
            run_cfg["_index"] = len(runs)
            run_cfg["_replicate"] = r
            runs.append(run_cfg)
    return runs

def _name_from_overrides(overrides: Dict[str, Any], index: int, replicate: int) -> str:
    parts = [f"{k}={str(v).lower()}" for k, v in sorted(overrides.items())]
    base = " | ".join(parts) if parts else f"run{index}"
    return f"{base}#rep{replicate}"

def load_runs(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Load a single experiment YAML and return a list of run configs
    expanded from either a 'sweep' matrix or explicit 'combinations'.
    """
    print(path)
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    name = doc.get("name") or os.path.splitext(os.path.basename(path))[0]

    fixed = doc.get("fixed", {})
    _validate_fixed(fixed)

    runs: List[Dict[str, Any]]
    if "sweep" in doc and isinstance(doc["sweep"], dict):
        runs = _expand_matrix(fixed, doc["sweep"])
    elif "combinations" in doc and isinstance(doc["combinations"], list):
        replicates = int(doc.get("replicates", 1))
        runs = _expand_combinations(fixed, doc["combinations"], replicates=replicates)
    else:
        # No sweep/combinations: single run (fixed only)
        run_cfg = dict(fixed)
        run_cfg["_name"] = "fixed-only#rep0"
        run_cfg["_index"] = 0
        run_cfg["_replicate"] = 0
        runs = [run_cfg]

    return name, runs

# Example entrypoint: iterate runs
if __name__ == "__main__":
    (file_path,) = _resolve_runtime_args()
    exp_name, runs = load_runs(file_path)
    print(f"Experiment '{exp_name}' -> {len(runs)} runs")
    for i, cfg in enumerate(runs):
        print(f"[{i}] {cfg['_name']}")
