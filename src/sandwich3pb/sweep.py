"""Parameter sweeps: expand list-valued parameters, run cases, collect results.

A sweep YAML file is a normal case file except that any parameter may be a
*list* (e.g. ``thickness: [5.0, 8.0, 12.0]`` for the core). All combinations
are expanded into a grid; each case runs in a subprocess via
``python -m sandwich3pb run`` so a crash in one case never kills the sweep.

Required keys:
    sweep:
        output_dir: results/sweepname     (optional, default results/sweep)
        parallel: 1                       (optional, processes > 1)
Everything else follows the case-file schema with list values allowed.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import yaml


def _find_lists(raw: Any, prefix: str = ""
                ) -> List[Tuple[str, List[Any]]]:
    """Locate (dotted-path, list) pairs inside the nested config dict.

    Traverses dicts *and* lists, so parameters nested inside the stackup
    (e.g. ``stackup.1.thickness``) are sweepable. A list is sweepable when
    it holds only scalars; structural lists (of dicts) are recursed into
    with their indices as path components. The top-level ``sweep``
    metadata section is not sweepable.
    """
    found: List[Tuple[str, List[Any]]] = []
    if isinstance(raw, dict):
        pairs = list(raw.items())
    elif isinstance(raw, (list, tuple)):
        pairs = [(str(i), v) for i, v in enumerate(raw)]
    else:
        return found

    for key, value in pairs:
        path = f"{prefix}{key}"
        if isinstance(value, list):
            is_scalar_list = bool(value) and all(
                isinstance(v, (int, float, str, bool)) or v is None
                for v in value
            )
            if is_scalar_list and path != "sweep":
                found.append((path, value))
            else:
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        found.extend(
                            _find_lists(item, prefix=f"{path}.{i}.")
                        )
        elif isinstance(value, dict):
            found.extend(_find_lists(value, prefix=f"{path}."))
    return found


def _set_path(raw: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node: Any = raw
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node[p]
    if isinstance(node, list):
        node[int(parts[-1])] = value
    else:
        node[parts[-1]] = value


def expand_cases(raw: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Expand list-valued parameters into named single cases."""
    lists = _find_lists(raw)
    if not lists:
        return [(raw.get("name", "case"), raw)]

    names: List[str] = []
    keys = [k for k, _ in lists]
    combos = list(itertools.product(*[vals for _, vals in lists]))

    sweep_cfg = raw.get("sweep", {}) or {}
    base_name = raw.get("name", "case")

    cases: List[Tuple[str, Dict[str, Any]]] = []
    for i, combo in enumerate(combos):
        case = json.loads(json.dumps(raw))  # deep copy
        for key, value in zip(keys, combo):
            _set_path(case, key, value)
        combo_name = "_".join(
            f"{k.split('.')[-1]}{str(v).replace('.', 'p')}"
            for k, v in zip(keys, combo)
        )
        name = f"{base_name}__{combo_name}"
        case["name"] = name
        case["output_dir"] = sweep_cfg.get("output_dir", "results/sweep")
        cases.append((name, case))
    return cases


def run_sweep(sweep_file: str, verbose: bool = True) -> str:
    """Run all expanded cases; collect summaries into sweep_results.csv."""
    with open(sweep_file) as f:
        raw = yaml.safe_load(f) or {}

    sweep_cfg = raw.get("sweep", {}) or {}
    out_root = sweep_cfg.get("output_dir", "results/sweep")
    parallel = int(sweep_cfg.get("parallel", 1))
    os.makedirs(out_root, exist_ok=True)

    cases = expand_cases(raw)
    if verbose:
        print(f"[sweep] {len(cases)} case(s) -> {out_root}")

    # write each case yaml and run it as a subprocess
    procs: List = []
    case_files: List[str] = []
    names: List[str] = []
    for name, case in cases:
        case_path = os.path.join(out_root, f"{name}.yaml")
        with open(case_path, "w") as f:
            yaml.safe_dump(case, f, sort_keys=False)
        case_files.append(case_path)
        names.append(name)

    def start(case_file: str):
        cmd = [sys.executable, "-m", "sandwich3pb", "run", case_file]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)

    pending: List = []
    for i, case_file in enumerate(case_files):
        pending.append((names[i], start(case_file)))
        while len(pending) >= max(1, parallel):
            name, proc = pending.pop(0)
            _, err = proc.communicate()
            if proc.returncode != 0 and verbose:
                print(f"[sweep] FAILED {name}:\n{err.decode()}", file=sys.stderr)

    for name, proc in pending:
        _, err = proc.communicate()
        if proc.returncode != 0 and verbose:
            print(f"[sweep] FAILED {name}:\n{err.decode()}", file=sys.stderr)

    # ---- collect -------------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    for name in names:
        summary_path = os.path.join(out_root, name, "summary.json")
        if not os.path.exists(summary_path):
            rows.append({"case": name, "status": "failed"})
            continue
        with open(summary_path) as f:
            s = json.load(f)
        rows.append(
            {
                "case": name,
                "status": "ok" if s.get("all_steps_converged") else "partial",
                "max_force_N": s.get("max_force_N"),
                "max_deflection_mm": s.get("max_deflection_mm"),
                "force_gradient_N_per_mm": s.get("force_gradient_N_per_mm"),
                "apparent_flexural_rigidity_Nmm2":
                    s.get("apparent_flexural_rigidity_Nmm2"),
                "layup_flexural_rigidity_Nmm2":
                    s.get("layup_flexural_rigidity_Nmm2"),
                "max_tsai_wu_faces": s.get("max_tsai_wu_faces"),
                "max_core_shear_ratio": s.get("max_core_shear_ratio"),
                "max_core_crushing_ratio": s.get("max_core_crushing_ratio"),
            }
        )

    csv_path = os.path.join(out_root, "sweep_results.csv")
    fieldnames = sorted({k for row in rows for k in row})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"[sweep] results table: {csv_path}")
        for row in rows:
            print(
                f"  {row['case']}: P_max="
                f"{row.get('max_force_N', '—')} N, "
                f"D_FE={row.get('apparent_flexural_rigidity_Nmm2', '—')}"
            )
    return csv_path
