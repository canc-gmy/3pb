"""High-level pipeline: run a case from a Config (or YAML file)."""

from __future__ import annotations

from typing import Optional, Union

from .config import Config, load_config
from .results import CaseResults
from .solver import ThreePointBendingSolver


def run_case(case: Union[str, Config], out_dir: Optional[str] = None,
             verbose: bool = True, write: bool = True) -> CaseResults:
    """Run a three-point-bending case end to end.

    Parameters
    ----------
    case:
        Path to a YAML case file or a prepared :class:`Config`.
    out_dir:
        Output directory override; defaults to ``cfg.output_dir/<case name>``.
    verbose:
        Print progress to stdout.
    write:
        Write all artifacts (CSV/JSON/XDMF/plots/report) to ``out_dir``.

    Returns
    -------
    CaseResults
        Extracted results; ``results.summary`` holds the key quantities and
        ``results.write_outputs(dir)`` can regenerate the artifacts.
    """
    cfg = load_config(case) if isinstance(case, str) else case
    solver = ThreePointBendingSolver(cfg, verbose=verbose)
    solution = solver.solve()
    results = CaseResults.from_solution(solution)

    if write:
        target = out_dir or f"{cfg.output_dir}/{cfg.name}"
        paths = results.write_outputs(target)
        if verbose:
            print(f"[sandwich3pb] outputs written to {target}")
            for key in ("report_md", "report_html", "report_pdf"):
                if key in paths:
                    print(f"[sandwich3pb]   {key}: {paths[key]}")
    return results
