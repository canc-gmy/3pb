"""Python API example: run the reference case and print key results.

Run inside the sandwich3pb conda environment:
    python examples/run_reference.py

Results are printed in the unit system the case file declares, so this
script prints m / Pa / N for the SI reference case and mm / MPa / N for a
case written with ``units: mm_n_mpa``.
"""

from sandwich3pb import load_config, run_case
from sandwich3pb.units import (
    FORCE,
    GRADIENT,
    LENGTH,
    NONE,
    RIGIDITY,
    fmt_qty,
)

if __name__ == "__main__":
    cfg = load_config("examples/case_glass_pvc.yaml")

    results = run_case(cfg, out_dir="results/glass_pvc_reference")

    s = results.summary
    u = cfg.unit_system

    def q(key, dim):
        return fmt_qty(s[key], u, dim)

    print(f"\n--- extracted results ({u.name}: length {u.length}, "
          f"stress {u.stress}, force {u.force}) ---")
    print(f"max force P        : {q('max_force_N', FORCE)}")
    print(f"max deflection w   : {q('max_deflection_mm', LENGTH)}")
    print(f"force gradient dP/dw: {q('force_gradient_N_per_mm', GRADIENT)} "
          f"(R2 = {fmt_qty(s['gradient_fit_r2'], u, NONE)})")
    print(f"apparent rigidity D: {q('apparent_flexural_rigidity_Nmm2', RIGIDITY)}")
    print(f"layup rigidity EI  : {q('layup_flexural_rigidity_Nmm2', RIGIDITY)}")
    print(f"ratio D/EI         : {fmt_qty(s['rigidity_ratio_FE_over_layup'], u, NONE)}")
    print(f"force balance resid: {fmt_qty(s['force_balance_residual'], u, NONE)}")
    print(f"max Tsai-Wu (faces): {fmt_qty(s['max_tsai_wu_faces'], u, NONE)}")
    print(f"core shear util.   : {fmt_qty(s['max_core_shear_ratio'], u, NONE)}")
    print(f"core crushing util.: {fmt_qty(s['max_core_crushing_ratio'], u, NONE)}")
    print("\nreport written to results/glass_pvc_reference/report.(md|html|pdf)")
