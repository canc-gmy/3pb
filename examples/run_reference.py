"""Python API example: run the reference case and print key results.

Run inside the sandwich3pb conda environment:
    python examples/run_reference.py
"""

from sandwich3pb import load_config, run_case

if __name__ == "__main__":
    cfg = load_config("examples/case_glass_pvc.yaml")

    results = run_case(cfg, out_dir="results/glass_pvc_reference")

    s = results.summary
    print("\n--- extracted results ---")
    print(f"max force P        : {s['max_force_N']:.2f} N")
    print(f"max deflection w   : {s['max_deflection_mm']:.4f} mm")
    print(f"force gradient dP/dw: {s['force_gradient_N_per_mm']:.2f} N/mm "
          f"(R2 = {s['gradient_fit_r2']:.4f})")
    print(f"apparent rigidity D: {s['apparent_flexural_rigidity_Nmm2']:.4g} N.mm2")
    print(f"layup rigidity EI  : {s['layup_flexural_rigidity_Nmm2']:.4g} N.mm2")
    print(f"ratio D/EI         : {s['rigidity_ratio_FE_over_layup']:.3f}")
    print(f"force balance resid: {s['force_balance_residual']:.2e}")
    print(f"max Tsai-Wu (faces): {s['max_tsai_wu_faces']}")
    print(f"core shear util.   : {s['max_core_shear_ratio']}")
    print(f"core crushing util.: {s['max_core_crushing_ratio']}")
    print("\nreport written to results/glass_pvc_reference/report.(md|html|pdf)")
