# sandwich3pb — 3D Three-Point Bending of Sandwich Composites

A Python package for **finite-element simulation of three-point bending of
composite sandwich beams** with FEniCSx/DOLFINx. It builds a layered 3D hex
mesh from a stackup definition, applies load and supports through **rigid
roller contact** (penalty / augmented Lagrangian), and extracts engineering
results into a **Markdown/HTML/PDF report** with the layup table, flexural
rigidity, max force, max deflection, force gradient and failure indices.

## Features

- **3D solid model** — structured hex8 mesh generated natively (no gmsh):
  facesheets and core share interface nodes (perfect bonding). Optional
  **quadratic displacement** (`element_order: 2`, Q2 on the hex8 geometry,
  superparametric) removes the transverse-shear locking of linear hex8
  elements — recommended for slender beams; verified against Euler–Bernoulli
  and Allen sandwich benchmarks.
- **Stackup-driven** — layers with material, thickness and *fibre orientation*
  (orthotropic plies rotated about the width axis); the same stackup drives
  both the mesh and the report's layup table.
- **Rigid-roller contact** — cylindrical load roller (displacement controlled)
  and support rollers; penalty enforcement with an augmented-Lagrangian outer
  loop for penetration-free solutions.
- **Result extraction** — load–deflection curve, force gradient dP/dw
  (robust linear fit), apparent flexural rigidity D = (dP/dw)·L³/48, analytic
  layup rigidity EI for cross-checking, support reactions with force-balance
  residual, Tsai-Wu indices for the faces, core shear/crushing utilization,
  through-thickness stress profiles at mid-span, XDMF field export for ParaView.
- **Interfaces** — Python API, YAML case files, CLI (`run`, `sweep`, `report`,
  `plot`), parameter sweeps with a collected CSV table.

## Installation

```bash
git clone https://github.com/canc-gmy/3pb.git
cd 3pb
```

FEniCSx must come from conda-forge (no PyPI wheels):

```bash
conda env create -f environment.yml
conda activate sandwich3pb
pip install -e .
```

## Quickstart

### 1. Run a case

```bash
sandwich3pb run examples/case_glass_pvc.yaml
```

Outputs land in `results/glass_pvc_reference/`:

| file | content |
|---|---|
| `report.md` / `report.html` / `report.pdf` | the results report (layup, rigidity, max force/deflection, gradient, failure) |
| `load_deflection.csv` | step history: travel, force, deflection |
| `profile_midspan.csv` | σxx, τxz through the thickness at mid-span |
| `profile_span.csv` | σxx, τxz along the span, sampled at each layer's mid-thickness |
| `summary.json` | every extracted quantity + the effective config |
| `plots/*.{svg,pdf,png}` | vector figures: stackup, load–deflection, through-thickness stress profile, stress along the span |
| `fields.xdmf` | displacement + stress fields for ParaView |

### 2. Python API

```python
from sandwich3pb import load_config, run_case

results = run_case("examples/case_glass_pvc.yaml")
s = results.summary
print(s["max_force_N"], s["force_gradient_N_per_mm"],
      s["apparent_flexural_rigidity_Nmm2"])
```

### 3. Parameter sweep

```bash
sandwich3pb sweep examples/sweep_core_thickness.yaml
# -> results/core_thickness_sweep/sweep_results.csv
```

Any parameter in the case file may be given as a list — including values
nested in the stackup (e.g. a core `thickness: [15.0e-3, 20.0e-3, 25.0e-3]`); the
sweep runner expands the grid and runs each case in a subprocess.

## Case file reference (YAML)

```yaml
units: si          # REQUIRED: si (m, Pa, N) or mm_n_mpa (mm, MPa, N)
name: my_case
output_dir: results

geometry:
  length: 0.300    # overall beam length (m)
  span: 0.200      # distance between support rollers (m)
  width: 0.040     # beam width (m)

materials:         # stresses in Pa under `units: si`
  glass_epoxy:
    kind: orthotropic
    E1: 3.9e10     # fibre direction
    E2: 9.0e9
    E3: 9.0e9
    nu12: 0.28
    nu13: 0.28
    nu23: 0.40
    G12: 3.8e9
    G13: 3.8e9
    G23: 3.2e9
    Xt: 9.0e8      # allowables (Tsai-Wu), in Pa
    Xc: 7.5e8
    Yt: 4.5e7
    Yc: 1.4e8
    S: 5.5e7
  pvc_foam:
    kind: isotropic
    E: 7.5e7       # Pa
    nu: 0.30
    shear: 8.0e5       # core shear allowable (Pa)
    compression: 1.1e6  # core crushing allowable (Pa)

stackup:           # bottom -> top, thicknesses in m
  - material: glass_epoxy
    thickness: 1.5e-3
    fibre_orientation: 0.0   # deg, rotation about the width axis
    role: face
  - material: pvc_foam
    thickness: 20.0e-3
    role: core
  - material: glass_epoxy
    thickness: 1.5e-3
    fibre_orientation: 0.0
    role: face

mesh:
  elements_x: 60
  elements_w: 4
  elements_per_layer: {face: 2, core: 10}
  element_order: 1    # 1 = hex8 (linear); 2 = quadratic Q2 (recommended)

contact:
  penalty: 1.0e15            # N/m^3
  roller_radius_load: 10.0e-3
  roller_radius_support: 5.0e-3
  augmented_lagrangian:
    enabled: true
    max_outer: 8
    tol: 1.0e-6              # max allowed penetration (m)
    update_rate: 10.0

loading:
  max_indentation: 2.0e-3    # roller travel (m)
  n_steps: 10

solver:
  rtol: 1.0e-8
  max_newton_iterations: 30
```

### Notes on modeling assumptions

- Linear elastic, small strain. Perfect face–core bonding (shared nodes).
- One-way rigid-deformable frictionless contact; supports act as cylindrical
  rigid surfaces from below.
- BCs: `uy = 0` at both end cross-sections; the axial direction stays free —
  frictionless rollers cannot carry axial force, and pinning `ux` at the ends
  would clamp the beam slope (`ux ~ -z w'`) and stiffen the span. The
  rigid-body modes (x-translation, and z before the supports engage) are
  anchored by negligible ground springs (`1e-9` x mean diagonal stiffness);
  force balance closes to <0.1%.
- Optional `half_model: true` runs x in [0, length/2]: symmetry plane at the
  load roller (x = 0), one support at x = span/2, overhang kept. Reported
  forces are doubled so they refer to the full beam (half/full agreement is
  verified to ~0.3%).
- `element_order: 2` solves with quadratic Lagrange displacements on the hex8
  geometry. At `span/depth ~ 8` this changes the stiffness by up to ~2x
  versus locked hex8 runs.
- Scalar YAML values such as `penalty: 1.0e6` (which YAML 1.1 parses as a
  string) are coerced to numbers automatically.

## Testing

```bash
pytest                # includes analytic benchmarks
pytest -k bench -m "not slow"   # fast subset
```

Benchmarks: isotropic beam vs. Euler–Bernoulli δ = PL³/48EI; sandwich vs.
Allen bending+shear theory; force balance and symmetry checks.

## Units

A case file declares its unit system with a required `units:` key:

| `units:` | length | stress | force | use |
|---|---|---|---|---|
| `si` (default) | m | Pa | N | new cases |
| `mm_n_mpa` | mm | MPa | N | legacy convention |

The declaration is **required**: SI and mm differ by 1000x in length and
1e6 in stress, so guessing would silently produce a beam three orders of
magnitude too small. Everything you see — report tables, figure axis
labels, CSV columns and values, `summary.json`, the `case_used.yaml` echo —
is written back in the units the case file declared.

Internally the solver always works in mm / N / MPa; conversion happens once,
at the config boundary.

Report quantities are written as TeX. Markdown keeps the `$...$` markup
(GitHub and pandoc typeset it), the PDF renders it with matplotlib's
mathtext, and the HTML inlines each snippet as a base64 SVG — so all three
show the same mathematics with no LaTeX install, no CDN and no JavaScript.

## License

MIT — see [LICENSE](LICENSE).
