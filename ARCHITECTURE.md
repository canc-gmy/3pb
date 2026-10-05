# sandwich3pb — Architecture Summary

**Purpose**: 3D finite-element simulation of three-point bending of composite sandwich beams using FEniCSx/DOLFINx.

---

## High-Level Data Flow

```
YAML case file
     │
     ▼
load_config() ──► Config (dataclass, internal units: mm/N/MPa)
     │
     ▼
ThreePointBendingSolver(cfg) ──► builds mesh, assembles K once, factorizes tangent
     │
     ▼
solve() ──► runs load steps (Newton + optional Augmented Lagrangian)
     │
     ▼
Solution (u_dofs, history, mesh_data, contact, FE points)
     │
     ▼
CaseResults.from_solution() ──► extracts engineering quantities
     │
     ▼
write_outputs() ──► CSV, JSON, XDMF, plots, Markdown/HTML/PDF/LaTeX report
```

---

## Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `config.py` | YAML loading, unit conversion (SI ↔ mm/N/MPa), validation, dataclass schemas |
| `geometry.py` | Structured hex mesh generation, layer/surface tagging, FE point set (Q1/Q2), dolfinx mesh creation, roller axes |
| `materials.py` | Constitutive matrices (iso/orthotropic), fibre rotation (full 4th-order tensor), Tsai-Wu, core failure criteria |
| `contact.py` | Rigid-roller frictionless contact, penalty + AL, nodal forces, gaps, tributary areas |
| `solver.py` | Model build, global stiffness assembly (Voigt, layer-wise), single factorization + Woodbury tangent, Newton/AL loops |
| `postprocess.py` | Nodal stress recovery (per layer, no mixing), mid-span & span profiles, failure indices, analytic layup rigidity |
| `results.py` | CaseResults container, gradient fit, summary dict, failure onset detection, output orchestration |
| `io.py` | CSV/JSON/XDMF/plot writers |
| `report.py` | Markdown/HTML/PDF/LaTeX report generation (TeX math via mathtext + inline SVG for HTML; `report.tex` compiles with pdflatex) |
| `cli.py` | `sandwich3pb run/sweep/report/plot` commands |
| `run.py` | High-level `run_case()` pipeline function |
| `units.py` | Unit system definitions and scaling |

---

## Key Design Decisions

### Mesh & Discretization
- **Structured hex8 grid** generated natively (no Gmsh). Layers share interface nodes → perfect bonding.
- **Element order**:
  - `1` = hex8 linear displacement (shear-locked for slender beams)
  - `2` = Q2 quadratic displacement on hex8 geometry (superparametric) → removes shear locking; recommended
- **Station-based FE points**: Grid corners + mid-side points for Q2; corner points keep grid node IDs.

### Contact Model
- **Rigid cylindrical rollers** (axis along width y). One-way, frictionless.
- **Collocation at FE points** (grid nodes for Q1, + mid-side for Q2).
- **Penalty** with optional **Augmented Lagrangian** outer loop.
- **Auto-penalty**: `penalty < 0` derives nodal `k*A = penalty_scale × mean(diag(K_ff))` — scale is "contact stiffness / bulk stiffness", mesh-independent.

### Boundary Conditions
- **Full model**: `uy = 0` at both end cross-sections only. **No `ux` constraint** — frictionless rollers exert no axial force; constraining both ends would clamp beam slope (`ux ~ -z w'`) and stiffen the span.
- **Half model** (`half_model: true`): `ux = 0` at symmetry plane `x=0`, `uy = 0` elsewhere. Forces doubled in output.
- **Weak ground springs** (`1e-9 × mean(diag(K_ff))`) on `ux`/`uz` DOFs to remove rigid modes — negligible force bleed (<0.01%).

### Solver Optimization (Critical)
- Global stiffness `K` assembled **once**.
- Newton tangent = `K_ff + diag(d)` where `d` is diagonal contact stiffness on contact DOFs `C`.
- **Single factorization** of `A = K_ff + diag(d_all)` (full contact stiffness) — **not** `K_ff` alone (near-singular due to ground springs).
- Active-set correction via **Woodbury** confined to `|C|` DOFs → two triangular solves + small dense solve per Newton iteration.
- Reference case (26.8k DOFs, 10 steps): **22.5 s → 0.4 s** solve time, bit-identical results.

### Stress Recovery & Post-Processing
- **Per-layer nodal averaging** (not global) — nodes shared between layers don't mix materials.
- **Q2 stress recovery**: evaluates Q2 shape-function gradients at physical grid corners using full Q2 displacement field.
- **Through-thickness profile** at mid-span (`x=0`).
- **Span profile**: samples `σxx` and `τxz` along beam axis at layer mid-thickness stations.
- **Failure**: Tsai-Wu (faces), shear/crushing ratio (core).
- **Analytic layup rigidity**: CLT with rotated uniaxial modulus, parallel-axis theorem.

### Units
- Case file **must declare** `units: si` (m/Pa/N) or `units: mm_n_mpa` (mm/MPa/N).
- Conversion happens **once at config boundary**. Solver works internally in mm/N/MPa.
- All outputs written in the **case file's declared units**.

---

## Configuration Structure (Key Fields)

```yaml
units: si                  # REQUIRED
name: case_name
output_dir: results
geometry:
  length: 0.300            # m
  span: 0.200
  width: 0.040
materials:
  mat_name:
    kind: orthotropic      # or isotropic
    E1, E2, E3, nu12...    # orthotropic (MPa in internal)
    E, nu                  # isotropic
    Xt, Xc, Yt, Yc, S      # Tsai-Wu allowables
    shear, compression     # core allowables
stackup:                   # bottom -> top
  - material: mat_name
    thickness: 1.5e-3
    fibre_orientation: 0.0 # deg, rotation about y (width)
    role: face|core        # inferred if omitted
mesh:
  elements_x: 60
  elements_w: 4
  elements_per_layer: {face: 2, core: 10}
  element_order: 1|2       # 2 = Q2 (recommended)
contact:
  penalty: -1.0            # <0 = auto-scale
  penalty_scale: 10.0      # k*A / mean(diag(K))
  roller_radius_load: 10e-3
  roller_radius_support: 5e-3
  augmented_lagrangian:
    enabled: true
    max_outer: 8
    tol: 1e-6
    update_rate: 10.0
loading:
  max_indentation: 2e-3
  n_steps: 10
solver:
  rtol: 1e-8
  max_newton_iterations: 30
```

---

## Output Artifacts (per run)

| File | Content |
|------|---------|
| `report.md/html/pdf/tex` | At a glance, layup, response, failure indices, model, solver checks |
| `load_deflection.csv` | Step history: travel, force, deflection |
| `profile_midspan.csv` | `σxx`, `τxz` through thickness at mid-span |
| `profile_span.csv` | `σxx`, `τxz` along span at layer mid-thickness |
| `summary.json` | All extracted quantities + effective config |
| `plots/*.{svg,pdf,png}` | Stackup, load-deflection, stress profiles |
| `fields.xdmf` | Displacement + stress fields for ParaView |
| `case_used.yaml` | Echo of config in original units (round-trips) |

---

## Testing & Benchmarks

```bash
pytest                          # all tests including analytic benchmarks
pytest -k bench -m "not slow"   # fast subset
```

Benchmarks verified:
- Isotropic beam vs Euler–Bernoulli `δ = PL³/48EI`
- Sandwich vs Allen bending+shear theory
- Force balance < 0.1%, symmetry checks
- Cached-contact tangent vs direct factorization (exact agreement, `test_tangent_solver.py`)

---

## Performance Characteristics

| Aspect | Detail |
|--------|--------|
| Assembly | Dominant cost after tangent optimization |
| Solve (ref case) | ~0.4 s (was 22.5 s before Woodbury) |
| Memory | Sparse `K_ff` + dense `|C|×|C|` Woodbury matrix |
| Scaling | Linear in DOFs for assembly; factorization is `O(n^1.5)` but done once |
| Quadrature | Degree 4 (5×5×5 = 125 pts/cell) — exact for Q2 on trilinear hex |

---

## Entry Points

```bash
# CLI
sandwich3pb run examples/case_glass_pvc.yaml
sandwich3pb sweep examples/sweep_core_thickness.yaml
sandwich3pb report results/glass_pvc_reference/summary.json
sandwich3pb plot results/glass_pvc_reference

# Python API
from sandwich3pb import load_config, run_case
results = run_case("examples/case_glass_pvc.yaml")
print(results.summary["max_force_N"], results.summary["force_gradient_N_per_mm"])
```

---

## Common Extension Points

| Need | Where to Modify |
|------|-----------------|
| New material model | `materials.py` + `config.py` validation |
| Different contact law | `contact.py` (ContactPair) + `solver.py` tangent |
| New failure criterion | `materials.py` + `postprocess.py:failure_indices` |
| Additional output quantity | `results.py:CaseResults.from_solution` + `io.py`/`report.py` |
| Parameter sweep dimension | `cli.py:sweep` (any YAML scalar → list) |
| Half/full model differences | `geometry.py:roller_axes`, `solver.py:_boundary_dofs`, `postprocess.py` |

---

## Dependencies

- **FEniCSx/DOLFINx** (from conda-forge only, no PyPI wheels)
- `numpy`, `scipy`, `matplotlib`, `pyyaml`
- Optional: `pytest`, `pytest-cov` (dev)

---

## Version
0.1.0 (MIT License)