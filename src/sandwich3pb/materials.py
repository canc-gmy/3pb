"""Linear-elastic material tensors, fibre rotation and failure criteria.

All quantities in MPa / mm / N consistent with the package convention.

Voigt conventions used throughout:
  * strain vector: [e11, e22, e33, g23, g13, g12]  (engineering shear)
  * stress vector: [s11, s22, s33, s23, s13, s12]  (tensor shear)
With these conventions the plain quadratic form 0.5 e^T C e equals the strain
energy density, and s = C e holds with G on the shear diagonal of C.

Rotations are performed on the full fourth-order stiffness tensor
(C_ijkl = R_i'a R_j'b R_k'c R_l'd C_abcd) which avoids the classic factor-2
pitfalls of Voigt-space rotation matrices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import MaterialSpec

# --------------------------------------------------------------------------
# Voigt index helpers
# --------------------------------------------------------------------------

# (i, j) index pairs in Voigt order (tensor shear for stress)
_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _pair_index(i: int, j: int) -> int:
    """Voigt index of the (i, j) component pair (order-independent)."""
    i, j = (i, j) if i <= j else (j, i)
    for k, pair in enumerate(_PAIRS):
        if pair == (i, j):
            return k
    raise ValueError(f"invalid index pair ({i}, {j})")


def voigt_to_tensor_stress(v: np.ndarray) -> np.ndarray:
    """Stress Voigt vector (tensor shear) -> symmetric 3x3 tensor."""
    s = np.zeros((3, 3))
    for k, (i, j) in enumerate(_PAIRS):
        s[i, j] = s[j, i] = v[k]
    return s


def tensor_to_voigt_stress(s: np.ndarray) -> np.ndarray:
    """Symmetric 3x3 tensor -> stress Voigt vector (tensor shear)."""
    return np.array([s[i, j] for i, j in _PAIRS])


def voigt_to_tensor_strain(v: np.ndarray) -> np.ndarray:
    """Strain Voigt vector (engineering shear) -> symmetric 3x3 tensor."""
    e = np.zeros((3, 3))
    for k, (i, j) in enumerate(_PAIRS):
        e[i, j] = e[j, i] = 0.5 * v[k] if i != j else v[k]
    return e


def tensor_to_voigt_strain(e: np.ndarray) -> np.ndarray:
    """Symmetric 3x3 tensor -> strain Voigt vector (engineering shear)."""
    return np.array(
        [e[i, j] if i == j else 2.0 * e[i, j] for i, j in _PAIRS]
    )


# --------------------------------------------------------------------------
# Rotation about the width (global y) axis
# --------------------------------------------------------------------------


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    """3x3 rotation R mapping material-axis components to global axes:
    a_global = R @ a_material. A positive angle rotates the material 1-axis
    (fibre) from +x towards +z."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def rotate_tensor4(C4: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate a 4th-order tensor: C'_ijkl = R_im R_jn R_ko R_lp C_mnop."""
    return np.einsum("im,jn,ko,lp,mnop->ijkl", R, R, R, R, C4, optimize=True)


def voigt_to_tensor4(C_v: np.ndarray) -> np.ndarray:
    """Eng-Voigt 6x6 stiffness -> 4th-order tensor C_ijkl (s_ij = C_ijkl e_kl).

    Fully minor-symmetrized: C_ijkl = C_jikl = C_ijlk = C_jilk.
    """
    C4 = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(_PAIRS):
        for b, (k, l) in enumerate(_PAIRS):
            val = C_v[a, b]
            C4[i, j, k, l] = val
            C4[j, i, k, l] = val
            C4[i, j, l, k] = val
            C4[j, i, l, k] = val
    return C4


def tensor4_to_voigt(C4: np.ndarray) -> np.ndarray:
    """4th-order tensor -> Eng-Voigt 6x6 stiffness matrix."""
    C_v = np.zeros((6, 6))
    for a, (i, j) in enumerate(_PAIRS):
        for b, (k, l) in enumerate(_PAIRS):
            C_v[a, b] = C4[i, j, k, l]
    return C_v


# --------------------------------------------------------------------------
# Stiffness matrices
# --------------------------------------------------------------------------


def isotropic_stiffness(E: float, nu: float) -> np.ndarray:
    """6x6 isotropic stiffness matrix (Eng-Voigt)."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    for i in range(3):
        C[i, i] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def orthotropic_stiffness(mat: MaterialSpec) -> np.ndarray:
    """Compliance-based 6x6 orthotropic stiffness in material axes.

    Requires a positive-definite compliance (thermodynamic admissibility).
    """
    S = np.zeros((6, 6))
    S[0, 0] = 1.0 / mat.E1
    S[1, 1] = 1.0 / mat.E2
    S[2, 2] = 1.0 / mat.E3
    S[0, 1] = S[1, 0] = -mat.nu12 / mat.E1
    S[0, 2] = S[2, 0] = -mat.nu13 / mat.E1
    S[1, 2] = S[2, 1] = -mat.nu23 / mat.E2
    S[3, 3] = 1.0 / mat.G23
    S[4, 4] = 1.0 / mat.G13
    S[5, 5] = 1.0 / mat.G12

    if np.any(np.linalg.eigvalsh(S) <= 0):
        raise ValueError(
            f"material '{mat.name}': engineering constants are not "
            "positive-definite (check nu reciprocity and G values)"
        )
    return np.linalg.inv(S)


def stiffness_matrix(mat: MaterialSpec) -> np.ndarray:
    """Material-axis 6x6 stiffness matrix (Eng-Voigt) of a MaterialSpec."""
    if mat.is_isotropic:
        return isotropic_stiffness(mat.E, mat.nu)
    return orthotropic_stiffness(mat)


@dataclass
class LocalizedQuad:
    """Global-axis stiffness data of one (material, orientation) pair."""

    C: np.ndarray            # global-axis 6x6 stiffness (Eng-Voigt)
    C_material: np.ndarray   # material-axis 6x6 stiffness (Eng-Voigt)
    R: np.ndarray            # material -> global rotation (3x3)

    def quad_form(self, eps_g: np.ndarray) -> float:
        """Strain energy density 0.5 * e^T C e for a global Eng-Voigt strain."""
        return 0.5 * float(eps_g @ self.C @ eps_g)

    def stress_material_from_global(self, sig_g: np.ndarray) -> np.ndarray:
        """Stress Voigt vector(s) global -> material axes.

        Accepts (6,) or (n, 6) arrays; uses sigma_m = R^T sigma_g R on the
        underlying tensors.
        """
        one = sig_g.ndim == 1
        sig = np.atleast_2d(sig_g)
        tensors = np.zeros((sig.shape[0], 3, 3), dtype=sig.dtype)
        tensors[:, 0, 0] = sig[:, 0]
        tensors[:, 1, 1] = sig[:, 1]
        tensors[:, 2, 2] = sig[:, 2]
        tensors[:, 1, 2] = tensors[:, 2, 1] = sig[:, 3]
        tensors[:, 0, 2] = tensors[:, 2, 0] = sig[:, 4]
        tensors[:, 0, 1] = tensors[:, 1, 0] = sig[:, 5]
        rotated = np.einsum("ia,nij,jb->nab", self.R, tensors, self.R,
                            optimize=True)
        out = np.column_stack((
            rotated[:, 0, 0], rotated[:, 1, 1], rotated[:, 2, 2],
            rotated[:, 1, 2], rotated[:, 0, 2], rotated[:, 0, 1],
        ))
        return out[0] if one else out

    def strain_material_from_global(self, eps_g: np.ndarray) -> np.ndarray:
        """Strain Voigt vector(s) global -> material axes (e_m = R^T e_g R)."""
        one = eps_g.ndim == 1
        eps = np.atleast_2d(eps_g)
        tensors = np.zeros((eps.shape[0], 3, 3), dtype=eps.dtype)
        tensors[:, 0, 0] = eps[:, 0]
        tensors[:, 1, 1] = eps[:, 1]
        tensors[:, 2, 2] = eps[:, 2]
        tensors[:, 1, 2] = tensors[:, 2, 1] = 0.5 * eps[:, 3]
        tensors[:, 0, 2] = tensors[:, 2, 0] = 0.5 * eps[:, 4]
        tensors[:, 0, 1] = tensors[:, 1, 0] = 0.5 * eps[:, 5]
        rotated = np.einsum("ia,nij,jb->nab", self.R, tensors, self.R,
                            optimize=True)
        out = np.column_stack((
            rotated[:, 0, 0], rotated[:, 1, 1], rotated[:, 2, 2],
            2.0 * rotated[:, 1, 2], 2.0 * rotated[:, 0, 2],
            2.0 * rotated[:, 0, 1],
        ))
        return out[0] if one else out


def localized_quad_form(mat: MaterialSpec, angle_deg: float) -> LocalizedQuad:
    """Global-axis stiffness of a (possibly rotated) material."""
    C_m = stiffness_matrix(mat)
    R = rotation_matrix_y(angle_deg)
    C4_g = rotate_tensor4(voigt_to_tensor4(C_m), R)
    C_g = tensor4_to_voigt(C4_g)
    return LocalizedQuad(C=C_g, C_material=C_m, R=R)


def material_stress(eps_g: np.ndarray, quad: LocalizedQuad) -> np.ndarray:
    """Global-axis stress Voigt vector from a global-axis strain vector."""
    return quad.C @ eps_g


# --------------------------------------------------------------------------
# Failure criteria
# --------------------------------------------------------------------------


def tsai_wu_3d(sig: np.ndarray, mat: MaterialSpec) -> float:
    """Tsai-Wu failure index for a stress state in *material* axes.

    ``sig``: Voigt vector [s11, s22, s33, s23, s13, s12] in MPa (tensor
    shear). Index > 1 predicts failure; the classic in-plane 2D Tsai-Wu is
    recovered when s33 = s23 = s13 = 0.
    """
    F1 = 1.0 / mat.Xt - 1.0 / mat.Xc
    F2 = 1.0 / mat.Yt - 1.0 / mat.Yc
    F11 = 1.0 / (mat.Xt * mat.Xc)
    F22 = 1.0 / (mat.Yt * mat.Yc)
    F66 = 1.0 / mat.S**2

    s1, s2 = sig[0], sig[1]
    s6 = sig[5]
    return float(
        F1 * s1 + F2 * s2 + F11 * s1 * s1 + F22 * s2 * s2 + F66 * s6 * s6
    )


def core_shear_ratio(tau_zx: float, mat: MaterialSpec) -> float:
    """|tau_zx| / allowable for a (foam) core material."""
    return abs(tau_zx) / mat.shear


def core_crushing_ratio(sigma_zz: float, mat: MaterialSpec) -> float:
    """-sigma_zz / crushing allowable (compressive through-thickness stress)."""
    return max(0.0, -sigma_zz) / mat.compression


def von_mises(voigt6: np.ndarray) -> float:
    """von Mises equivalent stress from a stress Voigt vector."""
    s = voigt_to_tensor_stress(voigt6)
    dev = s - np.trace(s) / 3.0 * np.eye(3)
    return float(np.sqrt(1.5 * np.tensordot(dev, dev)))
