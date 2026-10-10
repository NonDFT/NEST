#!/usr/bin/env python
# Copyright 2026 The NEST Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spin-orbit-coupling operators in the AO basis.

All operators are returned as a complex spherical tensor in component order
``[-1, 0, +1]``.
"""

import copy

import numpy as np
from pyscf.scf.jk import get_jk
from pyscf.data.nist import LIGHT_SPEED

X2CAMF_XRESP = True


def _cartesian_to_spherical(ao_soc):
    """Convert Cartesian SOC components to the ``[-1, 0, +1]`` convention."""
    ao_soc_1 = -0.5 * (ao_soc[0] + 1j * ao_soc[1])
    ao_soc_0 = np.sqrt(0.5) * ao_soc[2]
    ao_soc_m1 = 0.5 * (ao_soc[0] - 1j * ao_soc[1])
    return np.array([ao_soc_m1, ao_soc_0, ao_soc_1])


def _spinor_to_cartesian_soc(mol, h_spinor):
    """Extract the Cartesian SOC operator from a two-component Hamiltonian."""
    spinor_coeff = np.vstack(mol.sph2spinor_coeff())
    h_spin = np.einsum('ip,pq,jq->ij', spinor_coeff, h_spinor, spinor_coeff.conj(), optimize=True)
    nao = mol.nao_nr()
    h_aa = h_spin[:nao, :nao]
    h_bb = h_spin[nao:, nao:]
    h_ab = h_spin[:nao, nao:]
    h_ba = h_spin[nao:, :nao]
    return np.array([h_ab + h_ba, 1j * (h_ab - h_ba), h_aa - h_bb])


def sozeff(atom, zeff_type="one"):
    """Calculate an effective nuclear charge for an atomic number.

    The parametrization is adapted from PyGraSO by Masaya Hagai:
    https://github.com/masaya0222/PyGraSO/blob/main/pygraso/calc_ao_element.py

    Ref: J. Chem. Theory Comput. 2025, 21, 11604.
    """
    assert zeff_type in ["one", "orca", "pysoc"], f"{zeff_type=} is not valid"
    neval = {
        1: 1,
        2: 2,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
        8: 6,
        9: 7,
        10: 8,
        11: 1,
        12: 2,
        13: 3,
        14: 4,
        15: 5,
        16: 6,
        17: 7,
        18: 8,
        19: 1,
        20: 2,
        21: 3,
        22: 4,
        23: 5,
        24: 6,
        25: 7,
        26: 8,
        27: 9,
        28: 10,
        29: 11,
        30: 12,
        31: 3,
        32: 4,
        33: 5,
        34: 6,
        35: 7,
        36: 8,
        37: 1,
        38: 2,
        39: 3,
        40: 4,
        41: 5,
        42: 6,
        43: 7,
        44: 8,
        45: 9,
        46: 10,
        47: 11,
        48: 12,
        49: 3,
        50: 4,
        51: 5,
        52: 6,
        53: 7,
        54: 8,
    }
    if zeff_type == "one":
        return atom

    if zeff_type == "pysoc":
        if atom == 1:
            return 1.0
        elif atom == 2:
            return 2.0
        elif 3 <= atom <= 10:
            return (0.2517 + 0.0626 * neval[atom]) * atom
        elif 11 <= atom <= 18:
            return (0.7213 + 0.0144 * neval[atom]) * atom
        elif (19 <= atom <= 20) or (31 <= atom <= 36):
            return (0.8791 + 0.0039 * neval[atom]) * atom
        elif (37 <= atom <= 38) or (49 <= atom <= 54):
            return (0.9228 + 0.0017 * neval[atom]) * atom
        elif atom == 26:
            return 0.583289 * atom
        elif atom == 30:
            return 330.0
        elif 21 <= atom <= 30:
            return atom * (0.385 + 0.025 * (neval[atom] - 2))
        elif 39 <= atom <= 48:
            return atom * (4.680 + 0.060 * (neval[atom] - 2))
        elif atom == 72:
            return 1025.28
        elif atom == 73:
            return 1049.74
        elif atom == 74:
            return 1074.48
        elif atom == 75:
            return 1099.5
        elif atom == 76:
            return 1124.8
        elif atom == 77:
            return 1150.38
        elif atom == 78:
            return 1176.24
        elif atom == 79:
            return 1202.38
        elif atom == 80:
            return 1228.8
        else:
            raise ValueError(f"SOZEFF is not available for atomic number {atom}")
    if zeff_type == "orca":
        if atom == 1:
            return 1.0
        elif atom == 2:
            return 2.0
        elif 3 <= atom < 10:
            return (0.4 + 0.05 * neval[atom]) * atom
        elif 11 <= atom <= 18:
            return (0.925 - 0.0125 * neval[atom]) * atom
        elif 32 <= atom <= 35:  # Verified from orca output file
            if atom == 32:
                return 32.32
            elif atom == 33:
                return 31.68
            elif atom == 34:
                return 30.94
            elif atom == 35:
                return 30.10
        else:
            raise ValueError(f"SOZEFF is not available for atomic number {atom}")

def get_ao_soc_1e(mol, zeff_type='one'):
    """The one-body part of Hsoc operator with (effective) nuclear charge."""
    zeff_list = [sozeff(mol.atom_charge(i), zeff_type=zeff_type) for i in range(mol.natm)]
    ao_soc = np.zeros((3, mol.nao_nr(), mol.nao_nr()), dtype=np.complex128)
    for k in range(mol.natm):
        mol.set_rinv_orig(mol.atom_coord(k))
        ao_soc += (-1.0j) * zeff_list[k] * mol.intor('int1e_prinvxp')
    ao_soc /= (2.0 * LIGHT_SPEED**2)
    return _cartesian_to_spherical(ao_soc)


def get_ao_soc_x2camf(mol):
    """Return the X2CAMF SOC operator.
    Ref: Chem. Phys. Rev. 2025, 6, 031404.
    """
    try:
        from socutils.somf import somf_pt
    except ImportError as err:
        raise ImportError(
            'Please install socutils package to use X2CAMF SOC integrals. '
            'https://github.com/wtpeter/socutils'
        ) from err
    ao_soc = 2j * somf_pt.get_psoc_x2camf(mol, xresp=X2CAMF_XRESP)
    return _cartesian_to_spherical(ao_soc)


def get_ao_soc_x2c1e(mol):
    """Return the spin-dependent part of PySCF's spinor X2C1e core."""
    from pyscf.x2c.x2c import SpinorX2CHelper

    h_x2c1e = SpinorX2CHelper(mol).get_hcore(mol)
    return _cartesian_to_spherical(_spinor_to_cartesian_soc(mol, h_x2c1e))


def get_ao_soc_x2cmp(mol):
    """Build the X2CMP core and return its Pauli spin-dependent part.
    Ref: Chem. Phys. Rev. 2025, 6, 031404.
    """
    try:
        from socutils.somf.x2cmp import SpinorX2CMPHelper
    except ImportError as err:
        raise ImportError(
            'Please install socutils package to use X2CMP SOC integrals. '
            'https://github.com/wtpeter/socutils'
        ) from err

    x2cobj = SpinorX2CMPHelper(
        mol,
        x2cmp='x2cmp',
        with_gaunt=True,
        with_breit=True,
        with_pcc=False,
    )
    h_x2cmp = x2cobj.get_hcore(mol)
    return _cartesian_to_spherical(_spinor_to_cartesian_soc(mol, h_x2cmp))


def get_ao_soc_2e_somf(mf, amfi=False):
    """Return the two-electron SOC in the SOMF approximation.

    Direct SCF is used.  If ``amfi`` is true, only one-center AO blocks are
    retained.

    Ref: J. Chem. Phys. 2005, 122, 034107.
         J. Chem. Phys. 2022, 157, 224110.
    """
    mol = mf.mol
    dm = mf.make_rdm1()

    if dm.ndim == 3:
        dm = dm[0] + dm[1]

    dm_list = [dm, dm, dm]
    scripts = ['ijkl,lk->ij', 'ijkl,jk->il', 'ijkl,li->kj']
    if amfi:
        nao = mol.nao_nr()
        vj = np.zeros((3, nao, nao))
        vk1 = np.zeros_like(vj)
        vk2 = np.zeros_like(vj)
        aoslices = mol.aoslice_by_atom(mol.ao_loc_nr())
        atom = copy.copy(mol)
        for b0, b1, p0, p1 in aoslices:
            atom._bas = mol._bas[b0:b1]
            dm_atom = dm[p0:p1, p0:p1]
            vj_atom, vk1_atom, vk2_atom = get_jk(
                atom,
                [dm_atom, dm_atom, dm_atom],
                scripts=scripts,
                intor='int2e_p1vxp1',
                comp=3,
                aosym='a4ij',
            )
            vj[:, p0:p1, p0:p1] = vj_atom
            vk1[:, p0:p1, p0:p1] = vk1_atom
            vk2[:, p0:p1, p0:p1] = vk2_atom
    else:
        vj, vk1, vk2 = get_jk(
            mol,
            dm_list,
            scripts=scripts,
            intor='int2e_p1vxp1',
            comp=3,
            aosym='a4ij',
        )

    ao_soc = 1j / (2.0 * LIGHT_SPEED**2) * (vj - 1.5 * (vk1 + vk2))
    return _cartesian_to_spherical(ao_soc)


def _symmetrize_ao_soc(soc_ao):
    """Enforce the Hermiticity relations of a rank-one spherical tensor."""
    soc_m1, soc_0, soc_1 = soc_ao
    soc_1, soc_m1 = (
        0.5 * (soc_1 + soc_m1.conj()),
        0.5 * (soc_m1 + soc_1.conj()),
    )
    soc_0 = 0.5 * (soc_0 + soc_0.conj().T)
    soc_m1, soc_1 = (
        0.5 * (soc_m1 - soc_1.conj().T),
        0.5 * (soc_1 - soc_m1.conj().T),
    )
    return np.array([soc_m1, soc_0, soc_1])

def get_ao_soc(mf, soctype):
    """Dispatch a supported ``soctype`` and return its AO SOC tensor."""
    mol = mf.mol
    if soctype == 'SOMF':
        soc_ao = get_ao_soc_1e(mol, zeff_type='one')
        soc_ao += get_ao_soc_2e_somf(mf)
    elif soctype == 'SOMF_AMFI':
        soc_ao = get_ao_soc_1e(mol, zeff_type='one')
        soc_ao += get_ao_soc_2e_somf(mf, amfi=True)
    elif soctype == 'Zeff':
        soc_ao = get_ao_soc_1e(mol, zeff_type='orca')
    elif soctype == '1e':
        soc_ao = get_ao_soc_1e(mol, zeff_type='one')
    elif soctype == 'X2C1E':
        soc_ao = get_ao_soc_x2c1e(mol)
    elif soctype == 'X2CAMF':
        soc_ao = get_ao_soc_x2camf(mol)
    elif soctype == 'X2CMP':
        soc_ao = get_ao_soc_x2cmp(mol)
    else:
        raise ValueError(f'soctype={soctype} is not supported.')
    return _symmetrize_ao_soc(soc_ao)
