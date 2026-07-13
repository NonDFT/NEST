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

"""SOC Hamiltonian in AO basis."""

import numpy as np
from pyscf.scf.jk import get_jk
from pyscf.data.nist import LIGHT_SPEED

X2CAMF_XRESP = True

def sozeff(atom, zeff_type="one"):
    """
    Calculate effective nuclear charge for given atomic number
    copied from: https://github.com/masaya0222/PyGraSO/blob/main/pygraso/calc_ao_element.py
    Author: Masaya Hagai <hagai.masaya.v9@s.mail.nagoya-u.ac.jp>
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
    '''
    The one-body part of Hsoc operator with (effective) nuclear charge.
    '''
    zeff_list = [sozeff(mol.atom_charge(i), zeff_type=zeff_type) for i in range(mol.natm)]
    ao_soc = np.zeros((3, mol.nao_nr(), mol.nao_nr()), dtype=np.complex128)
    for k in range(mol.natm):
        mol.set_rinv_orig(mol.atom_coord(k))
        ao_soc += (-1.0j) * zeff_list[k] * mol.intor('int1e_prinvxp')
    ao_soc /= (2.0 * LIGHT_SPEED**2)
    ao_soc_1 = -0.5 * (ao_soc[0] + 1j * ao_soc[1])
    ao_soc_0 = np.sqrt(0.5) * ao_soc[2]
    ao_soc_m1 = 0.5 * (ao_soc[0] - 1j * ao_soc[1])
    return np.array([ao_soc_m1, ao_soc_0, ao_soc_1])

def get_ao_soc_x2camf(mol):
    import resource
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        target = resource.RLIM_INFINITY if hard == resource.RLIM_INFINITY else hard
        resource.setrlimit(resource.RLIMIT_STACK, (target, hard))
    except Exception as e:
        print(f"Warning: failed to increase stack size: {e}")
    try:
        from socutils.somf import somf_pt
    except ImportError:
        raise ImportError("Please install socutils package to use X2CAMF SOC integrals." \
        "https://github.com/wtpeter/socutils")
    ao_soc = 2j * somf_pt.get_psoc_x2camf(mol, xresp=X2CAMF_XRESP)
    ao_soc_1 = -0.5 * (ao_soc[0] + 1j * ao_soc[1])
    ao_soc_0 = np.sqrt(0.5) * ao_soc[2]
    ao_soc_m1 = 0.5 * (ao_soc[0] - 1j * ao_soc[1])
    return np.array([ao_soc_m1, ao_soc_0, ao_soc_1])

def get_ao_soc_2e_somf(mf):
    '''
    The two-electron part of the Hsoc under the SOMF approximation
    using direct SCF for UKS density matrix.
    '''
    mol = mf.mol
    dm = mf.make_rdm1()

    if dm.ndim == 2:
        dmaa = dmbb = 0.5 * dm
    else:
        dmaa, dmbb = dm

    dm_list = [dmaa, dmaa, dmaa, dmbb, dmbb, dmbb]
    scripts = [
        'ijkl,lk->ij', # J for dmaa
        'ijkl,jk->il', # K1 for dmaa
        'ijkl,li->kj', # K2 for dmaa
        'ijkl,lk->ij', # J for dmbb
        'ijkl,jk->il', # K1 for dmbb
        'ijkl,li->kj'  # K2 for dmbb
    ]
    v_matrices = get_jk(mol, dm_list, scripts=scripts, intor='int2e_p1vxp1', comp=3, aosym='a4ij')
    vj_aa, vk1_aa, vk2_aa = v_matrices[0:3]
    vj_bb, vk1_bb, vk2_bb = v_matrices[3:6]

    v_cart_1 = (vj_aa - vk1_aa - 2 * vk2_aa) + (vj_bb - 2 * vk1_bb - vk2_bb)
    v_cart_0 = (vj_aa + vj_bb) - 1.5 * (vk1_aa + vk1_bb) - 1.5 * (vk2_aa + vk2_bb)
    v_cart_m1 = (vj_aa - 2 * vk1_aa - vk2_aa) + (vj_bb - vk1_bb - 2 * vk2_bb)

    def to_spherical(v_cart_xyz, component):
        # v_cart_xyz shape is (3, nao, nao)
        vx = v_cart_xyz[0]
        vy = v_cart_xyz[1]
        vz = v_cart_xyz[2]
        if component == 1:
            return -0.5 * (vx + 1j * vy)
        elif component == 0:
            return np.sqrt(0.5) * vz
        elif component == -1:
            return 0.5 * (vx - 1j * vy)

    prefactor = 1j / (2.0 * LIGHT_SPEED**2)
    soc_somf_1 = to_spherical(v_cart_1, 1) * prefactor
    soc_somf_0 = to_spherical(v_cart_0, 0) * prefactor
    soc_somf_m1 = to_spherical(v_cart_m1, -1) * prefactor
    soc_somf_1, soc_somf_m1 = (
        0.5 * (soc_somf_1 + soc_somf_m1.conj()),
        0.5 * (soc_somf_m1 + soc_somf_1.conj()),
    )
    return np.array([soc_somf_m1, soc_somf_0, soc_somf_1])


def _symmetrize_ao_soc(soc_ao):
    """Enforce the Hermiticity relations of a rank-one spherical tensor."""
    soc_m1, soc_0, soc_1 = soc_ao
    soc_0 = 0.5 * (soc_0 + soc_0.conj().T)
    soc_m1, soc_1 = (
        0.5 * (soc_m1 - soc_1.conj().T),
        0.5 * (soc_1 - soc_m1.conj().T),
    )
    return np.array([soc_m1, soc_0, soc_1])

def get_ao_soc(mf, soctype):
    mol = mf.mol
    if soctype == 'SOMF':
        soc_ao = get_ao_soc_1e(mol, zeff_type='one')
        soc_ao += get_ao_soc_2e_somf(mf)
    elif soctype == 'Zeff':
        soc_ao = get_ao_soc_1e(mol, zeff_type='orca')
    elif soctype == '1e':
        soc_ao = get_ao_soc_1e(mol, zeff_type='one')
    elif soctype == 'X2CAMF':
        soc_ao = get_ao_soc_x2camf(mol)
    else:
        raise ValueError(f'soctype={soctype} is not supported.')
    return _symmetrize_ao_soc(soc_ao)
