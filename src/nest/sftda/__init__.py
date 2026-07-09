#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

from pyscf import scf, dft
from nest.sftda import uhf_sf


def TDA_SF(mf):
    mf = mf.remove_soscf()
    if isinstance(mf, scf.rohf.ROHF) or isinstance(mf, scf.hf_symm.SymAdaptedROHF):
        if isinstance(mf, dft.roks.ROKS) or isinstance(mf, dft.rks_symm.SymAdaptedROKS):
            mf = mf.to_uks()
        else:
            mf = mf.to_uhf()
    return mf.TDA_SF()


def TDDFT_SF(mf):
    mf = mf.remove_soscf()
    if isinstance(mf, scf.rohf.ROHF) or isinstance(mf, scf.hf_symm.SymAdaptedROHF):
        if isinstance(mf, dft.roks.ROKS) or isinstance(mf, dft.rks_symm.SymAdaptedROKS):
            mf = mf.to_uks()
        else:
            mf = mf.to_uhf()
    return mf.TDDFT_SF()


SFTDA = TDA_SF
SFTDDFT = TDDFT_SF

__all__ = [
    "SFTDA",
    "SFTDDFT",
    "TDA_SF",
    "TDDFT_SF",
    "uhf_sf",
]
