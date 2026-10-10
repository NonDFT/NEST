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

from pyscf import scf
from nest.sftda import uhf_sf


def TDA_SF(mf, extype=1, collinear="mcol", collinear_samples=20):
    """Construct SF-TDA from a UHF/UKS or ROHF/ROKS reference."""
    mf = mf.remove_soscf()
    restricted = isinstance(mf, scf.rohf.ROHF)
    if restricted:
        if mf.mo_coeff is None:
            mf.run()
        if isinstance(mf, scf.hf.KohnShamDFT):
            mf = mf.to_uks()
        else:
            mf = mf.to_uhf()
    td = uhf_sf.TDA_SF(mf, extype, collinear, collinear_samples)
    td._ro_reference = restricted
    return td


def TDDFT_SF(mf, extype=1, collinear="mcol", collinear_samples=20):
    """Construct SF-TDDFT from a UHF/UKS or ROHF/ROKS reference."""
    mf = mf.remove_soscf()
    restricted = isinstance(mf, scf.rohf.ROHF)
    if restricted:
        if mf.mo_coeff is None:
            mf.run()
        if isinstance(mf, scf.hf.KohnShamDFT):
            mf = mf.to_uks()
        else:
            mf = mf.to_uhf()
    td = uhf_sf.TDDFT_SF(mf, extype, collinear, collinear_samples)
    td._ro_reference = restricted
    return td


SFTDA = TDA_SF
SFTDDFT = TDDFT_SF

scf.uhf.UHF.TDA_SF = scf.uhf.UHF.SFTDA = TDA_SF
scf.uhf.UHF.TDDFT_SF = scf.uhf.UHF.SFTDDFT = TDDFT_SF
scf.rohf.ROHF.TDA_SF = scf.rohf.ROHF.SFTDA = TDA_SF
scf.rohf.ROHF.TDDFT_SF = scf.rohf.ROHF.SFTDDFT = TDDFT_SF

__all__ = [
    "SFTDA",
    "SFTDDFT",
    "TDA_SF",
    "TDDFT_SF",
    "uhf_sf",
]
