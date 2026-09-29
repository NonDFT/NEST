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

"""Run closed-shell TDA/TDDFT SOC calculation."""

import numpy as np
from pyscf import gto
from pyscf.data.nist import HARTREE2WAVENUMBER

from nest.soc import TDRHFSOC


mol = gto.M(
    atom="""
    O   0.64372820   0.14077399  -0.04477253
    O  -0.64862595  -0.12779073  -0.05445498
    H   1.16027512  -0.65947800   0.36730132
    H  -1.12109306   0.55561188   0.42651873
    """,
    basis="631g",
    charge=0,
    spin=0,
    symmetry=False,
)

# 1. Closed-shell scalar reference.
mf = mol.RKS(xc="PBE").set(conv_tol=1e-12).run()

# 2. Calculate singlets and triplets from the same reference.  Use TDA for the
# ORCA reference below.  Full TDDFT requires only replacing both TDA() calls by
# TDDFT(); the SOC driver then uses normalized (X+Y) Casida amplitudes.
singlet = mf.TDA().set(
    singlet=True, nstates=3, conv_tol=1e-6, max_cycle=300,
).run()
triplet = mf.TDA().set(
    singlet=False, nstates=3, conv_tol=1e-6, max_cycle=300,
).run()

# 3. The closed-shell reference is included by default.  It can be omitted with
# include_reference=False.  One or more RHF/RKS TD objects may be supplied; a
# single object is valid but produces a warning because only one spin sector is
# then represented.
soc_driver = TDRHFSOC(
    singlet,
    triplet,
    soctype="SOMF",
    include_reference=True,
).run()

# 4. Print scalar states, every SOC block, coupled eigenvalues, and eigenvectors.
soc_driver.analyze(verbose=6)

# Raw numerical data are available without parsing the analysis output.
np.set_printoptions(precision=4, linewidth=180)
print("\nNEST h_soc / Hartree")
print(soc_driver.h_soc)
print("\nNEST spin-orbit-coupled energies from the lowest state / cm^-1")
print((soc_driver.e - soc_driver.e.min()).real * HARTREE2WAVENUMBER)


r"""
ORCA input
==========

! PBE 6-31G NoRI NoAutoStart

%pal
  nprocs 1
end

%maxcore 2000

%scf
  Convergence Extreme
end

%tddft
  NRoots 3
  Triplets true
  TDA true
  DoSOC true
  PrintLevel 4
  ETol 1e-10
  RTol 1e-10
end

%rel
  SOCType 3
  SOCFlags 1,4,4,0
end

* xyz 0 1
 O   0.64372820    0.14077399   -0.04477253
 O  -0.64862595   -0.12779073   -0.05445498
 H   1.16027512   -0.65947800    0.36730132
 H  -1.12109306    0.55561188    0.42651873
*


ORCA 6.1 output
===============

      --------------------------------------------------------------------------------
                      CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS
      --------------------------------------------------------------------------------
           Root                          <T|HSO|S>  (Re, Im) cm-1
         T      S           MS= 0                  -1                    +1
      --------------------------------------------------------------------------------
         1      0    (0.00e+00 , 1.36e+01)    (-6.52e+00 , 1.27e+01)    (-6.52e+00 , -1.27e+01)
         1      1    (0.00e+00 , 4.21e+00)    (-1.01e+00 , -2.92e+00)    (-1.01e+00 , 2.92e+00)
         1      2    (0.00e+00 , 1.16e+00)    (-6.77e+00 , 1.58e+00)    (-6.77e+00 , -1.58e+00)
         1      3    (0.00e+00 , -3.15e+00)    (1.08e+01 , 3.78e+01)    (1.08e+01 , -3.78e+01)
         2      0    (0.00e+00 , -5.40e+00)    (-2.03e+01 , 2.61e+01)    (-2.03e+01 , -2.61e+01)
         2      1    (0.00e+00 , -6.52e+00)    (6.35e+00 , -7.29e+00)    (6.35e+00 , 7.29e+00)
         2      2    (0.00e+00 , 2.07e+00)    (-9.58e-01 , 1.27e+00)    (-9.58e-01 , -1.27e+00)
         2      3    (0.00e+00 , -2.35e-01)    (2.85e+00 , 6.15e+00)    (2.85e+00 , -6.15e+00)
         3      0    (0.00e+00 , 1.32e+01)    (-7.82e+00 , -2.58e+01)    (-7.82e+00 , 2.58e+01)
         3      1    (0.00e+00 , -7.95e-01)    (-1.03e+01 , -3.79e+01)    (-1.03e+01 , 3.79e+01)
         3      2    (0.00e+00 , -2.04e+00)    (2.66e+00 , 6.44e+00)    (2.66e+00 , -6.44e+00)
         3      3    (0.00e+00 , 3.89e+00)    (-2.43e+00 , -1.87e+00)    (-2.43e+00 , 1.87e+00)

SOC stabilization of the ground state:  -0.0948 cm-1
Eigenvalues of the SOC matrix:

   State:        cm-1         eV
     0:          0.00         0.0000
     1:      39053.72         4.8420
     2:      39053.75         4.8420
     3:      39053.87         4.8421
     4:      46821.26         5.8051
     5:      46828.18         5.8060
     6:      46828.22         5.8060
     7:      46861.13         5.8100
     8:      50359.20         6.2437
     9:      50359.20         6.2437
    10:      50359.76         6.2438
    11:      52098.34         6.4594
    12:      59318.26         7.3545
"""
