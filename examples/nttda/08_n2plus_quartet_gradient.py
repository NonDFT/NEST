"""Lowest quartet of N2+ at the initial geometry of the bond relaxation.

B3LYP/6-31G*, doublet AOCSCF orbitals, N--N distance 1.105 Angstrom.
The bond derivative is d(E_ref + omega)/dR, in hartree/bohr.
"""
from pyscf import gto
from nest import aocscf, nttda  # register average_occ and NTTDA

mol = gto.M(atom='N 0 0 -0.5525; N 0 0 0.5525', unit='Angstrom',
            basis='6-31g*', charge=1, spin=1, verbose=0)
mf = mol.ROKS(xc='B3LYP').average_occ()
mf.grids.level = 5
mf.set(conv_tol=1e-12, conv_tol_grad=1e-9).run()
td = mf.NTTDA().set(deltaS=1, nobeta=False, nstates=1,
                    conv_tol=1e-9, lindep=1e-20, max_cycle=200).run()
gradient = td.Gradients().kernel(state=1)
print('Quartet total energy (hartree):', td.e_tot[0])
print('Bond derivative (hartree/bohr):', (gradient[1, 2] - gradient[0, 2]) / 2)
