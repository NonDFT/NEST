"""Check the deltaS=+1 derivation with small, independent random tensors.

No SCF or production gradient helpers are used here. The model Fock and
response kernel depend on the reference density to exercise their derivatives.
"""

import numpy as np


def kernel_tensor(rng, size):
    tensor = rng.normal(size=(size,) * 4) / size
    # Self-adjoint and compatible with matrix transposition, like J/K/XC.
    tensor = 0.5 * (tensor + tensor.transpose(2, 3, 0, 1))
    return 0.5 * (tensor + tensor.transpose(1, 0, 3, 2))


def apply(kernel, density):
    return np.einsum('pqrs,rs->pq', kernel, density)


def inner(left, right):
    return np.sum(left * right)


def check():
    rng = np.random.default_rng(20261010)
    nao, nc, no, nv = 7, 2, 1, 2
    nmo = nc + no + nv
    coeff = np.linalg.qr(rng.normal(size=(nao, nmo)))[0]
    amplitude = rng.normal(size=(nc, nv))
    amplitude /= np.linalg.norm(amplitude)
    hole = amplitude @ amplitude.T
    particle = amplitude.T @ amplitude
    occupation = np.array([2] * nc + [1] * no + [0] * nv)
    c, o, v = slice(0, nc), slice(nc, nc + no), slice(nc + no, nmo)
    core, core_derivative, density_weight = (rng.normal(size=(nao, nao)) for _ in range(3))
    core = 0.5 * (core + core.T)
    core_derivative = 0.5 * (core_derivative + core_derivative.T)
    density_weight = 0.5 * (density_weight + density_weight.T)
    fock_kernel, response_kernel, kernel_density_derivative, kernel_derivative = (
        kernel_tensor(rng, nao) for _ in range(4)
    )

    def evaluate(mo, coordinate=0.0):
        closed, opened, virtual = mo[:, c], mo[:, o], mo[:, v]
        density_open = opened @ opened.T
        density = 2 * closed @ closed.T + density_open
        transition = virtual @ amplitude.T @ closed.T
        dp = virtual @ particle @ virtual.T
        dh = closed @ hole @ closed.T
        p0, pz = dp - dh, dp + dh
        kernel = (
            response_kernel
            + inner(density_weight, 0.5 * density) * kernel_density_derivative
            + coordinate * kernel_derivative
        )
        f0 = core + coordinate * core_derivative + apply(fock_kernel, density)
        fz = 0.5 * apply(kernel, density_open)
        potential = apply(kernel, transition)
        scalar = inner(p0, f0) + inner(pz, fz) + inner(transition, potential)

        action = amplitude @ (virtual.T @ (f0 + fz) @ virtual).T
        action -= (closed.T @ (f0 - fz) @ closed).T @ amplitude
        action += (virtual.T @ potential @ closed).T
        closure = abs(scalar - inner(amplitude, action))

        m_matrix = np.zeros((nmo, nmo))
        m_matrix[:, v] += 2 * mo.T @ (f0 + fz) @ virtual @ particle
        m_matrix[:, c] -= 2 * mo.T @ (f0 - fz) @ closed @ hole
        m_matrix[:, v] += 2 * mo.T @ potential @ closed @ amplitude
        m_matrix[:, c] += 2 * mo.T @ potential.T @ virtual @ amplitude.T
        open_potential = 0.5 * apply(kernel, pz)
        m_matrix[:, o] += mo.T @ (open_potential + open_potential.T) @ opened

        reference_potential = apply(fock_kernel, p0)
        kernel_weight = 0.5 * inner(pz, apply(kernel_density_derivative, density_open)) + inner(
            transition, apply(kernel_density_derivative, transition)
        )
        reference_potential += 0.5 * kernel_weight * density_weight
        m_matrix += (mo.T @ (reference_potential + reference_potential.T) @ mo) * occupation
        direct = (
            inner(p0, core_derivative)
            + 0.5 * inner(pz, apply(kernel_derivative, density_open))
            + inner(transition, apply(kernel_derivative, transition))
        )
        return scalar, closure, m_matrix, direct

    _, closure_error, m_matrix, direct = evaluate(coeff)
    step = 1e-6
    errors = []
    # General (not just antisymmetric) changes check the overlap-relevant part.
    for _ in range(12):
        perturbation = rng.normal(size=(nmo, nmo))
        plus = evaluate(coeff @ (np.eye(nmo) + step * perturbation))[0]
        minus = evaluate(coeff @ (np.eye(nmo) - step * perturbation))[0]
        errors.append(abs((plus - minus) / (2 * step) - inner(m_matrix, perturbation)))
    plus = evaluate(coeff, step)[0]
    minus = evaluate(coeff, -step)[0]
    direct_error = abs((plus - minus) / (2 * step) - direct)
    for label, error in (
        ('scalar closure', closure_error),
        ('MO derivative', max(errors)),
        ('explicit derivative', direct_error),
    ):
        print(f'{label}: max absolute error = {error:.3e}')
        if error >= 2e-8:
            raise AssertionError(f'{label} failed: {error}')


if __name__ == '__main__':
    check()
