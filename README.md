# NEST

### Noncollinear Electronic Structure Theory

NEST is an open-source framework for developing methods based on noncollinear electronic structure theory.
It is built on top of the PySCF electronic structure package.

## Framework

```text
                                 PySCF
                                   │
                                   ▼
                                 NEST
                    (Multicollinear Framework)
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
       GKS                        UKS                        ROKS
        │                          │                          │
        ▼                          ▼                          ▼
Noncollinear TDDFT         Noncollinear SF-TDDFT      Noncollinear Tensor TDA
                                   ├── Gradients            ├── Gradients
                                   ├── Oscillator strengths └── SOC (in progress)
                                   ├── NADC
                                   └── SOC
```

## Features

- Noncollinear DFT

  A general framework for extending collinear density functionals to noncollinear systems using the multicollinear approach.

- Noncollinear TDDFT

  A general TDDFT framework applicable to both two-component and four-component noncollinear theories.

- Noncollinear Spin-Flip TDDFT (SF-TDDFT)

  Within the noncollinear TDDFT formalism, SF-TDDFT and conventional spin-conserving TDDFT naturally emerge as two decoupled sectors for collinear reference states.
   
  - Analytic gradients
  - Oscillator strengths
  - Analytic nonadiabatic derivative couplings (NADC)
  - Spin–orbit coupling (SOC)

- Noncollinear Tensor TDA (NT-TDA)

  NT-TDA is a spin-consistent extension of noncollinear TDDFT that provides a unified treatment of spin-conserving and spin-flip excitations. For an open-shell reference state with total spin S, it can describe target states with total spins S−1 (except for S = 1/2), S, and S+1. All resulting states are free from spin contamination.

  - Analytic gradients for ΔS = −1 and 0 on ROKS and average-occupation ROKS (AOCSCF) references
  - Finite-difference gradients for all three spin channels
  - SOC *(in progress)*

  `td.Gradients().kernel(state=n)` returns the gradient of the selected total
  energy (`state=1` is the lowest NTTDA root; `state=0` is the reference).
  `td.Gradients().as_scanner(state=n)` provides the energy and gradient for geometry optimization.
  Analytic DFT gradients omit numerical grid response; finite-difference checks
  use fixed grids by default. Density fitting, X2C, solvent response, and NLC
  are not supported by the NTTDA gradient driver.
  See the [AOCSCF gradient example](examples/nttda/02_nttda_aocscf_grad.py).


## Authors

### Scientific Lead

**Yunlong Xiao**  
Associate Professor, Peking University  
Theory and project direction.  
Email: xiaoyl@pku.edu.cn

### Current Software Lead

**Tai Wang**  
Ph.D. Student, Peking University  
Software development and maintenance.  
Email: wtpeter@pku.edu.cn

### Benchmarking

**Hao Yang**  
Undergraduate Student, Peking University  
Benchmark calculations and validation.

### Module Contributors

| Module | Contributors |
|---------|--------------|
| MCfun (Multicollinear Approach) | Zhichen Pu, Hao Li |
| Noncollinear DFT | Zhichen Pu, Qiming Sun |
| Noncollinear TDDFT | Hao Li, Qiming Sun |
| Noncollinear SF-TDA / SF-TDDFT | Hao Li, Tai Wang |
| Analytic Nuclear Gradients | Hao Li, Tai Wang |
| Analytic Nonadiabatic Derivative Couplings (NADC) | Yu Jing, Tai Wang |
| Oscillator Strengths | Tai Wang |
| Noncollinear Tensor TDA (NT-TDA) | Tai Wang, Wenxian Qin |
| Spin–Orbit Coupling (SOC) | Tai Wang |
    
## Installation

From the repository root, install NEST and its runtime dependencies in editable
mode:

```bash
python -m pip install -e .
```

Install the test and lint tools for development:

```bash
python -m pip install -e ".[dev]"
```

Importing a feature module registers its methods on the corresponding PySCF
mean-field objects:

```python
from pyscf import gto
from nest import nttda, sftda

mol = gto.M(atom="H 0 0 0; H 0 0 1", spin=2)
uks = mol.UKS(xc="HF")
roks = mol.ROKS(xc="HF")
sf = uks.SFTDA()
nt = roks.NTTDA()
```

See [`examples`](examples) for complete calculations.

## License

NEST is released under the Apache License 2.0.

## Citation

If you use NEST in your research, please cite the relevant publication(s) listed below.

### Noncollinear DFT

- Multicollinear approach (PRR, 2023) https://doi.org/10.1103/PhysRevResearch.5.013036
- Matrix representation (WIREs, 2026) https://doi.org/10.1002/wcms.70063
- Nonlocal functionals (JCP, 2025) https://doi.org/10.1063/5.0260762

### Noncollinear TDDFT

- Noncollinear TDDFT and noncollinear SF-TDDFT (JCTC, 2023) https://doi.org/10.1021/acs.jctc.3c00059
- Real-time noncollinear TDDFT (JCTC, 2024) https://doi.org/10.1021/acs.jctc.4c01218

### Noncollinear SF-TDDFT

- Analytic gradients (JCTC, 2025) https://doi.org/10.1021/acs.jctc.5c00115
- Zero-excitation-energy theorem (JCTC, 2025) https://doi.org/10.1021/acs.jctc.5c00714
- Conical intersections and spin crossings (JCTC, 2025) https://doi.org/10.1021/acs.jctc.5c01272
- Analytic nonadiabatic derivative couplings (JCTC, 2026) https://doi.org/10.1021/acs.jctc.6c00556

### Noncollinear Tensor TDA (NT-TDA)

- NT-TDA (https://arxiv.org/abs/2607.19933, 2026)
