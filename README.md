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
                                   ├── Gradients            ├── Gradients (in progress)
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

  - Analytic gradients *(in progress)*
  - SOC *(in progress)*


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

- Multicollinear approach (PRR, 2023)
- Matrix representation (WIREs, 2026)
- Nonlocal functionals (JCP, 2025)

### Noncollinear TDDFT

- Noncollinear TDDFT and noncollinear SF-TDDFT (JCTC, 2023)
- Real-time noncollinear TDDFT (JCTC, 2024)

### Noncollinear SF-TDDFT

- Analytic gradients (JCTC, 2025)
- Zero-excitation-energy theorem (JCTC, 2025)
- Conical intersections and spin crossings (JCTC, 2025)
- Analytic nonadiabatic derivative couplings (JCTC, 2026)

### Noncollinear Tensor TDA (NT-TDA)

- NT-TDA (https://arxiv.org/abs/2607.19933, 2026)
