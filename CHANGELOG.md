## v0.1.0 (2026-07-10)

### Added

- First public release of the NEST electronic structure package, built on top of the PySCF framework.

- `nest.sftda`: Noncollinear spin-flip TDA and TDDFT (SF-TDA/SF-TDDFT) based on
  unrestricted Kohn–Sham references. Supports collinear (`col`) and multicollinear
  (`mcol`) exchange–correlation kernels. Registers `TDA_SF` and `TDDFT_SF` methods
  on PySCF UKS/UHF objects.

- `nest.nttda`: Noncollinear Tensor TDA (NT-TDA) based on ROKS references.
  Provides spin-pure excited states for ΔS = 0 and ±1 excitations. 
  Registers `NTTDA` on PySCF ROKS objects.

- `nest.grad`: Analytic nuclear gradients for SF-TDA and SF-TDDFT.

- Examples for SF-TDDFT, ROKS SF-TDDFT, NT-TDA, and SF-TDA analytic gradients
  under `examples/`.

- GitHub Actions CI: lint with ruff and test suite via pytest.

- Licensed under the Apache License 2.0.
