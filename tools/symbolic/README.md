# NTTDA algebra checks

Run from the repository root with NumPy available:

```bash
python tools/symbolic/check_nttda_delta_s_plus_one.py
```

The deterministic random-tensor check validates the spin-raising scalar,
its full MO derivative, and its explicit coordinate derivative. Its model
kernel depends on the reference density, so the MO check also exercises the
kernel derivative. This supports the
[derivation](../../docs/derivations/nttda_delta_s_plus_one_gradient.md);
physical validation uses the targeted tests in `src/nest/grad/tests/test_nttda*.py`.
