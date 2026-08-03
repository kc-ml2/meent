# Reference data

Golden values for `test_regression.py`. Nothing here is generated automatically
at test time — regeneration is an explicit, reviewed act.

## Layout

```
<case>.npz          # meent output: de_ri, de_ti (+ provenance keys)
reti_<case>.npz     # Reticolo output, produced locally via benchmarks/interface/
grcwa_<case>.npz
torcwa_<case>.npz
```

Every file should carry provenance alongside the arrays:

| key | meaning |
|---|---|
| `meent_version` | version that produced it |
| `backend` | 0/1/2 |
| `type_complex` | 0/1 |
| `option` | the full option dict (JSON string) |
| `source` | `meent` or the external solver + its version |

## Regenerating

```
pytest -m regression --regen-golden      # rewrites, then fails on purpose
git diff --stat tests/data/              # review before committing
```

A changed golden file is a physics change. It belongs in its own commit with the
reason in the message — not bundled into a refactor.

## External references

Reticolo needs MATLAB, GRCWA and TORCWA are heavy optional deps, so none of them
run in CI. Generate their output locally with the scripts in `benchmarks/`, store
it here, and let CI regression-test against the stored arrays.
