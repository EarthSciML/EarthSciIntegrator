# nested_default: expected plan contents

Inline defaults over two index sets (`M.A`, `k × m`, 2 × 3) and three (`M.B`, `k × m × k`), neither symmetric, read by one pointwise equation, `D(u)_i = -Σ_j A[i,j] B[i,j,i] u_i`. The point of the fixture is spec/plan.md §6.1.1: each derived document declares `A` and `B` with their defaults nested as the model writes them, outermost index first:

- `M.A`: `[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]`
- `M.B`: `[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]`

in both `default_pointwise.rhs.esm` and `default_pointwise.jacobian.esm`. Before this fixture the Julia planner wrote them as one flat list in column-major order, and EarthSciAST refused to build the document (`E_TREEWALK_UNSUPPORTED_SHAPE`).

## auto

**Operator `default_pointwise`.** `locality` `{"class": "pointwise"}`; `writes` `["M.u"]`; `reads` `["M.A", "M.B", "M.u"]`; `structure.class` `block_diagonal`, one band `M.u` on `M.u` with `row_idx` = `col_idx`; `stiffness` `{"stiff", "run_config"}`.

**Schedule.** One `Rodas4` leaf with `dense_lu`, `reltol` 1e-6, `abstol` 1e-10. **Validation.** `rhs_sum`, `band_model` and `non_empty` pass.
