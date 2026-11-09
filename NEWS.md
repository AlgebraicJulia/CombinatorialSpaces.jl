# Release notes

## Unreleased

- `PrimalGeometricMapSeries` now stores the primal subdivision meshes,
  retrievable with the new exported accessor `primal_meshes(series)`
  ([#150](https://github.com/AlgebraicJulia/CombinatorialSpaces.jl/issues/150)).
  - The struct gains a `primal_meshes` field (the two type parameters are
    unchanged). Objects serialized with the previous two-field layout cannot
    be deserialized into the new three-field struct; persist or reconstruct
    series accordingly.
  - The legacy two-argument constructor `(meshes, matrices)` is unchanged and
    stores no primal meshes; calling `primal_meshes` on such a series raises
    an explicit error.
  - `primal_meshes(series)[end]` is the mesh passed to the constructor,
    stored by reference, not copied; do not mutate it after construction.
