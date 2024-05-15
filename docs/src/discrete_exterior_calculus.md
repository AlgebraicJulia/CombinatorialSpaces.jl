# Discrete exterior calculus

CombinatorialSpaces.jl provides discrete differential operators defined on simplicial sets via the Discrete Exterior Calculus (DEC).

There are two modules for these DEC operators. The first, `CombinatorialSpaces.DiscreteExteriorCalculus`, serves as our reference implementation. It is suitable for any manifold-like delta dual complex. The second, `CombinatorialSpaces.FastDEC`, offers more efficient operators, both in construction, and in execution time. The operators offered by both modules agree up to differences introduced by re-ordering floating-point operations.

Certain operators in `FastDEC` are made more efficient by assuming that the delta dual complex has not been altered in any way after it is created and volumes have been assigned with `subdivide_duals!`. So, such operators should not be called on such a complex if it has been manually edited.

## API docs

```@autodocs
Modules = [ DiscreteExteriorCalculus, FastDEC ]
Private = false
```
