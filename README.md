# CombinatorialSpaces.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://AlgebraicJulia.github.io/CombinatorialSpaces.jl/stable)
[![Development Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://AlgebraicJulia.github.io/CombinatorialSpaces.jl/dev)
[![Code Coverage](https://codecov.io/gh/AlgebraicJulia/CombinatorialSpaces.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/AlgebraicJulia/CombinatorialSpaces.jl)
[![CI/CD](https://github.com/AlgebraicJulia/CombinatorialSpaces.jl/actions/workflows/julia_ci.yml/badge.svg)](https://github.com/AlgebraicJulia/CombinatorialSpaces.jl/actions/workflows/julia_ci.yml)

This package provides combinatorial models of geometric spaces, such as
simplicial sets and combinatorial maps. These "combinatorial spaces" are useful
in computational physics, computer graphics, and other applications where
geometry plays a large role. They are also potentially useful in non-geometric
applications, since structures like simplicial sets generalize graphs from
binary relations to higher-arity relations.

Combinatorial spaces, like graphs, are typically _C_-sets (copresheaves) on some
category _C_. They are implemented here using the general data structures for
_C_-sets offered by [Catlab.jl](https://github.com/AlgebraicJulia/Catlab.jl).
Thus, this package complements and extends the family of graph data structures
shipped with Catlab in the module
[`Catlab.Graphs`](https://algebraicjulia.github.io/Catlab.jl/stable/apis/graphs/).

Current features include:

- delta sets (semi-simplicial sets) in dimensions one, two, and three, optionally
  oriented and/or embedded in Euclidean space
- construction of the dual complex associated with a delta set, via
  combinatorial and geometric subdivision
- core operators of the [discrete exterior
  calculus](https://en.wikipedia.org/wiki/Discrete_exterior_calculus), including
  the boundary, exterior deriviative, Hodge star, codifferential, and
  Laplace-Beltrami operators
- experimental support for [rotation
  systems](https://www.algebraicjulia.org/blog/post/2020/09/cset-graphs-2/) and
  combinatorial maps

## Installation

To install this package, open the Julia shell, press `]` to enter Pkg mode, and
run the command

```julia
(@v1.5) pkg> add CombinatorialSpaces
```
