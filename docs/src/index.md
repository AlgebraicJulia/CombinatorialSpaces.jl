# CombinatorialSpaces.jl

This package provides combinatorial models of geometric spaces, such as
simplicial sets and combinatorial maps. These "combinatorial spaces" are useful
in computational physics, computer graphics, and other applications where
geometry plays a large role. They are also potentially useful in non-geometric
applications, since structures like simplicial sets generalize graphs from
binary relations to higher-arity relations. Combinatorial spaces are typically
*C*-sets (copresheaves) on some category *C* and are implemented here using the
general machinery for *C*-sets offered by
[Catlab.jl](https://github.com/AlgebraicJulia/Catlab.jl).

Current features include:

- delta sets (semi-simplicial sets) in dimensions one and two, optionally
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
