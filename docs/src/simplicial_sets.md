# Simplicial sets

As a core feature, this package provides data structures and algorithms for a
flavor of simplicial sets known as *semi-simplicial sets* or *delta sets*. The
first section explains how delta sets relate to simplicial complexes and other
structures. Readers not interested in these distinctions may proceed directly to
the next section, on [delta sets](#Delta-sets).

## Varieties of simplicial stuff

A wide, possibly bewildering variety of concepts fall under the heading of
"simplicial stuff," including:

- simplicial complexes
- abstract simplicial complexes
- simplicial sets
- semi-simplicial sets, aka delta sets
- augmented simplicial sets
- symmetric (simplicial) sets

The most familiar of these are [simplicial
complexes](https://en.wikipedia.org/wiki/Simplicial_complex): coherent
collections of $n$-simplices of different dimensions $n$ embedded in an ambient
Euclidean space. A simplicial complex may include points $(n=0)$, line segments
$(n=1)$, triangles $(n=2)$, tetrahedra $(n=3)$, and higher-dimensional
simplices. Of the structures listed here, only simplicial complexes are
geometrical objects. All of the others can be seen as combinatorial abstractions
of simplicial complexes.

[Abstract simplicial
complexes](https://en.wikipedia.org/wiki/Abstract_simplicial_complex) are the
oldest and most obvious abstraction of simplicial complexes, but nowadays
mathematicians tend to prefer simplicial sets, which enjoy excellent algebraic
properties. A [simplicial set](https://en.wikipedia.org/wiki/Simplicial_set) $X$
consists of sets $X_n$, for $n \geq 0$, of abstract $n$-simplices whose $n+1$
different faces are ordered and hence can be numerically indexed, via the *face
maps*.

In this package, we implement a variant of simplicial sets called
[semi-simplicial sets](https://ncatlab.org/nlab/show/semi-simplicial+set), or
[delta sets](https://en.wikipedia.org/wiki/Delta_set) for short. The difference
is that delta sets contain only the face maps, whereas simplicial sets also
contain *degeneracy maps*. The main effect of the degeneracy maps is to enlarge
the space of simplicial morphisms by allowing simplices to be "collapsed" onto
lower-dimensional ones. Degeneracy maps have their pros and cons, and in the
future we will likely provide simplicial sets as well as semi-simplicial ones.
For more details, the [paper by Greg Friedman](https://arxiv.org/abs/0809.4221)
is an excellent illustrated introduction to semi-simplicial and simplicial sets.

Simplicial sets generalize graphs from one dimension to higher dimensions. The
following table gives the precise correspondence between different flavors of
simplicial stuff and graphs.

| 1-dimensional                | $n$-dimensional               |
|------------------------------|-------------------------------|
| straight-line embedded graph | simplicial complex            |
| simple graph                 | abstract simplicial complex   |
| graph                        | semi-simplicial set           |
| reflexive graph              | simplicial set                |
| symmetric graph              | symmetric semi-simplicial set |
| symmetric reflexive graph    | symmetric simplicial set      |

!!! note

    In this table, as in this package and the rest of the AlgebraicJulia 
    ecosystem, a *graph* without qualification is always a category theorist's
    graph (a directed multigraph), not a simple graph (an undirected graph
    with no self-loops or multiple edges).

### Ordered structures in geometric applications

**TODO**: Clean up this subsection.

Simplicial sets are inherently ordered structures. The "unordered" analogue of
simplicial sets are symmetric simplicial sets, sometimes called just "symmetric
sets." In one dimension, symmetric semi-simplicial sets are symmetric graphs.

This module does not implement symmetric simplicial sets as such. However,
symmetric sets can be simulated with simplicial sets by enforcing that the
ordering of the vertices of each face matches the ordering of the integer vertex
IDs. The simplicial set then "presents" a symmetric set in a canonical way. The
functions [`add_sorted_edge!`](@ref) and [`glue_sorted_triangle!`](@ref)
automatically sort their inputs to ensure that the ordering condition is
satisfied.

## Delta sets

A *delta set* $X$ is a family of sets $X_n$ for $n = 0,1,2,\dots$, called the
*$n$-simplices*, together with functions

$$X(\partial_n^i): X_n \to X_{n-1}, \qquad n \geq 1, \quad i=0,1,\dots,n,$$

called the *face maps*, which must satisfy the *semi-simplicial identities*

$$X(\partial_{n+1}^i) \cdot X(\partial_n^j)
  = X(\partial_{n+1}^{j+1}) \cdot X(\partial_n^i): X_{n+1} \to X_{n-1},
  \qquad 0 \leq i \leq j \leq n.$$

The function $X(\partial_n^i): X_n \to X_{n-1}$ gives the face of an $n$-simplex
that is opposite its $i$-th vertex. The semi-simplicial identities then ensure
that the faces of each $n$-simplex fit together properly, for example, that the
edges of a 2-simplex actually form a triangle.

In CombinatorialSpaces, the generic function [`∂`](@ref) provides all the face
maps of a delta set. Specifically, the function call `∂(i, n, x, k)` gives the
`i`-th face of the `n`-simplex in the delta set `x` with index `k`, and the call
`∂(i, n, x)` gives the `i`-faces of all the `n`-simplices in the delta set `x`,
which is a vector of integers.

## API docs

```@autodocs
Modules = [ SimplicialSets ]
Private = false
```
