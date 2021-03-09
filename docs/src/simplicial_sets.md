# Simplicial sets

As a core feature, this package provides data structures and algorithms for a
flavor of simplicial sets known as *semi-simplicial sets* or *delta sets*. The
following section explains how delta sets relate to simplicial complexes and
similar structures; readers not interested in these distinctions may proceed
directly to the next section.

## Varieties of simplicial stuff

There is a large, at first bewildering, variety of concepts that fall under the
heading of "simplicial stuff," such as:

- simplicial complexes
- abstract simplicial complexes
- simplicial sets
- semi-simplicial sets, aka delta sets
- augmented simplicial sets
- symmetric (simplicial) sets

The most familiar are [simplicial
complexes](https://en.wikipedia.org/wiki/Simplicial_complex): coherent
collections of $n$-simplices of different dimensions $n$ embedded in an ambient
Euclidean space. A simplicial complex may include points $(n=0)$, line segments
$(n=1)$, triangles $(n=2)$, tetrahedra $(n=3)$, and higher-dimensional
simplices. Of the structures listed here, only simplicial complexes are
geometrical objects. All the others can be seen as combinatorial abstractions of
simplicial complexes.

[Abstract simplicial
complexes](https://en.wikipedia.org/wiki/Abstract_simplicial_complex) are the
oldest and most obvious abstraction of simplicial complexes, but nowadays
mathematicians prefer simplicial sets, which enjoy excellent algebraic
properties. A [simplicial set](https://en.wikipedia.org/wiki/Simplicial_set) $X$
consists of collections $X_n$, for $n \geq 0$, of abstract $n$-simplices whose
$n+1$ different faces are ordered and hence can be indexed by number, via the
*face maps*. In this package, we actually implement *semi-simplicial sets*,
called *delta sets* for short. The difference is that delta sets contain only
the face maps, whereas simplicial sets also contain *degeneracy maps*. The main
effect of degeneracy maps is to enlarge the space of simplicial morphisms by
allowing simplices to be "collapsed" onto lower-dimensional ones. Degeneracy
maps have their pros and cons, and in the future we will likely provide
simplicial sets as well as semi-simplicial ones. To learn more, the [paper by
Greg Friedman](https://arxiv.org/abs/0809.4221) is an excellent illustrated
introduction to semi-simplicial and simplicial sets.

Different kinds of simplicial stuff generalize different kinds of graphs from
one dimension to higher dimensions, as shown in the table below. Note that here,
as in the rest of the AlgebraicJulia ecosystem, a *graph* is a category
theorist's graph (a directed multigraph), not a simple graph.

| 1-dimensional              | $n$-dimensional                |
|----------------------------|--------------------------------|
| simple graphs              | abstract simplicial complexes  |
| graphs                     | semi-simplicial sets           |
| reflexive graphs           | simplicial sets                |
| symmetric graphs           | symmetric semi-simplicial sets |
| symmetric reflexive graphs | symmetric simplicial sets      |

### Ordered versus unordered in geometric applications

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

## API docs

```@autodocs
Modules = [ SimplicialSets ]
Private = false
```
