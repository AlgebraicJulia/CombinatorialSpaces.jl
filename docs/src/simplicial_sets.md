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

### Ordered faces in geometric applications

That the faces of each simplex in a simplicial set are ordered is convenient for
many purposes but may seem problematic for geometric applications, where the
faces usually regarded as unordered.

One solution to this problem would be to use [symmetric simplicial
sets](https://ncatlab.org/nlab/show/symmetric+set), which are simplicial sets
$X$ equipped with an action of the symmetric group $S_{n+1}$ on the
$n$-simplices $X_n$, for every $n$. This is computationally inconvenient because
every "unordered $n$-simplex" is then really an equivalence class of $(n+1)!$
different $n$-simplices, a number that grows rapidly with $n$. At this time,
symmetric simplicial sets of dimension greater than 1 are not implemented in
this package.

To simulate unordered simplicial sets, we instead adopt the convention of a
choosing the representative of the equivalence class that orders the vertices of
the simplex according to the integer IDs of the vertices. The simplicial set
then "presents" a symmetric simplicial set in a canonical way. Indeed, the
[standard method](https://ncatlab.org/nlab/show/simplicial+complex#vsSSet) of
converting an abstract simplicial complex to a simplicial set is to pick a total
ordering of its vertices. When following this convention, use the functions
[`add_sorted_edge!`](@ref) and [`glue_sorted_triangle!`](@ref), which
automatically sort their inputs to ensure that the ordering condition is
satisfied, rather than the functions [`add_edge!`](@ref) and
[`glue_triangle!`](@ref).

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

In our implementation, the generic function [`∂`](@ref) supplies all the face
maps of a delta set. Specifically, the function call `∂(i, n, x, k)` gives the
`i`-th face of the `n`-simplex in the delta set `x` with index `k`, and the call
`∂(i, n, x)` gives the `i`-faces of all `n`-simplices in the delta set `x`,
which is a vector of integers.

A finite delta set—the only kind supported here—has no simplices above a certain
dimension. For any fixed $N$, an *$N$-dimensional delta set* is a delta set $X$
such that $X_n = \emptyset$ for $n > N$. CombinatorialSpaces provides dedicated
data structures for delta sets of a given dimension.

### 1D delta sets

Since a one-dimensional delta set is the same thing as a graph, the type
[`DeltaSet1D`](@ref) has the same methods as the type `Graph` in
[`Catlab.Graphs`](https://algebraicjulia.github.io/Catlab.jl/stable/apis/graphs/),
which should be consulted for further documentation.

```@example deltaset1d
using CombinatorialSpaces # hide

dset = DeltaSet1D()
add_vertices!(dset, 4)
add_edges!(dset, [1,2,2], [2,3,4])
dset
```

One potentially confusing point is that the face map $\partial_1^0$ gives the
target vertex (the vertex of an edge opposite vertex 0), while the face map
$\partial_1^1$ gives the source vertex (the vertex of an edge opposite vertex
1).

```@example deltaset1d
@assert ∂(1,0,dset) == tgt(dset)
@assert ∂(1,1,dset) == src(dset)
```

### 2D delta sets

Two-dimensional delta sets, comprised of vertices, edges, and triangles, are
supplied by the type [`DeltaSet2D`](@ref). There are two ways to add triangles
to a delta set. If appropriately arranged edges have already been added, a
triangle having those edges as boundary can be added using the
[`add_triangle!`](@ref) function. However, it often more convenient to use the
[`glue_triangle!`](@ref) function, which takes vertices rather than edges as
arguments, creating any boundary edges that do not already exist.

For example, the following 2D delta set has the shape of a triangulated
commutative square.

```@example deltaset2d
using CombinatorialSpaces # hide

dset = DeltaSet2D()
add_vertices!(dset, 4)
glue_triangle!(dset, 1, 2, 3)
glue_triangle!(dset, 1, 4, 3)
dset
```

As the table above illustrates, only the edges of each triangle are explicitly
stored. The vertices of a triangle can be accessed using the function
[`triangle_vertices`](@ref). The correctness of this function depends on the
semi-simplicial identities.

```@example deltaset2d
map(triangles(dset)) do t
  triangle_vertices(dset, t)
end
```

## API docs

```@autodocs
Modules = [ SimplicialSets ]
Private = false
```
