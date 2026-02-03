```@meta
Draft = false
```

# Mesh Decomposition

We can decompose meshes into overlapping submeshes using `Subobject`s
from Catlab. Given a mesh and a partition function on its vertices,
we can create a cover of the mesh by creating submeshes corresponding
to each part of the partition. The opens in the cover are by taking the non of the negation of the subobject defined by the vertices in each part.  These submeshes can overlap along their boundaries.

We can then compute intersections of these
submeshes using the `meet` operation. This allows us to analyze how the submeshes
interact and overlap. The nerve of the cover can also be constructed, which encodes the combinatorial structure of the overlaps between the submeshes.

The idea is to use this nerve to build sheaves of vector spaces over the mesh decomposition, allowing for algebraic analysis of functions defined on the mesh.


```@example mesh_decomposition
using CombinatorialSpaces
using GeometryBasics: Point3d
using LinearAlgebra: norm
using CairoMakie
using Catlab
using Catlab.CategoricalAlgebra: ACSetCategory, ACSetCat
using Catlab.CategoricalAlgebra.Subobjects: Subobject, negate, non, meet, join
using Catlab.Theories: @withmodel
using Catlab.BasicSets: FinFunction
```

We are going to draw our cover by drawing all the submeshes in orange
and the total mesh in blue. The cover submeshes will be drawn on the top left triangle of a grid of plots. The diagonal entries are the individual submeshes, and the upper triangle entries are their pairwise intersections. A pairwise intersection will be empty if the two submeshes do not overlap, and the corresponding plot will be completely blue.

```@example mesh_decomposition
function draw(mesh; color=:blue)
  f = Figure()
  ax = CairoMakie.Axis(f[1,1], aspect=1)
  draw!(ax, mesh, color=color)
  return f
end

function draw!(ax, mesh::EmbeddedDeltaDualComplex1D; color=:blue)
  ax = scatter!(ax, mesh[:point], color=color)
end

function draw!(ax, mesh::Union{EmbeddedDeltaSet2D,EmbeddedDeltaDualComplex2D}; color=:blue)
  wireframe!(ax, mesh, color=color)
  return ax
end

function draw!(ax, submesh::Subobject; color=:orange)
  ϕ = hom(submesh)
  @show nparts(dom(ϕ), :V)
  @show nparts(codom(ϕ), :V)
  draw!(ax, codom(ϕ),color=color)
  draw!(ax, dom(ϕ), color=:orange)
end

function draw(cover::Vector{T}; color=:blue, cat) where T <: Subobject
  f = Figure()
  n = length(cover)
  for i in 1:n
    for j in i:n
      ax = CairoMakie.Axis(f[i,j])
      ui,uj = cover[i], cover[j]
      @withmodel cat (meet,) begin
        draw!(ax, meet(ui,uj), color=color)
      end
    end
  end
  f
end
```

## Example 1: Quadrants of a Rectangle
First we create a triangulated grid mesh and its dual complex. Our partition function will divide the mesh into four quadrants that overlap along their boundaries.

```@example mesh_decomposition
s = triangulated_grid(100,100,15,15,Point3d);
sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}(s);
subdivide_duals!(sd, Barycenter());

# Create a category instance for working with Subobjects
# Pass the ACSet instance to ACSetCat constructor (Catlab 0.17 API)
const 𝒞 = ACSetCategory(ACSetCat(s))

f = draw(sd)
f
```

```@example mesh_decomposition
quadrants(x) = Int(x[1] > 50) + 2*Int(x[2] > 50)

function cover_mesh(partition_function, s, cat)
  vertex_partition = map(partition_function, s[:point])
  parts = map(unique(vertex_partition)) do p
    vp = findall(i->i==p, vertex_partition)
    subobj = Subobject(s; V=vp)
    @withmodel cat (negate, non) begin
      sp = non(negate(subobj))
    end
  end
  return parts
end
quads = cover_mesh(quadrants, s, 𝒞)
q = quads[1]
draw(q)
```
We can look at the individual submeshes in the cover, their joins, and their intersections.

```@example mesh_decomposition
draw(quads[3])
```

```@example mesh_decomposition
@withmodel 𝒞 (join,) begin
  q = join(quads[1], quads[3])
end
draw(q)
```

```@example mesh_decomposition
@withmodel 𝒞 (meet,) begin
  draw(meet(quads[1], quads[2]))
end
```

```@example mesh_decomposition
draw(quads; cat=𝒞)
```

The nerve of the cover can be constructed by computing all pairwise intersections of the submeshes in the cover.

```@example mesh_decomposition
using Catlab.FreeDiagrams
function nerve(cover::Vector{T}, cat) where T <: Subobject
  n = length(cover)
  @withmodel cat (meet,) begin
    map(1:n) do i
      map(i:n) do j
        ui,uj = cover[i], cover[j]
        uij = meet(ui,uj)
        (uij, i, j)
      end
    end |> Iterators.flatten
  end
end
D = nerve(quads, 𝒞)
```

## Example 2: Arcs of a Circle

We can also create a circular mesh and decompose it into overlapping arcs. Each arc will overlap with its neighbors along their endpoints.

```@example mesh_decomposition
function circle(n, c)
  mesh = EmbeddedDeltaSet1D{Bool, Point2D}()
  map(range(0, 2pi - (pi/(2^(n-1))); step=pi/(2^(n-1)))) do t
    add_vertex!(mesh, point=Point2D(cos(t),sin(t))*(c/2pi))
  end
  add_edges!(mesh, 1:(nv(mesh)-1), 2:nv(mesh))
  add_edge!(mesh, nv(mesh), 1)
  dualmesh = EmbeddedDeltaDualComplex1D{Bool, Float64, Point2D}(mesh)
  subdivide_duals!(dualmesh, Circumcenter())
  mesh,dualmesh
end
mesh,dualmesh = circle(9, 100)
```

```@example mesh_decomposition
draw(dualmesh)
```

The circle can be decomposed into four overlapping arcs using a partition function based on the quadrants of the Cartesian plane. Notice how we supply a partition function rather than a list of vertex indices. The `cover_mesh` function will compute the vertex indices for us.

```@example mesh_decomposition
function pizza_slices(x)
  (x[1] > 0) + 2*(x[2] > 0)
end

# Create a category instance for the circle mesh
# Pass the ACSet instance to ACSetCat constructor (Catlab 0.17 API)
const 𝒞₁ = ACSetCategory(ACSetCat(dualmesh))

circ_quads = cover_mesh(pizza_slices, dualmesh, 𝒞₁)
draw(circ_quads[1])
draw(circ_quads; cat=𝒞₁)
```

## Diagram Interpretation in Vect

In order to build sheaves over the mesh decomposition, we first need to create a diagram representing the cover. Each object in the diagram is a morphism in `FinSet` representing the inclusion of one submesh into another.

```@example mesh_decomposition
function finsetdiagram(cover; object=:V)
  n = length(cover)
  u1 = cover[1]
  f = hom(u1)
  X = codom(f)
  ObT = FinSet
  HomT = FinFunction
  homs = [(hom(cover[i]).components[object], i+1, 1) for i in 1:n]
  opens = dom.(hom.(cover))
  obs = [X]
  append!(obs, opens)
  obs = FinSet.(nparts.(obs, object))
  # @show obs
  # edge_homs = map(homs) do (ui, s, t)
  #   println("$s --> $t = $ui")
  # end
  diag = FreeGraph(obs, homs)
end

diag = finsetdiagram(quads)
```

The free vector space sheaf over the diagram can be constructed by composing the diagram with the free vector space functor and the appropriate pushforward or pullback operations. This creates a diagram in Vect representing the sheaf of vector spaces over the mesh decomposition. This is just the vertex component; similar constructions can be done for edges and faces.

```@example mesh_decomposition
using Catlab.Sheaves: FVect
import Catlab.Sheaves: pullback_matrix, FMatPullback, FMatPushforward

# Construct Vdiag using FMatPushforward functor as defined in Catlab
# FMatPushforward maps:
#   - Objects: FinSets to their dimensions (vector space dimensions)
#   - Morphisms: FinFunctions to pushforward matrices (transpose of pullback)
pushforward_matrix(f) = pullback_matrix(f)'
vect_obs = [length(ob) for ob in diag[:ob]]
vect_homs = [(pushforward_matrix(diag[e, :hom]), diag[e, :src], diag[e, :tgt]) 
             for e in edges(diag)]
Vdiag = FreeGraph(vect_obs, vect_homs)
```

```@example mesh_decomposition
# compose(FinDomFunctor(diag), op(FMatPullback))
Catlab.dom(m::Matrix) = FinSet(size(m, 2))
Catlab.codom(m::Matrix) = FinSet(size(m, 1))

function vectdiagram(diag)
  obs = diag[:ob]
  homs = map(enumerate(diag[edges(diag), :hom])) do (e, f)
    (pullback_matrix(f), diag[e, :src], diag[e,:tgt])
  end
  # FreeDiagram(obs, homs)
  return obs, homs
end
vectdiagram(diag)
```

## Nerve Cover Type

This data is getting rather involved, so we will encapsulate it in a `NerveCover` type for easier use and display.

```@example mesh_decomposition
import Catlab.Sheaves: AbstractCover

struct NerveCover{T, X, C} <: AbstractCover
  vertices::Dict{T, Int}
  basis::Vector{X}
  cat::C
end

function NerveCover(subobjects::Vector{X}, cat) where X <: Subobject
  lookup = enumerate(subobjects)
  vertices = Dict{Int, Int}(i=>i for (i, _) in lookup)
  return NerveCover{Int, Subobject, typeof(cat)}(vertices, subobjects, cat)
end

function NerveCover(subobjects::Dict{T,Subobject}, cat) where T
  lookup = enumerate(keys(subobjects))
  vertices = Dict{T, Int}(k=>i for (i, k) in lookup)
  opens = collect(values(subobjects))
  return NerveCover{T, Subobject, typeof(cat)}(vertices, opens, cat)
end

Base.length(K::NerveCover) = length(K.basis)

Base.show(io::IO, K::NerveCover) = begin
  print(io, "$(typeof(K)) with $(length(K)) generating opens:\n\tNV, NE, NT")
  for (i, ui) in enumerate(K.basis)
    print(io, "\n  ")
    V = nv(dom(hom(ui)))
    E = ne(dom(hom(ui)))
    T = ntriangles(dom(hom(ui)))
    print(io, "K[$i]: $V, $E, $T")
  end
end

import Catlab.CategoricalAlgebra.Pointwise.SubCSets: SubACSetComponentwise

function Base.show(io::IO, U::SubACSetComponentwise{X}) where X <: HasDeltaSet
  print(io, "Subdelta-set")
  V = nv(dom(hom(U)))
  E = ne(dom(hom(U)))
  T = ntriangles(dom(hom(U)))
  print(io, "with size $V, $E, $T")
  V = nv(codom(hom(U)))
  E = ne(codom(hom(U)))
  T = ntriangles(codom(hom(U)))
  print(io, " of object with size $V, $E, $T")
end

function Base.getindex(K::NerveCover, I::Vararg{Int})
  @withmodel K.cat (meet,) begin
    map(I) do i
      K.basis[i]
    end |> x->foldl(meet, x)
  end
end

using Combinatorics: powerset
function resolve(K::NerveCover, dim=2)
  map(powerset(K.vertices, 1, dim)) do S
    S => K[S...]
  end |> Dict
end
```

```@example mesh_decomposition
K = NerveCover(quads, 𝒞)
K[1,2]
```

```@example mesh_decomposition
resolve(K)
```

```@example mesh_decomposition
resolve(K, 3)
```
