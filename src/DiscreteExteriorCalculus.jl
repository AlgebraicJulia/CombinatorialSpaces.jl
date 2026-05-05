""" The discrete exterior calculus (DEC) for simplicial sets.

This module provides the dual complex associated with a delta set (the primal
complex), which is a discrete incarnation of Hodge duality, as well as the many
operators of the DEC that depend on it, such as the Hodge star, codifferential,
wedge product, interior product, and Lie derivative. The main reference for this
module is Hirani's 2003 PhD thesis.
"""
module DiscreteExteriorCalculus
export DualSimplex, DualV, DualE, DualTri, DualTet, DualChain, DualForm,
  PrimalVectorField, DualVectorField,
  AbstractDeltaDualComplex1D, DeltaDualComplex1D, SchDeltaDualComplex1D,
  OrientedDeltaDualComplex1D, SchOrientedDeltaDualComplex1D,
  EmbeddedDeltaDualComplex1D, SchEmbeddedDeltaDualComplex1D,
  AbstractDeltaDualComplex2D, DeltaDualComplex2D, SchDeltaDualComplex2D,
  OrientedDeltaDualComplex2D, SchOrientedDeltaDualComplex2D,
  EmbeddedDeltaDualComplex2D, SchEmbeddedDeltaDualComplex2D,
  AbstractDeltaDualComplex3D, DeltaDualComplex3D, SchDeltaDualComplex3D,
  OrientedDeltaDualComplex3D, SchOrientedDeltaDualComplex3D,
  EmbeddedDeltaDualComplex3D, SchEmbeddedDeltaDualComplex3D,
  DeltaDualComplex, EmbeddedDeltaDualComplex, OrientedDeltaDualComplex,
  SimplexCenter, Barycenter, Circumcenter, Incenter, geometric_center,
  subsimplices, primal_vertex, elementary_duals, dual_boundary, dual_derivative,
  ⋆, hodge_star, ⋆⁻¹, inv_hodge_star, δ, codifferential, ∇², laplace_beltrami, Δ, laplace_de_rham,
  ♭, flat, ♭_mat, ♯, ♯_mat, sharp, ∧, wedge_product, interior_product, interior_product_flat,
  ℒ, lie_derivative, lie_derivative_flat,
  vertex_center, edge_center, triangle_center, tetrahedron_center, dual_tetrahedron_vertices, dual_triangle_vertices, dual_edge_vertices,
  dual_point, dual_volume, subdivide_duals!, DiagonalHodge, GeometricHodge,
  subdivide, PDSharp, PPSharp, AltPPSharp, DesbrunSharp, LLSDDSharp, de_sign,
  DPPFlat, PPFlat,
  ♭♯, ♭♯_mat, flat_sharp, flat_sharp_mat, dualize,
  p2_d2_interpolation, p3_d3_interpolation, eval_constant_primal_form, eval_constant_dual_form

import Base: ndims
import Base: *
import LinearAlgebra: mul!
import StaticArrays: deleteat

using GeometryBasics: Point2, Point3, Point2d, Point3d
using LinearAlgebra: Diagonal, dot, norm, cross, pinv, normalize
using SparseArrays
using StaticArrays: @SVector, SVector, SMatrix, MVector, MMatrix, StaticVector
using Statistics: mean
using Unitful: Units, dimension, @u_str

# TODO: This is not consistent with other definitions and should be removed
const Point2D = SVector{2,Float64}
const Point3D = SVector{3,Float64}

using ACSets.DenseACSets: attrtype_type
using Catlab, Catlab.CategoricalAlgebra.CSets
using Catlab.BasicSets
using Catlab.BasicSets.FinSets
using Catlab.CategoricalAlgebra.FunctorialDataMigrations: DeltaMigration, migrate
import Catlab.CategoricalAlgebra.CSets: ∧
import Catlab.Theories: Δ

using ..ArrayUtils, ..SimplicialSets
using ..SimplicialSets: CayleyMengerDet, operator_nz, ∂_nz, d_nz,
  cayley_menger, negate, numeric_sign

import ..SimplicialSets: ∂, d, volume

# This non-mutating version of deleteat returns a new (static) vector.
deleteat(vec::Vector, i) = deleteat!(copy(vec), i)

abstract type DiscreteFlat end
struct DPPFlat <: DiscreteFlat end
struct PPFlat <: DiscreteFlat end

abstract type DiscreteSharp end
struct PDSharp <: DiscreteSharp end
struct PPSharp <: DiscreteSharp end
struct AltPPSharp <: DiscreteSharp end
struct DesbrunSharp <: DiscreteSharp end
struct LLSDDSharp <: DiscreteSharp end

abstract type DiscreteHodge end
struct GeometricHodge <: DiscreteHodge end
struct DiagonalHodge  <: DiscreteHodge end

# Euclidean geometry
####################
""" A notion of "geometric center" of a simplex.

See also: [`geometric_center`](@ref).
"""
abstract type SimplexCenter end

""" Calculate the center of simplex spanned by given points.

The first argument is a list of points and the second specifies the notion of
"center", via an instance of [`SimplexCenter`](@ref).
"""
function geometric_center end

""" Barycenter, aka centroid, of a simplex.
"""
struct Barycenter <: SimplexCenter end

function geometric_center(points, ::Barycenter)
  sum(points) / length(points)
end

""" Circumcenter, or center of circumscribed circle, of a simplex.

The circumcenter is calculated by inverting the Cayley-Menger matrix, as
explained by
[Westdendorp](https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm).
This method of calculation is also used in the package
[AlphaShapes.jl](https://github.com/harveydevereux/AlphaShapes.jl).
"""
struct Circumcenter <: SimplexCenter end

function geometric_center(points, ::Circumcenter)
  CM = cayley_menger(points...)
  barycentric_coords = inv(CM)[1,2:end]
  mapreduce(*, +, barycentric_coords, points)
end

function geometric_center(points::StaticVector{N}, ::Circumcenter) where N
  CM = cayley_menger(points...)
  inv_CM = inv(CM)
  barycentric_coords = SVector(ntuple(i -> inv_CM[1, i+1], Val(N)))
  mapreduce(*, +, barycentric_coords, points)
end

""" Incenter, or center of inscribed circle, of a simplex.
"""
struct Incenter <: SimplexCenter end

function geometric_center(points, ::Incenter)
  length(points) > 2 || return geometric_center(points, Barycenter())
  face_volumes = map(i -> volume(deleteat(points, i)), eachindex(points))
  barycentric_coords = face_volumes / sum(face_volumes)
  mapreduce(*, +, barycentric_coords, points)
end

# 1D dual complex
#################

# Should be expressed using a coproduct of two copies of `SchDeltaSet1D`.

@present SchDeltaDualComplex1D <: SchDeltaSet1D begin
  # Dual vertices and edges.
  (DualV, DualE)::Ob
  (D_∂v0, D_∂v1)::Hom(DualE, DualV)

  # Centers of primal simplices are dual vertices.
  vertex_center::Hom(V, DualV)
  edge_center::Hom(E, DualV)

  # Every primal edge is subdivided into two dual edges.
  #
  # (∂v0_dual, ∂v1_dual)::Hom(E,DualE)
  #
  # ∂v0_dual ⋅ D_∂v1 == ∂v0 ⋅ vertex_center
  # ∂v1_dual ⋅ D_∂v1 == ∂v1 ⋅ vertex_center
  # ∂v0_dual ⋅ D_∂v0 == edge_center
  # ∂v1_dual ⋅ D_∂v0 == edge_center
  #
  # We could, and arguably should, track these through dedicated morphisms, as
  # in the commented code above. We don't because it scales badly in dimension:
  # an ``n``-simplex is subdivided into ``n!`` sub-simplices. So we would need 6
  # morphisms for triangles and 24 for tets, and an even larger number of
  # equations to say how everything fits together. Moreover, in practice, the
  # incidence data for the dual complex suffices to construct the dual cells of
  # primal simplices.
end

""" Abstract type for dual complex of a 1D delta set.
"""
@abstract_acset_type AbstractDeltaDualComplex1D <: HasDeltaSet1D

""" Dual complex of a one-dimensional delta set.

The data structure includes both the primal complex and the dual complex, as
well as the mapping between them.
"""
@acset_type DeltaDualComplex1D(SchDeltaDualComplex1D,
  index=[:∂v0,:∂v1,:D_∂v0,:D_∂v1]) <: AbstractDeltaDualComplex1D

""" Dual vertex corresponding to center of primal vertex.
"""
vertex_center(s::HasDeltaSet, args...) = s[args..., :vertex_center]

""" Dual vertex corresponding to center of primal edge.
"""
edge_center(s::HasDeltaSet1D, args...) = s[args..., :edge_center]

subsimplices(::Val{1}, s::HasDeltaSet1D, e::Int) =
  SVector{2}(incident(s, edge_center(s, e), :D_∂v0))

primal_vertex(::Val{1}, s::HasDeltaSet1D, e...) = s[e..., :D_∂v1]

elementary_duals(::Val{0}, s::AbstractDeltaDualComplex1D, v::Int) =
  incident(s, vertex_center(s,v), :D_∂v1)
elementary_duals(::Val{1}, s::AbstractDeltaDualComplex1D, e::Int) =
  SVector(edge_center(s,e))

""" Boundary dual vertices of a dual edge.

This accessor assumes that the simplicial identities for the dual hold.
"""
function dual_edge_vertices(s::HasDeltaSet1D, t...)
    SVector(s[t..., :D_∂v0],
            s[t..., :D_∂v1])
end


""" Boundary dual vertices of a dual triangle.

This accessor assumes that the simplicial identities for the dual hold.
"""
function dual_triangle_vertices(s::HasDeltaSet1D, t...)
  SVector(s[s[t..., :D_∂e1], :D_∂v1],
          s[s[t..., :D_∂e0], :D_∂v1],
          s[s[t..., :D_∂e0], :D_∂v0])
end

# 1D oriented dual complex
#-------------------------

@present SchOrientedDeltaDualComplex1D <: SchDeltaDualComplex1D begin
  Orientation::AttrType
  edge_orientation::Attr(E, Orientation)
  D_edge_orientation::Attr(DualE, Orientation)
end

""" Oriented dual complex of an oriented 1D delta set.
"""
@acset_type OrientedDeltaDualComplex1D(SchOrientedDeltaDualComplex1D,
  index=[:∂v0,:∂v1,:D_∂v0,:D_∂v1]) <: AbstractDeltaDualComplex1D

dual_boundary_nz(::Val{1}, s::AbstractDeltaDualComplex1D, x::Int) =
  # Boundary vertices of dual 1-cell ↔
  # Dual vertices for cofaces of (i.e. edges incident to) primal vertex.
  d_nz(Val(0), s, x)

dual_derivative_nz(::Val{0}, s::AbstractDeltaDualComplex1D, x::Int) =
  negatenz(∂_nz(Val(1), s, x))

negatenz((I, V)) = (I, negate.(V))

""" Construct 1D dual complex from 1D delta set.
"""
function (::Type{S})(s::AbstractDeltaSet1D) where S <: AbstractDeltaDualComplex1D
  t = S()
  copy_primal_1D!(t, s) # TODO: Revert to copy_parts! when performance is improved
  make_dual_simplices_1d!(t)
  return t
end

function copy_primal_1D!(t::HasDeltaSet1D, s::HasDeltaSet1D)

  @assert nv(t) == 0
  @assert ne(t) == 0

  v_range = add_parts!(t, :V, nv(s))
  e_range = add_parts!(t, :E, ne(s))

  if has_subpart(s, :point)
    @inbounds for v in v_range
      t[v, :point] = s[v, :point]
    end
  end

  @inbounds for e in e_range
    t[e, :∂v0] = s[e, :∂v0]
    t[e, :∂v1] = s[e, :∂v1]
  end

  if has_subpart(s, :edge_orientation)
    @inbounds for e in e_range
      t[e, :edge_orientation] = s[e, :edge_orientation]
    end
  end
end

make_dual_simplices_1d!(s::AbstractDeltaDualComplex1D) = make_dual_simplices_1d!(s, E(0))

""" Make dual vertices and edges for dual complex of dimension ≧ 1.

Although zero-dimensional duality is geometrically trivial (subdividing a vertex
gives back the same vertex), we treat the dual vertices as disjoint from the
primal vertices. Thus, a dual vertex is created for every primal vertex.

If the primal complex is oriented, an orientation is induced on the dual
complex. The dual edges are oriented relative to the primal edges they subdivide
(Hirani 2003, PhD thesis, Ch. 2, last sentence of Remark 2.5.1).
"""
function make_dual_simplices_1d!(s::HasDeltaSet1D, ::Simplex{n}) where n
  # Make dual vertices and edges.
  s[:vertex_center] = vcenters = add_parts!(s, :DualV, nv(s))
  s[:edge_center] = ecenters = add_parts!(s, :DualV, ne(s))
  D_edges = map((0,1)) do i
    add_parts!(s, :DualE, ne(s);
               D_∂v0 = ecenters, D_∂v1 = view(vcenters, ∂(1,i,s)))
  end

  # Orient elementary dual edges.
  if has_subpart(s, :edge_orientation)
    # If orientations are not set, then set them here.
    if any(isnothing, s[:edge_orientation])
      # 1-simplices only need to be orientable if the delta set is 1D.
      # (The 1-simplices in a 2D delta set need not represent a valid 1-Manifold.)
      if n == 1
        orient!(s, Val(1)) || error("The 1-simplices of the given 1D delta set are non-orientable.")
      else
        s[findall(isnothing, s[:edge_orientation]), :edge_orientation] = zero(attrtype_type(s, :Orientation))
      end
    end
    edge_orient = s[:edge_orientation]
    s[D_edges[1], :D_edge_orientation] = negate.(edge_orient)
    s[D_edges[2], :D_edge_orientation] = edge_orient
  end

  D_edges
end


# 1D embedded dual complex
#-------------------------

@present SchEmbeddedDeltaDualComplex1D <: SchOrientedDeltaDualComplex1D begin
  (Real, Point)::AttrType
  point::Attr(V, Point)
  length::Attr(E, Real)
  dual_point::Attr(DualV, Point)
  dual_length::Attr(DualE, Real)
end

""" Embedded dual complex of an embedded 1D delta set.

Although they are redundant information, the lengths of the primal and dual
edges are precomputed and stored.
"""
@acset_type EmbeddedDeltaDualComplex1D(SchEmbeddedDeltaDualComplex1D,
  index=[:∂v0,:∂v1,:D_∂v0,:D_∂v1]) <: AbstractDeltaDualComplex1D

""" Point associated with dual vertex of complex.
"""
dual_point(s::HasDeltaSet, args...) = s[args..., :dual_point]

struct PrecomputedVol end

volume(::Val{n}, s::EmbeddedDeltaDualComplex1D, x) where n =
  volume(Val(n), s, x, PrecomputedVol())
dual_volume(::Val{n}, s::EmbeddedDeltaDualComplex1D, x) where n =
  dual_volume(Val(n), s, x, PrecomputedVol())

volume(::Val{1}, s::HasDeltaSet1D, e, ::PrecomputedVol) = s[e, :length]
dual_volume(::Val{1}, s::HasDeltaSet1D, e, ::PrecomputedVol) =
  s[e, :dual_length]

dual_volume(::Val{1}, s::HasDeltaSet1D, e::Int, ::CayleyMengerDet) =
  volume(dual_point(s, SVector(s[e,:D_∂v0], s[e,:D_∂v1])))

hodge_diag(::Val{0}, s::AbstractDeltaDualComplex1D, v::Int) =
  sum(dual_volume(Val(1), s, elementary_duals(Val(0), s, v)))
hodge_diag(::Val{1}, s::AbstractDeltaDualComplex1D, e::Int) =
  1 / volume(Val(1), s, e)

""" Compute geometric subdivision for embedded dual complex.

Supports different methods of subdivision through the choice of geometric
center, as defined by [`geometric_center`](@ref). In particular, barycentric
subdivision and circumcentric subdivision are supported.
"""
function subdivide_duals!(sd::EmbeddedDeltaDualComplex1D{_o, _l, point_type} where {_o, _l}, alg) where point_type
  subdivide_duals_1d!(sd, point_type, alg)
  precompute_volumes_1d!(sd, point_type)
end

# TODO: Replace the individual accesses with vector accesses
function subdivide_duals_1d!(sd::HasDeltaSet1D, ::Type{point_type}, alg) where point_type

  point_arr = MVector{2, point_type}(undef)

  @inbounds for v in vertices(sd)
    sd[v, :dual_point] = sd[v, :point]
  end

  @inbounds for e in edges(sd)
    p1, p2 = edge_vertices(sd, e)
    point_arr[1] = sd[p1, :point]
    point_arr[2] = sd[p2, :point]

    sd[sd[e, :edge_center], :dual_point] = geometric_center(point_arr, alg)
  end
end

# TODO: Replace the individual accesses with vector accesses
function precompute_volumes_1d!(sd::HasDeltaSet1D, ::Type{point_type}) where point_type

  point_arr = MVector{2, point_type}(undef)

  @inbounds for e in edges(sd)
    p1, p2 = edge_vertices(sd, e)
    point_arr[1] = sd[p1, :point]
    point_arr[2] = sd[p2, :point]

    sd[e, :length] = volume(point_arr)
  end

  @inbounds for e in parts(sd, :DualE)
    p1, p2 = dual_edge_vertices(sd, e)
    point_arr[1] = sd[p1, :dual_point]
    point_arr[2] = sd[p2, :dual_point]

    sd[e, :dual_length] = volume(point_arr)
  end
end

# TODO: Orientation on subdivisions

# 2D dual complex
#################

# Should be expressed using a coproduct of two copies of `SchDeltaSet2D` or
# perhaps a pushout of `SchDeltaDualComplex2D` and `SchDeltaSet1D`.

@present SchDeltaDualComplex2D <: SchDeltaSet2D begin
  # Dual vertices, edges, and triangles.
  (DualV, DualE, DualTri)::Ob
  (D_∂v0, D_∂v1)::Hom(DualE, DualV)
  (D_∂e0, D_∂e1, D_∂e2)::Hom(DualTri, DualE)

  # Simplicial identities for dual simplices.
  D_∂e1 ⋅ D_∂v1 == D_∂e2 ⋅ D_∂v1
  D_∂e0 ⋅ D_∂v1 == D_∂e2 ⋅ D_∂v0
  D_∂e0 ⋅ D_∂v0 == D_∂e1 ⋅ D_∂v0

  # Centers of primal simplices are dual vertices.
  vertex_center::Hom(V, DualV)
  edge_center::Hom(E, DualV)
  tri_center::Hom(Tri, DualV)
end

""" Abstract type for dual complex of a 2D delta set.
"""
@abstract_acset_type AbstractDeltaDualComplex2D <: HasDeltaSet2D

""" Dual complex of a two-dimensional delta set.
"""
@acset_type DeltaDualComplex2D(SchDeltaDualComplex2D,
  index=[:∂v0,:∂v1,:∂e0,:∂e1,:∂e2,:D_∂v0,:D_∂v1,:D_∂e0,:D_∂e1,:D_∂e2]) <: AbstractDeltaDualComplex2D

""" Dual vertex corresponding to center of primal triangle.
"""
triangle_center(s::HasDeltaSet2D, args...) = s[args..., :tri_center]

subsimplices(::Val{2}, s::HasDeltaSet2D, t::Int) =
  SVector{6}(incident(s, triangle_center(s,t), @SVector [:D_∂e1, :D_∂v0]))

primal_vertex(::Val{2}, s::HasDeltaSet2D, t...) =
  primal_vertex(Val(1), s, s[t..., :D_∂e2])

elementary_duals(::Val{0}, s::AbstractDeltaDualComplex2D, v::Int) =
  incident(s, vertex_center(s,v), @SVector [:D_∂e1, :D_∂v1])
elementary_duals(::Val{1}, s::AbstractDeltaDualComplex2D, e::Int) =
  incident(s, edge_center(s,e), :D_∂v1)
elementary_duals(::Val{2}, s::AbstractDeltaDualComplex2D, t::Int) =
  SVector(triangle_center(s,t))

# 2D oriented dual complex
#-------------------------

@present SchOrientedDeltaDualComplex2D <: SchDeltaDualComplex2D begin
  Orientation::AttrType
  edge_orientation::Attr(E, Orientation)
  tri_orientation::Attr(Tri, Orientation)
  D_edge_orientation::Attr(DualE, Orientation)
  D_tri_orientation::Attr(DualTri, Orientation)
end

""" Oriented dual complex of an oriented 2D delta set.
"""
@acset_type OrientedDeltaDualComplex2D(SchOrientedDeltaDualComplex2D,
  index=[:∂v0,:∂v1,:∂e0,:∂e1,:∂e2,:D_∂v0,:D_∂v1,:D_∂e0,:D_∂e1,:D_∂e2]) <: AbstractDeltaDualComplex2D

dual_boundary_nz(::Val{1}, s::AbstractDeltaDualComplex2D, x::Int) =
  # Boundary vertices of dual 1-cell ↔
  # Dual vertices for cofaces of (triangles incident to) primal edge.
  negatenz(d_nz(Val(1), s, x))
dual_boundary_nz(::Val{2}, s::AbstractDeltaDualComplex2D, x::Int) =
  # Boundary edges of dual 2-cell ↔
  # Dual edges for cofaces of (edges incident to) primal vertex.
  d_nz(Val(0), s, x)

dual_derivative_nz(::Val{0}, s::AbstractDeltaDualComplex2D, x::Int) =
  ∂_nz(Val(2), s, x)
dual_derivative_nz(::Val{1}, s::AbstractDeltaDualComplex2D, x::Int) =
  negatenz(∂_nz(Val(1), s, x))

""" Construct 2D dual complex from 2D delta set.
"""
function (::Type{S})(s::AbstractDeltaSet2D) where S <: AbstractDeltaDualComplex2D
  t = S()
  copy_primal_2D!(t, s) # TODO: Revert to copy_parts! when performance is improved
  make_dual_simplices_2d!(t)
  return t
end

function copy_primal_2D!(t::HasDeltaSet2D, s::HasDeltaSet2D)

  @assert ntriangles(t) == 0

  copy_primal_1D!(t, s)
  tri_range = add_parts!(t, :Tri, ntriangles(s))

  @inbounds for tri in tri_range
    t[tri, :∂e0] = s[tri, :∂e0]
    t[tri, :∂e1] = s[tri, :∂e1]
    t[tri, :∂e2] = s[tri, :∂e2]
  end

  if has_subpart(s, :tri_orientation)
    @inbounds for tri in tri_range
      t[tri, :tri_orientation] = s[tri, :tri_orientation]
    end
  end
end

make_dual_simplices_1d!(s::AbstractDeltaDualComplex2D) = make_dual_simplices_1d!(s, Tri(0))

make_dual_simplices_2d!(s::AbstractDeltaDualComplex2D) = make_dual_simplices_2d!(s, Tri(0))

""" Make dual simplices for dual complex of dimension ≧ 2.

If the primal complex is oriented, an orientation is induced on the dual
complex. The elementary dual edges are oriented following (Hirani, 2003, Example
2.5.2) or (Desbrun et al, 2005, Table 1) and the dual triangles are oriented
relative to the primal triangles they subdivide.
"""
function make_dual_simplices_2d!(s::HasDeltaSet2D, ::Simplex{n}) where n
  # Fetch faces.
  ∂2 = (∂(2,0,s), ∂(2,1,s), ∂(2,2,s))
  # Make dual vertices and edges.
  D_edges01 = make_dual_simplices_1d!(s)
  s[:tri_center] = tri_centers = add_parts!(s, :DualV, ntriangles(s))
  D_edges12 = map((0,1,2)) do e
    add_parts!(s, :DualE, ntriangles(s);
               D_∂v0=tri_centers, D_∂v1=edge_center(s, ∂2[e+1]))
  end
  tri_verts = SVector(s[∂2[2], :∂v1], s[∂2[3], :∂v0], s[∂2[2], :∂v0])
  D_edges02 = map(tri_verts) do vs
    add_parts!(s, :DualE, ntriangles(s);
               D_∂v0=tri_centers, D_∂v1=vertex_center(s, vs))
  end

  # Make dual triangles.
  # Counterclockwise order in drawing with vertices 0, 1, 2 from left to right.
  D_triangle_schemas = ((0,1,1),(0,2,1),(1,2,0),(1,0,1),(2,0,0),(2,1,0))
  D_triangles = map(D_triangle_schemas) do (v,e,ev)
    add_parts!(s, :DualTri, ntriangles(s);
               D_∂e0=D_edges12[e+1], D_∂e1=D_edges02[v+1],
               D_∂e2=view(D_edges01[ev+1], ∂2[e+1]))
  end

  if has_subpart(s, :tri_orientation)
    tri_orient_buf = s[:tri_orientation]
    # If orientations are not set, then set them here.
    if any(isnothing, tri_orient_buf)
      # 2-simplices only need to be orientable if the delta set is 2D.
      # (The 2-simplices in a 3D delta set need not represent a valid 2-Manifold.)
      if n == 2
        orient!(s, Val(2)) || error("The 2-simplices of the given 2D delta set are non-orientable.")
      else
        orient_zero = zero(attrtype_type(s, :Orientation))
        @inbounds for i in eachindex(tri_orient_buf)
          if isnothing(tri_orient_buf[i])
            s[i, :tri_orientation] = orient_zero
          end
        end
      end
      tri_orient_buf = s[:tri_orientation]
    end
    # Orient elementary dual triangles.
    rev_tri_orient = negate.(tri_orient_buf)
    for (i, D_tris) in enumerate(D_triangles)
      s[D_tris, :D_tri_orientation] = isodd(i) ? rev_tri_orient : tri_orient_buf
    end

    # Orient elementary dual edges.
    for e in (0,1,2)
      s[D_edges12[e+1], :D_edge_orientation] = relative_sign.(
        s[∂2[e+1], :edge_orientation],
        isodd(e) ? rev_tri_orient : tri_orient_buf)
    end
    # Remaining dual edges are oriented arbitrarily.
    orient_one = one(attrtype_type(s, :Orientation))
    for D_edges in D_edges02
        s[D_edges, :D_edge_orientation] = orient_one
    end
  end

  D_triangles
end

relative_sign(x, y) = sign(x*y)
relative_sign(x::Bool, y::Bool) = (x && y) || (!x && !y)

# 2D embedded dual complex
#-------------------------

@present SchEmbeddedDeltaDualComplex2D <: SchOrientedDeltaDualComplex2D begin
  (Real, Point)::AttrType
  point::Attr(V, Point)
  length::Attr(E, Real)
  area::Attr(Tri, Real)
  dual_point::Attr(DualV, Point)
  dual_length::Attr(DualE, Real)
  dual_area::Attr(DualTri, Real)
end

""" Embedded dual complex of an embedded 2D delta set.

Although they are redundant information, the lengths and areas of the
primal/dual edges and triangles are precomputed and stored.
"""
@acset_type EmbeddedDeltaDualComplex2D(SchEmbeddedDeltaDualComplex2D,
  index=[:∂v0,:∂v1,:∂e0,:∂e1,:∂e2,:D_∂v0,:D_∂v1,:D_∂e0,:D_∂e1,:D_∂e2]) <: AbstractDeltaDualComplex2D

volume(::Val{n}, s::EmbeddedDeltaDualComplex2D, x) where n =
  volume(Val(n), s, x, PrecomputedVol())
dual_volume(::Val{n}, s::EmbeddedDeltaDualComplex2D, x) where n =
  dual_volume(Val(n), s, x, PrecomputedVol())

volume(::Val{2}, s::HasDeltaSet2D, t, ::PrecomputedVol) = s[t, :area]
dual_volume(::Val{2}, s::HasDeltaSet2D, t, ::PrecomputedVol) =
  s[t, :dual_area]

function dual_volume(::Val{2}, s::HasDeltaSet2D, t::Int, ::CayleyMengerDet)
  dual_vs = SVector(s[s[t, :D_∂e1], :D_∂v1],
                    s[s[t, :D_∂e2], :D_∂v0],
                    s[s[t, :D_∂e0], :D_∂v0])
  volume(dual_point(s, dual_vs))
end

hodge_diag(::Val{0}, s::AbstractDeltaDualComplex2D, v::Int) =
  sum(dual_volume(Val(2), s, elementary_duals(Val(0), s, v)))
hodge_diag(::Val{1}, s::AbstractDeltaDualComplex2D, e::Int) =
  sum(dual_volume(Val(1), s, elementary_duals(Val(1), s, e))) / volume(Val(1), s, e)
hodge_diag(::Val{2}, s::AbstractDeltaDualComplex2D, t::Int) =
  1 / volume(Val(2), s, t)

function ♭(s::AbstractDeltaDualComplex2D, X::AbstractVector, ::DPPFlat)
  # XXX: Creating this lookup table shouldn't be necessary. Of course, we could
  # index `tri_center` but that shouldn't be necessary either. Rather, we should
  # loop over incident triangles instead of the elementary duals, which just
  # happens to be inconvenient.
  tri_map = Dict{Int,Int}(triangle_center(s,t) => t for t in triangles(s))

  # TODO: Remove these comments before merging.
  # For each primal edge:
  map(edges(s)) do e
    # Get the vector from src to tgt (oriented correctly).
    e_vec = (point(s, tgt(s,e)) - point(s, src(s,e))) * sign(1,s,e)
    # Grab all the dual edges of this primal edge.
    dual_edges = elementary_duals(1,s,e)
    # And the corresponding lengths.
    dual_lengths = dual_volume(1, s, dual_edges)
    # For each of these dual edges:
    mapreduce(+, dual_edges, dual_lengths) do dual_e, dual_length
      # Get the vector at the center of the triangle this edge is pointing at.
      X_vec = X[tri_map[s[dual_e, :D_∂v0]]]
      # Take their dot product and multiply by the length of this dual edge.
      dual_length * dot(X_vec, e_vec)
      # When done, sum these weights up and divide by the total length.
    end / sum(dual_lengths)
  end
end

♭_mat(s::AbstractDeltaDualComplex2D, f::DPPFlat) =
  ♭_mat(s, ∂(2,s), f)

function ♭_mat(s::AbstractDeltaDualComplex2D, p2s, ::DPPFlat)
  mat_type = SMatrix{1, length(eltype(s[:point])), eltype(eltype(s[:point])), length(eltype(s[:point]))}
  ♭_mat = spzeros(mat_type, ne(s), ntriangles(s))
  for e in edges(s)
    # The vector associated with this primal edge.
    e_vec = (point(s, tgt(s,e)) - point(s, src(s,e))) * sign(1,s,e)
    # The triangles associated with this primal edge.
    tris = p2s[e,:].nzind
    # The dual vertex at the center of this primal edge.
    center = edge_center(s, e)
    # The centers of the triangles associated with this primal edge.
    dvs = triangle_center(s, tris)
    # The dual edge pointing to each triangle.
    des = map(dvs) do dv
      # (This is the edges(s,src,tgt) function.)
      only(de for de in incident(s, dv, :D_∂v0) if s[de, :D_∂v1] == center)
    end
    # The lengths of those dual edges.
    dels = volume(s, DualE(des))
    # The sum of the lengths of the dual edges at each primal edge.
    dels_sum = sum(dels)

    for (tri, del) in zip(tris, dels)
      ♭_mat[e, tri] = del * mat_type(e_vec) / dels_sum
    end
  end
  ♭_mat
end

function ♭(s::AbstractDeltaDualComplex2D, X::AbstractVector, ::PPFlat)
  map(edges(s)) do e
    vs = edge_vertices(s,e)
    l_vec = mean(X[vs])
    e_vec = (point(s, tgt(s,e)) - point(s, src(s,e))) * sign(1,s,e)
    dot(l_vec, e_vec)
  end
end

function ♭_mat(s::AbstractDeltaDualComplex2D, ::PPFlat)
  mat_type = SMatrix{1, length(eltype(s[:point])), eltype(eltype(s[:point])), length(eltype(s[:point]))}
  ♭_mat = spzeros(mat_type, ne(s), nv(s))
  for e in edges(s)
    e_vec = (point(s, tgt(s,e)) - point(s, src(s,e))) * sign(1,s,e)
    vs = edge_vertices(s,e)
    ♭_mat[e, vs[1]] = 0.5 * mat_type(e_vec)
    ♭_mat[e, vs[2]] = 0.5 * mat_type(e_vec)
  end
  ♭_mat
end

function ♯(s::AbstractDeltaDualComplex1D, X::AbstractVector, ::PDSharp)
  e_vecs = (s[s[:∂v0], :point] .- s[s[:∂v1], :point]) .* sign(1,s,edges(s))
  # Normalize once to undo the line integral.
  # Normalize again to compute direction of the vector.
  e_vecs .* X ./ map(x -> iszero(x) ? 1 : x, (norm.(e_vecs).^2))
end

function ♯(s::AbstractDeltaDualComplex1D, X::AbstractVector, ::PPSharp)
  dvf = ♯(s, X, PDSharp())
  map(vertices(s)) do v
    # The 1 or 2 dual edges around a primal vertex:
    des = incident(s, s[v, :vertex_center], :D_∂v1) # elementary_duals
    # The primal edges to which those dual edges belong:
    es = reduce(vcat, incident(s, s[des, :D_∂v0], :edge_center))
    weights = reverse!(normalize(s[des, :dual_length], 1))
    sum(dvf[es] .* weights)
  end
end

function ♯(s::AbstractDeltaDualComplex2D, α::AbstractVector, DS::DiscreteSharp)
  α♯ = zeros(attrtype_type(s, :Point), nv(s))
  for t in triangles(s)
    tri_center, tri_edges = triangle_center(s,t), triangle_edges(s,t)
    tri_point = dual_point(s, tri_center)
    for (i, (v₀, e₀)) in enumerate(zip(triangle_vertices(s,t), tri_edges))
      e_vec = point(s, tgt(s, e₀)) - point(s, src(s, e₀))
      e_vec /= norm(e_vec)
      e2_vec = point(s, v₀) - point(s, src(s, e₀))
      out_vec = e2_vec - dot(e2_vec, e_vec)*e_vec
      h = norm(out_vec)
      out_vec /= h^2 # length == 1/h
      for e in deleteat(tri_edges, i)
        v, sgn = src(s,e) == v₀ ? (tgt(s,e), -1) : (src(s,e), +1)
        dual_area = sum(dual_volume(2,s,d) for d in elementary_duals(0,s,v)
                        if s[s[d, :D_∂e0], :D_∂v0] == tri_center)
        area = ♯_denominator(s, v, t, DS)
        α♯[v] += sgn * sign(1,s,e) * α[e] * (dual_area / area) * out_vec
      end
    end
  end
  α♯
end

function ♯(s::AbstractDeltaDualComplex2D, α::AbstractVector, ::LLSDDSharp)
  ♯_m = ♯_mat(s, LLSDDSharp())
  ♯_m * α
end

""" Divided weighted normals by | σⁿ | .

This weighting is that used in equation 5.8.1 from Hirani.

See Hirani §5.8.
"""
♯_denominator(s::AbstractDeltaDualComplex2D, _::Int, t::Int, ::DiscreteSharp) =
  volume(2,s,t)

""" Divided weighted normals by | ⋆v | .

This weighting is NOT that of equation 5.8.1, but a different weighting scheme.
We essentially replace the denominator in equation 5.8.1 with | ⋆v | . This
may be what Hirani intended, and perhaps the denominator | σⁿ | in that equation
is either a mistake or clerical error.

See Hirani §5.8.
"""
♯_denominator(s::AbstractDeltaDualComplex2D, v::Int, _::Int, ::AltPPSharp) =
  sum(dual_volume(2,s, elementary_duals(0,s,v)))

"""    function get_orthogonal_vector(s::AbstractDeltaDualComplex2D, v::Int, e::Int)

Find a vector orthogonal to e pointing into the triangle shared with v.
"""
function get_orthogonal_vector(s::AbstractDeltaDualComplex2D, v::Int, e::Int)
  e_vec = point(s, tgt(s, e)) - point(s, src(s, e))
  e_vec /= norm(e_vec)
  e2_vec = point(s, v) - point(s, src(s, e))
  e2_vec - dot(e2_vec, e_vec)*e_vec
end

function ♯_assign!(♯_mat::AbstractSparseMatrix, s::AbstractDeltaDualComplex2D,
  v₀::Int, _::Int, t::Int, i::Int, tri_edges::SVector{3, Int}, tri_center::Int,
  out_vec, DS::DiscreteSharp)
  for e in deleteat(tri_edges, i)
    v, sgn = src(s,e) == v₀ ? (tgt(s,e), -1) : (src(s,e), +1)
    # | ⋆vₓ ∩ σⁿ |
    dual_area = sum(dual_volume(2,s,d) for d in elementary_duals(0,s,v)
                    if s[s[d, :D_∂e0], :D_∂v0] == tri_center)
    area = ♯_denominator(s, v, t, DS)
    ♯_mat[v,e] += sgn * sign(1,s,e) * (dual_area / area) * out_vec
  end
end

function ♯_assign!(♯_mat::AbstractSparseMatrix, s::AbstractDeltaDualComplex2D,
  _::Int, e₀::Int, t::Int, _::Int, _::SVector{3, Int}, tri_center::Int,
  out_vec, DS::DesbrunSharp)
  for v in edge_vertices(s, e₀)
    sgn = v == tgt(s,e₀) ? -1 : +1
    # | ⋆vₓ ∩ σⁿ |
    dual_area = sum(dual_volume(2,s,d) for d in elementary_duals(0,s,v)
                    if s[s[d, :D_∂e0], :D_∂v0] == tri_center)
    area = ♯_denominator(s, v, t, DS)
    ♯_mat[v,e₀] += sgn * sign(1,s,e₀) * (dual_area / area) * out_vec
  end
end

"""    function ♯_mat(s::AbstractDeltaDualComplex2D, DS::DiscreteSharp)

Sharpen a 1-form into a vector field.

3 primal-primal methods are supported. See [`♯_denominator`](@ref) for the distinction between Hirani's method and and an "Alternative" method. Desbrun's definition is selected with `DesbrunSharp`, and is like Hirani's, save for dividing by the norm twice.

A dual-dual method which uses linear least squares to estimate a vector field is selected with `LLSDDSharp`.
"""
function ♯_mat(s::AbstractDeltaDualComplex2D, DS::DiscreteSharp)
  ♯_mat = spzeros(attrtype_type(s, :Point), (nv(s), ne(s)))
  for t in triangles(s)
    tri_center, tri_edges = triangle_center(s,t), triangle_edges(s,t)
    for (i, (v₀, e₀)) in enumerate(zip(triangle_vertices(s,t), tri_edges))
      out_vec = get_orthogonal_vector(s, v₀, e₀)
      h = norm(out_vec)
      out_vec /= DS == DesbrunSharp() ? h : h^2
      ♯_assign!(♯_mat, s, v₀, e₀, t, i, tri_edges, tri_center, out_vec, DS)
    end
  end
  ♯_mat
end

de_sign(s,de) = s[de, :D_edge_orientation] ? +1 : -1

"""    function ♯_mat(s::AbstractDeltaDualComplex2D, ::LLSDDSharp)

Sharpen a dual 1-form into a DualVectorField, using linear least squares.

Up to floating point error, this method perfectly produces fields which are constant over any triangle in the domain. Assume that the contribution of each half-edge to the value stored on the entire dual edge is proportional to their lengths. Since this least squares method does not perform pre-normalization, the contribution of each half-edge value is proportional to its length on the given triangle. Satisfying the continuous exterior calculus, sharpened vectors are constrained to lie on their triangle (i.e. they are indeed tangent).

It is not known whether this method has been exploited previously in the DEC literature, or defined in code elsewhere.
"""
function ♯_mat(s::AbstractDeltaDualComplex2D, ::LLSDDSharp)
  # TODO: Grab point information out of s at the type level.
  pt = attrtype_type(s, :Point)
  ♯_m = spzeros(SVector{length(pt), eltype(pt)},
                findnz(d(1,s))[[1,2]]...)
  for t in triangles(s)
    tri_center, tri_edges = triangle_center(s,t), sort(triangle_edges(s,t))
    # | ⋆eₓ ∩ σⁿ |
    star_e_cap_t = map(tri_edges) do e
      only(filter(elementary_duals(1,s,e)) do de
        s[de, :D_∂v0] == tri_center
      end)
    end
    de_vecs = map(star_e_cap_t) do de
      de_sign(s,de) *
        (dual_point(s,s[de, :D_∂v0]) - dual_point(s,s[de, :D_∂v1]))
    end
    weights = s[star_e_cap_t, :dual_length] ./
      map(tri_edges) do e
        sum(s[elementary_duals(1,s,e), :dual_length])
      end
    # TODO: Move around ' as appropriate to minimize transposing.
    X = stack(de_vecs)'
    # See: https://arxiv.org/abs/1102.1845
    #QRX = qr(X, ColumnNorm())
    #LLS = (inv(QRX.R) * QRX.Q')[QRX.p,:]
    #LLS = pinv(X'*(X))*(X')
    LLS = pinv(X)
    for (i,e) in enumerate(tri_edges)
      ♯_m[t, e] = LLS[:,i]'*weights[i]
    end
  end
  ♯_m
end

# XXX: This reference implementation is kept for pedagogical purposes;
# it is faster to vectorize coefficient generation.
# Wedge product of two primal 1-forms, as in Hirani 2003, Example 7.1.2.
function ∧(::Val{1}, ::Val{1}, s::HasDeltaSet2D, α, β, x::Int)
  dual_vs = vertex_center(s, triangle_vertices(s, x))
  dual_es = sort(SVector{6}(incident(s, triangle_center(s, x), :D_∂v0)),
                 by=e -> s[e,:D_∂v1] .== dual_vs, rev=true)[1:3]
  ws = map(dual_es) do e
    sum(dual_volume(2, s, SVector{2}(incident(s, e, :D_∂e1))))
  end / volume(2, s, x)

  e0, e1, e2 = s[x, :∂e0], s[x, :∂e1], s[x, :∂e2]
  α0, α1, α2 = α[[e0, e1, e2]]
  β0, β1, β2 = β[[e0, e1, e2]]
  # Take a weighted average of co-parallelogram areas
  # at each pair of edges.
  form = sign(2, s, x) * dot(ws, SVector(
    sign(1, s, e1) * sign(1, s, e2) * (β1*α2 - α1*β2),
    sign(1, s, e0) * sign(1, s, e2) * (β0*α2 - α0*β2),
    sign(1, s, e0) * sign(1, s, e1) * (β0*α1 - α0*β1)))
  # Convert from parallelogram areas to triangles.
  form / 2
end

function subdivide_duals!(sd::EmbeddedDeltaDualComplex2D{_o, _l, point_type} where {_o, _l}, alg) where point_type
  subdivide_duals_2d!(sd, point_type, alg)
  precompute_volumes_2d!(sd, point_type)
end

# TODO: Replace the individual accesses with vector accesses
function subdivide_duals_2d!(sd::HasDeltaSet2D, ::Type{point_type}, alg) where point_type
  subdivide_duals_1d!(sd, point_type, alg)

  point_arr = MVector{3, point_type}(undef)

  @inbounds for t in triangles(sd)
    p1, p2, p3 = triangle_vertices(sd, t)
    point_arr[1] = sd[p1, :point]
    point_arr[2] = sd[p2, :point]
    point_arr[3] = sd[p3, :point]

    sd[sd[t, :tri_center], :dual_point] = geometric_center(point_arr, alg)
  end
end

function precompute_volumes_2d!(sd::HasDeltaSet2D, p::Type{point_type}) where point_type
  precompute_volumes_1d!(sd, point_type)
  set_volumes_2d!(Val(2), sd, p)
  set_dual_volumes_2d!(Val(2), sd, p)
end

# TODO: Replace the individual accesses with vector accesses
function set_volumes_2d!(::Val{2}, sd::HasDeltaSet2D, ::Type{point_type}) where point_type

  point_arr = MVector{3, point_type}(undef)

  @inbounds for t in triangles(sd)
    p1, p2, p3 = triangle_vertices(sd, t)
    point_arr[1] = sd[p1, :point]
    point_arr[2] = sd[p2, :point]
    point_arr[3] = sd[p3, :point]

    sd[t, :area] = volume(point_arr)
  end
end

# TODO: Replace the individual accesses with vector accesses
function set_dual_volumes_2d!(::Val{2}, sd::HasDeltaSet2D, ::Type{point_type}) where point_type

  point_arr = MVector{3, point_type}(undef)

  @inbounds for t in parts(sd, :DualTri)
    p1, p2, p3 = dual_triangle_vertices(sd, t)
    point_arr[1] = sd[p1, :dual_point]
    point_arr[2] = sd[p2, :dual_point]
    point_arr[3] = sd[p3, :dual_point]

    sd[t, :dual_area] = volume(point_arr)
  end
end

# 3D dual complex
#################

# Should be expressed using a coproduct of two copies of `SchDeltaSet3D`...

@present SchDeltaDualComplex3D <: SchDeltaSet3D begin
  # Dual vertices, edges, triangles, and tetrahedra.
  (DualV, DualE, DualTri, DualTet)::Ob
  (D_∂v0, D_∂v1)::Hom(DualE, DualV)
  (D_∂e0, D_∂e1, D_∂e2)::Hom(DualTri, DualE)
  (D_∂t0, D_∂t1, D_∂t2, D_∂t3)::Hom(DualTet, DualTri)

  # Simplicial identities for dual simplices.
  D_∂t3 ⋅ D_∂e2 == D_∂t2 ⋅ D_∂e2
  D_∂t3 ⋅ D_∂e1 == D_∂t1 ⋅ D_∂e2
  D_∂t3 ⋅ D_∂e0 == D_∂t0 ⋅ D_∂e2

  D_∂t2 ⋅ D_∂e1 == D_∂t1 ⋅ D_∂e1
  D_∂t2 ⋅ D_∂e0 == D_∂t0 ⋅ D_∂e1

  D_∂t1 ⋅ D_∂e0 == D_∂t0 ⋅ D_∂e0

  # Centers of primal simplices are dual vertices.
  vertex_center::Hom(V, DualV)
  edge_center::Hom(E, DualV)
  tri_center::Hom(Tri, DualV)
  tet_center::Hom(Tet, DualV)
end

""" Abstract type for dual complex of a 3D delta set.
"""
@abstract_acset_type AbstractDeltaDualComplex3D <: HasDeltaSet3D
const AbstractDeltaDualComplex = Union{AbstractDeltaDualComplex1D, AbstractDeltaDualComplex2D, AbstractDeltaDualComplex3D}
""" Dual complex of a three-dimensional delta set.
"""
@acset_type DeltaDualComplex3D(SchDeltaDualComplex3D,
  index=[:∂v0,:∂v1,:∂e0,:∂e1,:∂e2,:D_∂v0,:D_∂v1,:D_∂e0,:D_∂e1,:D_∂e2,:D_∂t0,:D_∂t1,:D_∂t2,:D_∂t3]) <: AbstractDeltaDualComplex3D

""" Dual vertex corresponding to center of primal tetrahedron.
"""
tetrahedron_center(s::HasDeltaSet3D, args...) = s[args..., :tet_center]

subsimplices(::Val{3}, s::HasDeltaSet3D, tet::Int) =
  SVector{24}(incident(s, tetrahedron_center(s,tet), @SVector [:D_∂t1, :D_∂e1, :D_∂v0]))

primal_vertex(::Val{3}, s::HasDeltaSet3D, tet...) =
  primal_vertex(Val(2), s, s[tet..., :D_∂t1])

elementary_duals(::Val{0}, s::AbstractDeltaDualComplex3D, v::Int) =
  incident(s, vertex_center(s,v), @SVector [:D_∂t1, :D_∂e1, :D_∂v1])
elementary_duals(::Val{1}, s::AbstractDeltaDualComplex3D, e::Int) =
  incident(s, edge_center(s,e), @SVector [:D_∂e1, :D_∂v1])
elementary_duals(::Val{2}, s::AbstractDeltaDualComplex3D, t::Int) =
  incident(s, triangle_center(s,t), :D_∂v1)
elementary_duals(::Val{3}, s::AbstractDeltaDualComplex3D, tet::Int) =
  SVector(tetrahedron_center(s,tet))

""" Boundary dual vertices of a dual tetrahedron.

This accessor assumes that the simplicial identities for the dual hold.
"""
function dual_tetrahedron_vertices(s::HasDeltaSet3D, t...)
  SVector(s[s[s[t..., :D_∂t2], :D_∂e2], :D_∂v1],
          s[s[s[t..., :D_∂t2], :D_∂e2], :D_∂v0],
          s[s[s[t..., :D_∂t0], :D_∂e0], :D_∂v1],
          s[s[s[t..., :D_∂t0], :D_∂e0], :D_∂v0])
end

# 3D oriented dual complex
#-------------------------

@present SchOrientedDeltaDualComplex3D <: SchDeltaDualComplex3D begin
  Orientation::AttrType
  edge_orientation::Attr(E, Orientation)
  tri_orientation::Attr(Tri, Orientation)
  tet_orientation::Attr(Tet, Orientation)
  D_edge_orientation::Attr(DualE, Orientation)
  D_tri_orientation::Attr(DualTri, Orientation)
  D_tet_orientation::Attr(DualTet, Orientation)
end

""" Oriented dual complex of an oriented 3D delta set.
"""
@acset_type OrientedDeltaDualComplex3D(SchOrientedDeltaDualComplex3D,
  index=[:∂v0,:∂v1,:∂e0,:∂e1,:∂e2,:D_∂v0,:D_∂v1,:D_∂e0,:D_∂e1,:D_∂e2,:D_∂t0,:D_∂t1,:D_∂t2,:D_∂t3]) <: AbstractDeltaDualComplex3D

dual_boundary_nz(::Val{1}, s::AbstractDeltaDualComplex3D, x::Int) =
  # Boundary vertices of dual 1-cell ↔
  # Dual vertices for cofaces of (tetrahedra incident to) primal triangle.
  d_nz(Val(2), s, x)
dual_boundary_nz(::Val{2}, s::AbstractDeltaDualComplex3D, x::Int) =
  # Boundary edges of dual 2-cell ↔
  # Dual edges for cofaces of (i.e. triangles incident to) primal edge.
  negatenz(d_nz(Val(1), s, x))
dual_boundary_nz(::Val{3}, s::AbstractDeltaDualComplex3D, x::Int) =
  # Boundary triangles of dual 3-cell ↔
  # Dual triangles for cofaces of (i.e. edges incident to) primal vertex.
  d_nz(Val(0), s, x)

dual_derivative_nz(::Val{0}, s::AbstractDeltaDualComplex3D, x::Int) =
  negatenz(∂_nz(Val(3), s, x))
dual_derivative_nz(::Val{1}, s::AbstractDeltaDualComplex3D, x::Int) =
  ∂_nz(Val(2), s, x)
dual_derivative_nz(::Val{2}, s::AbstractDeltaDualComplex3D, x::Int) =
  negatenz(∂_nz(Val(1), s, x))

""" Construct 3D dual complex from 3D delta set.
"""
function (::Type{S})(t::AbstractDeltaSet3D) where S <: AbstractDeltaDualComplex3D
  s = S()
  copy_parts!(s, t)
  make_dual_simplices_3d!(s)
  return s
end

make_dual_simplices_1d!(s::AbstractDeltaDualComplex3D) = make_dual_simplices_1d!(s, Tet(0))

make_dual_simplices_2d!(s::AbstractDeltaDualComplex3D) = make_dual_simplices_2d!(s, Tet(0))

make_dual_simplices_3d!(s::AbstractDeltaDualComplex3D) = make_dual_simplices_3d!(s, Tet(0))

# Note: these accessors are isomorphic to those for their primal counterparts.
# These can be eliminated by the DualComplex schema refactor.
add_dual_edge!(s::AbstractDeltaDualComplex3D, d_src::Int, d_tgt::Int; kw...) =
  add_part!(s, :DualE; D_∂v1=d_src, D_∂v0=d_tgt, kw...)

function get_dual_edge!(s::AbstractDeltaDualComplex3D, d_src::Int, d_tgt::Int; kw...)
  es = (e for e in incident(s, d_src, :D_∂v1) if s[e, :D_∂v0] == d_tgt)
  isempty(es) ? add_part!(s, :DualE; D_∂v1=d_src, D_∂v0=d_tgt, kw...) : first(es)
end

add_dual_triangle!(s::AbstractDeltaDualComplex3D, d_src2_first::Int, d_src2_last::Int, d_tgt2::Int; kw...) =
  add_part!(s, :DualTri; D_∂e0=d_src2_last, D_∂e1=d_tgt2, D_∂e2=d_src2_first, kw...)

function glue_dual_triangle!(s::AbstractDeltaDualComplex3D, d_v₀::Int, d_v₁::Int, d_v₂::Int; kw...)
  add_dual_triangle!(s, get_dual_edge!(s, d_v₀, d_v₁), get_dual_edge!(s, d_v₁, d_v₂),
                get_dual_edge!(s, d_v₀, d_v₂); kw...)
end

add_dual_tetrahedron!(s::AbstractDeltaDualComplex3D, d_tri0::Int, d_tri1::Int, d_tri2::Int, d_tri3::Int; kw...) =
  add_part!(s, :DualTet; D_∂t0=d_tri0, D_∂t1=d_tri1, D_∂t2=d_tri2, D_∂t3=d_tri3, kw...)

function dual_triangles(s::AbstractDeltaDualComplex3D, d_v₀::Int, d_v₁::Int, d_v₂::Int)
  d_e₀s = incident(s, d_v₂, :D_∂v0) ∩ incident(s, d_v₁, :D_∂v1)
  isempty(d_e₀s) && return Int[]
  d_e₁s = incident(s, d_v₂, :D_∂v0) ∩ incident(s, d_v₀, :D_∂v1)
  isempty(d_e₁s) && return Int[]
  d_e₂s = incident(s, d_v₁, :D_∂v0) ∩ incident(s, d_v₀, :D_∂v1)
  isempty(d_e₂s) && return Int[]
  intersect(
    incident(s, d_e₀s, :D_∂e0)...,
    incident(s, d_e₁s, :D_∂e1)...,
    incident(s, d_e₂s, :D_∂e2)...)
end

function get_dual_triangle!(s::AbstractDeltaDualComplex3D, d_v₀::Int, d_v₁::Int, d_v₂::Int)
  d_ts = dual_triangles(s, d_v₀, d_v₁, d_v₂)
  isempty(d_ts) ? glue_dual_triangle!(s, d_v₀, d_v₁, d_v₂) : first(d_ts)
end

function glue_dual_tetrahedron!(s::AbstractDeltaDualComplex3D, d_v₀::Int, d_v₁::Int, d_v₂::Int, d_v₃::Int; kw...)
  add_dual_tetrahedron!(s,
    get_dual_triangle!(s, d_v₁, d_v₂, d_v₃), # d_t₀
    get_dual_triangle!(s, d_v₀, d_v₂, d_v₃), # d_t₁
    get_dual_triangle!(s, d_v₀, d_v₁, d_v₃), # d_t₂
    get_dual_triangle!(s, d_v₀, d_v₁, d_v₂); # d_t₃
    kw...)
end

function glue_sorted_dual_tetrahedron!(s::AbstractDeltaDualComplex3D, d_v₀::Int, d_v₁::Int, d_v₂::Int, d_v₃::Int; kw...)
  d_v₀, d_v₁, d_v₂, d_v₃ = sort(SVector(d_v₀, d_v₁, d_v₂, d_v₃))
  glue_dual_tetrahedron!(s, d_v₀, d_v₁, d_v₂, d_v₃; kw...)
end

""" Make dual simplices for dual complex of dimension ≧ 3.

If the primal complex is oriented, an orientation is induced on the dual
complex.
"""
function make_dual_simplices_3d!(s::HasDeltaSet3D, ::Simplex{n}) where n
  make_dual_simplices_2d!(s)
  s[:tet_center] = add_parts!(s, :DualV, ntetrahedra(s))
  for tet in tetrahedra(s)
    tvs = tetrahedron_vertices(s,tet)
    tes = tetrahedron_edges(s,tet)
    tts = tetrahedron_triangles(s,tet)
    tc = tetrahedron_center(s,tet)
    dvs = [vertex_center(s,tvs)...,   # v₀ v₁ v₂ v₃
           edge_center(s,tes)...,     # e₀ e₁ e₂ e₃ e₄ e₅
           triangle_center(s,tts)...] # t₀ t₁ t₂ t₃
    v₀, v₁, v₂, v₃         = 1:4
    e₀, e₁, e₂, e₃, e₄, e₅ = 5:10
    t₀, t₁, t₂, t₃         = 11:14
    # Note: You could write `D_tetrahedron_schemas` using:
    #es_per_v = [(3,4,5), (1,2,5), (0,2,4), (0,1,3)]
    #ts_per_e = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    # and/or the fact that vertex vᵢ is a vertex of triangles {1,2,3,4} - {i}.

    D_tetrahedron_schemas = [
      (v₀,e₄,t₃), (v₀,e₄,t₁), (v₀,e₃,t₁), (v₀,e₃,t₂), (v₀,e₅,t₂), (v₀,e₅,t₃),
      (v₁,e₅,t₃), (v₁,e₅,t₂), (v₁,e₁,t₂), (v₁,e₁,t₀), (v₁,e₂,t₀), (v₁,e₂,t₃),
      (v₂,e₂,t₃), (v₂,e₂,t₀), (v₂,e₀,t₀), (v₂,e₀,t₁), (v₂,e₄,t₁), (v₂,e₄,t₃),
      (v₃,e₃,t₂), (v₃,e₃,t₁), (v₃,e₀,t₁), (v₃,e₀,t₀), (v₃,e₁,t₀), (v₃,e₁,t₂)]
    foreach(D_tetrahedron_schemas) do (x,y,z)
      # Exploit the fact that `glue_sorted_dual_tetrahedron!` adds only
      # necessary new dual triangles.
      glue_sorted_dual_tetrahedron!(s, dvs[x], dvs[y], dvs[z], tc)
    end
  end

  if has_subpart(s, :tet_orientation)
    if any(isnothing, s[:tet_orientation])
      # Primal 3-simplices only need to be orientable if the delta set is 3D.
      if n == 3
        orient!(s, Val(3)) || error("The 3-simplices of the given 3D delta set are non-orientable.")
      else
        # This line would be called if the complex is 4D.
        s[findall(isnothing, s[:tet_orientation]), :tet_orientation] = zero(attrtype_type(s, :Orientation))
      end
    end

    # Orient elementary dual tetrahedra.
    # Exploit the fact that triangles are added in the order of
    # D_tetrahedron_schemas.
    for tet in tetrahedra(s)
      tet_orient = s[tet, :tet_orientation]
      rev_tet_orient = negate(tet_orient)
      D_tets = (24*(tet-1)+1):(24*tet)
      s[D_tets, :D_tet_orientation] = repeat([tet_orient, rev_tet_orient], 12)
    end

    # Orient elementary dual triangles.
    for e in edges(s)
      # TODO: Perhaps multiply by tet_orientation.
      primal_edge_orient = s[e, :edge_orientation]
      d_ts = elementary_duals(1,s,e)
      s[d_ts, :D_tri_orientation] = primal_edge_orient
    end

    # Orient elementary dual edges.
    for t in triangles(s)
      # TODO: Perhaps multiply by tet_orientation.
      primal_tri_orient = s[t, :tri_orientation]
      d_es = elementary_duals(2,s,t)
      s[d_es, :D_edge_orientation] = primal_tri_orient
    end

    # Remaining dual edges and dual triangles are oriented arbitrarily.
    s[findall(isnothing, s[:D_tri_orientation]), :D_tri_orientation] = one(attrtype_type(s, :Orientation))
    # These will be dual edges from vertex_center to tc, and from
    # edge_center to tc.
    s[findall(isnothing, s[:D_edge_orientation]), :D_edge_orientation] = one(attrtype_type(s, :Orientation))
  end

  return parts(s, :DualTet)
end

# 3D embedded dual complex
#-------------------------

@present SchEmbeddedDeltaDualComplex3D <: SchOrientedDeltaDualComplex3D begin
  (Real, Point)::AttrType
  point::Attr(V, Point)
  length::Attr(E, Real)
  area::Attr(Tri, Real)
  vol::Attr(Tet, Real)
  dual_point::Attr(DualV, Point)
  dual_length::Attr(DualE, Real)
  dual_area::Attr(DualTri, Real)
  dual_vol::Attr(DualTet, Real)
end

""" Embedded dual complex of an embedded 3D delta set.

"""
@acset_type EmbeddedDeltaDualComplex3D(SchEmbeddedDeltaDualComplex3D,
  index=[:∂v0,:∂v1,:∂e0,:∂e1,:∂e2,:D_∂v0,:D_∂v1,:D_∂e0,:D_∂e1,:D_∂e2,:D_∂t0,:D_∂t1,:D_∂t2,:D_∂t3]) <: AbstractDeltaDualComplex3D

volume(::Val{n}, s::EmbeddedDeltaDualComplex3D, x) where n =
  volume(Val(n), s, x, PrecomputedVol())
dual_volume(::Val{n}, s::EmbeddedDeltaDualComplex3D, x) where n =
  dual_volume(Val(n), s, x, PrecomputedVol())

volume(::Val{3}, s::HasDeltaSet3D, tet, ::PrecomputedVol) = s[tet, :vol]
dual_volume(::Val{3}, s::HasDeltaSet3D, tet, ::PrecomputedVol) =
  s[tet, :dual_vol]

function dual_volume(::Val{3}, s::HasDeltaSet3D, tet::Int, ::CayleyMengerDet)
  dual_vs = SVector(s[s[s[tet, :D_∂t2], :D_∂e2], :D_∂v1],
                    s[s[s[tet, :D_∂t2], :D_∂e2], :D_∂v0],
                    s[s[s[tet, :D_∂t0], :D_∂e0], :D_∂v1],
                    s[s[s[tet, :D_∂t0], :D_∂e0], :D_∂v0])
  volume(dual_point(s, dual_vs))
end

hodge_diag(::Val{0}, s::AbstractDeltaDualComplex3D, v::Int) =
  sum(dual_volume(Val(3), s, elementary_duals(Val(0),s,v)))
# 1 / |⋆σᵖ| <*α,⋆σᵖ> := 1 / |σᵖ| <α,σᵖ>
hodge_diag(::Val{1}, s::AbstractDeltaDualComplex3D, e::Int) =
  sum(dual_volume(Val(2), s, elementary_duals(Val(1),s,e))) / volume(Val(1),s,e)
hodge_diag(::Val{2}, s::AbstractDeltaDualComplex3D, t::Int) =
  sum(dual_volume(Val(1), s, elementary_duals(Val(2),s,t))) / volume(Val(2),s,t)
hodge_diag(::Val{3}, s::AbstractDeltaDualComplex3D, tet::Int) =
  1 / volume(Val(3),s,tet)

# TODO: Instead of rewriting ♭_mat by replacing tris with tets, use multiple dispatch.
#function ♭_mat(s::AbstractDeltaDualComplex3D)
# TODO: Instead of rewriting ♯_mat by replacing tris with tets, use multiple dispatch.
#function ♯_mat(s::AbstractDeltaDualComplex3D, DS::DiscreteSharp)

# TODO: subdivide_duals! may also be simplified via multiple dispatch.
function subdivide_duals!(sd::EmbeddedDeltaDualComplex3D{_o, _l, point_type} where {_o, _l}, alg) where point_type
  subdivide_duals_3d!(sd, point_type, alg)
  precompute_volumes_3d!(sd, point_type)
end

function subdivide_duals_3d!(sd::HasDeltaSet3D, ::Type{point_type}, alg) where point_type
  # TODO: Double-check what gets called by subdivide_duals_2d!.
  subdivide_duals_2d!(sd, point_type, alg)

  point_arr = MVector{4, point_type}(undef)

  @inbounds for tet in tetrahedra(sd)
    p1, p2, p3, p4 = tetrahedron_vertices(sd, tet)
    point_arr[1] = sd[p1, :point]
    point_arr[2] = sd[p2, :point]
    point_arr[3] = sd[p3, :point]
    point_arr[4] = sd[p4, :point]
    sd[tetrahedron_center(sd,tet), :dual_point] = geometric_center(point_arr, alg)
  end
end

function precompute_volumes_3d!(sd::HasDeltaSet3D, p::Type{point_type}) where point_type
  precompute_volumes_2d!(sd, p)
  for tet in tetrahedra(sd)
    sd[tet, :vol] = volume(3,sd,tet,CayleyMengerDet())
  end
  for tet in parts(sd, :DualTet)
    sd[tet, :dual_vol] = dual_volume(3,sd,tet,CayleyMengerDet())
  end
end

# XXX: This reference implementation is for pedagogical purposes;
# it is faster to vectorize coefficient generation.
# Wedge product of a primal 2-form with a primal 1-form.
function ∧(::Val{2}, ::Val{1}, s::HasDeltaSet3D, α, β, x::Int)
  d_tets = subsimplices(3, s, x)
  d_volume(tets) = sum(s[tets, :dual_vol])

  # Since these weights must sum to 1, you can avoid the division by s[x, :vol],
  # and simply normalize ws w.r.t. L₁., or postpone the division until after the
  # linear combination.
  # This intersection computation would not work for a 2-1 wedge product in a 4D complex.
  ws = map(tetrahedron_vertices(s,x)) do v
    d_volume(d_tets ∩ elementary_duals(0,s,v)) / s[x, :vol]
  end

  t0, t1, t2, t3         = tetrahedron_triangles(s, x)
  e0, e1, e2, e3, e4, e5 = tetrahedron_edges(s, x)
  α0, α1, α2, α3         = α[[t0, t1, t2, t3]]
  β0, β1, β2, β3, β4, β5 = β[[e0, e1, e2, e3, e4, e5]]
  # Take a weighted average of co-parallelepiped areas at each vertex.
  #
  # Each β*α term is an edge-triangle pair that shares a single vertex vᵢ.
  # These pairs could be generated from:
  # map(x -> triangle_vertices(s, x), tetrahedron_triangles(s,x))
  # and
  # map(x -> edge_vertices(s, x), tetrahedron_edges(s,x))
  # or by thinking through the simplicial identities, of course.
  # Observe that e.g. β3 and α3 share v0, but differ in all other endpoints.
  # TODO: Replace signs with shorter variable names
  form = sign(3, s, x) * dot(ws, [
     # v₀:
     # [v3,v0][v0,v1,v2] [v2,v0][v0,v1,v3] [v1,v0][v0,v2,v3]
    sign(1, s, e3) * sign(2, s, t3) * β3*α3 + sign(1, s, e4) * sign(2, s, t2) * -β4*α2 + sign(1, s, e5) * sign(2, s, t1) * β5*α1,
     # v₁
     # [v3,v1][v0,v1,v2] [v2,v1][v0,v1,v3] [v1,v0][v1,v2,v3]
    sign(1, s, e1) * sign(2, s, t3) * β1*α3 + sign(1, s, e2) * sign(2, s, t2) * -β2*α2 + sign(1, s, e5) * sign(2, s, t0) * β5*α0,
     # v₂
     # [v3,v2][v0,v1,v2] [v2,v1][v0,v2,v3] [v2,v0][v1,v2,v3]
    sign(1, s, e0) * sign(2, s, t3) * β0*α3 + sign(1, s, e2) * sign(2, s, t1) * -β2*α1 + sign(1, s, e4) * sign(2, s, t0) * β4*α0,
     # v₃
     # [v3,v2][v0,v1,v3] [v3,v1][v0,v2,v3] [v3,v0][v1,v2,v3]
    sign(1, s, e0) * sign(2, s, t2) * β0*α2 + sign(1, s, e1) * sign(2, s, t1) * -β1*α1 + sign(1, s, e3) * sign(2, s, t0) * β3*α0])
  # Convert from parallelepiped volumes to tetrahedra.
  form / 3
end

∧(::Val{1}, ::Val{2}, s::HasDeltaSet3D, α, β, x::Int) =
  ∧(Val(2), Val(1), s, β, α, x)

# General operators
###################

""" Wrapper for dual simplex or simplices of dimension `D`.

See also: [`DualV`](@ref), [`DualE`](@ref), [`DualTri`](@ref).
"""
@parts_array_struct DualSimplex{D}

""" Vertex in simplicial set: alias for `Simplex{0}`.
"""
const DualV = DualSimplex{0}

""" Edge in simplicial set: alias for `Simplex{1}`.
"""
const DualE = DualSimplex{1}

""" Triangle in simplicial set: alias for `Simplex{2}`.
"""
const DualTri = DualSimplex{2}

""" Tetrahedron in simplicial set: alias for `Simplex{3}`.
"""
const DualTet = DualSimplex{3}

""" Wrapper for chain of dual cells of dimension `n`.

In an ``N``-dimensional complex, the elementary dual simplices of each
``n``-simplex together comprise the dual ``(N-n)``-cell of the simplex. Using
this correspondence, a basis for primal ``n``-chains defines the basis for dual
``(N-n)``-chains.

!!! note

    In (Hirani 2003, Definition 3.4.1), the duality operator assigns a certain
    sign to each elementary dual simplex. For us, all of these signs shall be
    regarded as positive because we have already incorporated them into the
    orientation of the dual simplices.
"""
@vector_struct DualChain{n}

""" Wrapper for form, aka cochain, on dual cells of dimension `n`.
"""
@vector_struct DualForm{n}

""" Wrapper for vector field on primal vertices.
"""
@vector_struct PrimalVectorField

""" Wrapper for vector field on dual vertices.
"""
@vector_struct DualVectorField

# When the user wraps a vector of SVectors in DualVectorField(), automatically
# reinterpret the result of multiplication from a vector of length 1 SVectors
# to a vector of floats.
function *(A::AbstractMatrix{T}, x::DualVectorField) where T
  reinterpret(Float64, invoke(*, Tuple{AbstractMatrix{T} where T, AbstractVector{S} where S}, A, x))
end

function mul!(C::Vector{H}, A::AbstractVecOrMat,
  B::DualVectorField, α::Number, β::Number) where {H <: Number}
  size(A, 2) == size(B, 1) || throw(DimensionMismatch())
  size(A, 1) == size(C, 1) || throw(DimensionMismatch())
  size(B, 2) == size(C, 2) || throw(DimensionMismatch())
  nzv = nonzeros(A)
  rv = rowvals(A)
  if β != 1
      #β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
      β != 0 ? rmul!(C, β) : fill!(C, zero(H))
  end
  for k in 1:size(C, 2)
      @inbounds for col in 1:size(A, 2)
          αxj = B[col,k] * α
          for j in nzrange(A, col)
              #C[rv[j], k] += nzv[j]*αxj
              C[rv[j], k] += only(nzv[j]*αxj)
          end
      end
  end
  C
end

ndims(s::AbstractDeltaDualComplex1D) = 1
ndims(s::AbstractDeltaDualComplex2D) = 2
ndims(s::AbstractDeltaDualComplex3D) = 3

volume(s::HasDeltaSet, x::DualSimplex{n}, args...) where n =
  dual_volume(Val(n), s, x.data, args...)
@inline dual_volume(n::Int, s::HasDeltaSet, args...) =
  dual_volume(Val(n), s, args...)

""" List of dual simplices comprising the subdivision of a primal simplex.

A primal ``n``-simplex is always subdivided into ``(n+1)!`` dual ``n``-simplices,
not be confused with the [`elementary_duals`](@ref) which have complementary
dimension.

The returned list is ordered such that subsimplices with the same primal vertex
appear consecutively.
"""
subsimplices(s::HasDeltaSet, x::Simplex{n}) where n =
  DualSimplex{n}(subsimplices(Val(n), s, x.data))
@inline subsimplices(n::Int, s::HasDeltaSet, args...) =
  subsimplices(Val(n), s, args...)

""" Primal vertex associated with a dual simplex.
"""
primal_vertex(s::HasDeltaSet, x::DualSimplex{n}) where n =
  V(primal_vertex(Val(n), s, x.data))
@inline primal_vertex(n::Int, s::HasDeltaSet, args...) =
  primal_vertex(Val(n), s, args...)

""" List of elementary dual simplices corresponding to primal simplex.

In general, in an ``n``-dimensional complex, the elementary duals of primal
``k``-simplices are dual ``(n-k)``-simplices. Thus, in 1D dual complexes, the
elementary duals of...

- primal vertices are dual edges
- primal edges are (single) dual vertices

In 2D dual complexes, the elementary duals of...

- primal vertices are dual triangles
- primal edges are dual edges
- primal triangles are (single) dual vertices

In 3D dual complexes, the elementary duals of...

- primal vertices are dual tetrahedra
- primal edges are dual triangles
- primal triangles are dual edges
- primal tetrahedra are (single) dual vertices
"""
elementary_duals(s::HasDeltaSet, x::Simplex{n}) where n =
  DualSimplex{ndims(s)-n}(elementary_duals(Val(n), s, x.data))
@inline elementary_duals(n::Int, s::HasDeltaSet, args...) =
  elementary_duals(Val(n), s, args...)

""" Boundary of chain of dual cells.

Transpose of [`dual_derivative`](@ref).
"""
@inline dual_boundary(n::Int, s::HasDeltaSet, args...) =
  dual_boundary(Val(n), s, args...)
∂(s::HasDeltaSet, x::DualChain{n}) where n =
  DualChain{n-1}(dual_boundary(Val(n), s, x.data))

function dual_boundary(::Val{n}, s::HasDeltaSet, args...) where n
  operator_nz(Int, nsimplices(ndims(s)-n+1,s),
              nsimplices(ndims(s)-n,s), args...) do x
    dual_boundary_nz(Val(n), s, x)
  end
end

""" Discrete exterior derivative of dual form.

Transpose of [`dual_boundary`](@ref). For more info, see (Desbrun, Kanso, Tong,
2008: Discrete differential forms for computational modeling, §4.5).
"""
@inline dual_derivative(n::Int, s::HasDeltaSet, args...) =
  dual_derivative(Val(n), s, args...)
d(s::HasDeltaSet, x::DualForm{n}) where n =
  DualForm{n+1}(dual_derivative(Val(n), s, x.data))

function dual_derivative(::Val{n}, s::HasDeltaSet, args...) where n
  operator_nz(Int, nsimplices(ndims(s)-n-1,s),
              nsimplices(ndims(s)-n,s), args...) do x
    dual_derivative_nz(Val(n), s, x)
  end
end

# TODO: Determine whether an ACSetType is Embedded in a more principled way.
"""
Checks whether a DeltaSet is embedded by  searching for 'Embedded' in the name
of its type. This could also check for 'Point' in the schema, which
would feel better but be less trustworthy.
"""
is_embedded(d::HasDeltaSet) = is_embedded(typeof(t))
is_embedded(t::Type{T}) where {T<:HasDeltaSet} = !isnothing(findfirst("Embedded",string(t.name.name)))
const REPLACEMENT_FOR_DUAL_TYPE = "Set" => "DualComplex"
rename_to_dual(s::Symbol) = Symbol(replace(string(s),REPLACEMENT_FOR_DUAL_TYPE))
rename_from_dual(s::Symbol) = Symbol(replace(string(s),reverse(REPLACEMENT_FOR_DUAL_TYPE)))

const EmbeddedDeltaSet = Union{EmbeddedDeltaSet1D,EmbeddedDeltaSet2D,EmbeddedDeltaSet3D}
const EmbeddedDeltaDualComplex = Union{EmbeddedDeltaDualComplex1D,EmbeddedDeltaDualComplex2D}

"""
Adds the Real type for lengths in the EmbeddedDeltaSet case, and removes it in the EmbeddedDeltaDualComplex case.
Will need further customization
if we add another type whose dual has different parameters than its primal.
"""
dual_param_list(d::HasDeltaSet) = typeof(d).parameters
dual_param_list(d::EmbeddedDeltaSet) =
  begin t = typeof(d) ; [t.parameters[1],eltype(t.parameters[2]),t.parameters[2]] end
dual_param_list(d::EmbeddedDeltaDualComplex) =
  begin t = typeof(d); [t.parameters[1],t.parameters[3]] end

"""
Keys are symbols for all the DeltaSet and DeltaDualComplex types.
Values are the types themselves, without parameters, so mostly UnionAlls.
Note there aren't any 0D or 3D types in here thus far.
"""
type_dict = Dict{Symbol,Type}()
const prefixes = ["Embedded","Oriented",""]
const postfixes = ["1D","2D"]
const midfixes = ["DeltaDualComplex","DeltaSet"]
for (pre,mid,post) in Iterators.product(prefixes, midfixes, postfixes)
  s = Symbol(pre,mid,post)
  type_dict[s] = eval(s)
end

"""
Get the dual type of a plain, oriented, or embedded DeltaSet1D or 2D.
Will always return a `DataType`, i.e. any parameters will be evaluated.
"""
function dual_type(d::HasDeltaSet)
  n = type_dict[rename_to_dual(typeof(d).name.name)]
  ps = dual_param_list(d)
  length(ps) > 0 ? n{ps...} : n
end
function dual_type(d::AbstractDeltaDualComplex)
  n = type_dict[rename_from_dual(typeof(d).name.name)]
  ps = dual_param_list(d)
  length(ps) > 0 ? n{ps...} : n
end

"""
Calls the constructor for d's dual type on d, including parameters.
Does not call `subdivide_duals!` on the result.
Should work out of the box on new DeltaSet types if (1) their dual type
has the same name as their primal type with "Set" substituted by "DualComplex"
and (2) their dual type has the same parameter set as their primal type. At the
time of writing (PR 117) only "Embedded" types fail criterion (2) and get special treatment.

# Examples
s = EmbeddedDeltaSet2D{Bool,SVector{2,Float64}}()
dualize(s)::EmbeddedDeltaDualComplex2D{Bool,Float64,SVector{2,Float64}}
"""
dualize(d::HasDeltaSet) = dual_type(d)(d)
function dualize(d::HasDeltaSet,center::SimplexCenter)
  dd = dualize(d)
  subdivide_duals!(dd,center)
  dd
end

"""
Get the acset schema, as a Presentation, of a HasDeltaSet.
XXX: upstream to Catlab.
"""
fancy_acset_schema(d::HasDeltaSet) = Presentation(acset_schema(d))

""" Hodge star operator from primal ``n``-forms to dual ``N-n``-forms.

!!! note

    Some authors, such as (Hirani 2003) and (Desbrun 2005), use the symbol ``⋆``
    for the duality operator on chains and the symbol ``*`` for the Hodge star
    operator on cochains. We do not explicitly define the duality operator and
    we use the symbol ``⋆`` for the Hodge star.
"""
⋆(s::HasDeltaSet, x::SimplexForm{n}; kw...) where n =
  DualForm{ndims(s)-n}(⋆(Val(n), s, x.data; kw...))
""" Hodge star on a primal form with an explicit Unitful unit annotation.
"""
⋆(s::HasDeltaSet, x::SimplexForm{n}, unit::Units; kw...) where n =
  DualForm{ndims(s)-n}(⋆(Val(n), s, x.data, unit; kw...))
@inline ⋆(n::Int, s::HasDeltaSet, args...; kw...) =
  ⋆(Val(n), s, args...; kw...)
@inline ⋆(::Val{n}, s::HasDeltaSet;
          hodge::DiscreteHodge=GeometricHodge()) where n =
  ⋆(Val(n), s, hodge)
@inline ⋆(::Val{n}, s::HasDeltaSet, form::AbstractVector;
          hodge::DiscreteHodge=GeometricHodge()) where n =
  ⋆(Val(n), s, form, hodge)

⋆(::Val{n}, s::HasDeltaSet, form::AbstractVector, ::DiagonalHodge) where n =
  applydiag(form) do x, a; a * hodge_diag(Val(n),s,x) end
⋆(::Val{n}, s::HasDeltaSet, ::DiagonalHodge) where n =
  Diagonal([ hodge_diag(Val(n),s,x) for x in simplices(n,s) ])

# Note that this cross product defines the positive direction for flux to
# always be in the positive z direction. This will likely not generalize to
# arbitrary meshes embedded in 3D space, and so will need to be revisited.
# Potentially this orientation can be provided by the simplicial triangle
# orientation?
# TODO: Revisit this assumption based on changes to `orient!`.
crossdot(v1, v2) = begin
  v1v2 = cross(v1, v2)
  norm(v1v2) * (last(v1v2) == 0 ? 1.0 : sign(last(v1v2)))
end

""" Hodge star operator from primal 1-forms to dual 1-forms.

This specific hodge star implementation is based on the hodge star presented in
(Ayoub et al 2020), which generalizes the operator presented in (Hirani 2003).
This reproduces the diagonal hodge for a dual mesh generated under
circumcentric subdivision and provides off-diagonal correction factors for
meshes generated under other subdivision schemes (e.g. barycentric).
"""
function ⋆(::Val{1}, s::AbstractDeltaDualComplex2D, ::GeometricHodge)

  vals = Dict{Tuple{Int64, Int64}, Float64}()
  I = Vector{Int64}()
  J = Vector{Int64}()
  V = Vector{Float64}()

  rel_orient = 0.0
  for t in triangles(s)
    e = reverse(triangle_edges(s, t))
    ev = point(s, tgt(s, e)) .- point(s, src(s,e))

    tc = dual_point(s, triangle_center(s, t))
    dv = map(enumerate(dual_point(s, edge_center(s, e)))) do (i,v)
      (tc - v) * (i == 2 ? -1 : 1)
    end

    diag_dot = map(1:3) do i
      dot(ev[i], dv[i]) / dot(ev[i], ev[i])
    end

    # This relative orientation needs to be redefined for each triangle in the
    # case that the mesh has multiple independent connected components
    rel_orient = 0.0
    for i in 1:3
      diag_cross = sign(Val(2), s, t) * crossdot(ev[i], dv[i]) /
                      dot(ev[i], ev[i])
      if diag_cross != 0.0
        # Decide the orientation of the mesh relative to z-axis (see crossdot)
        # For optimization, this could be moved out of this loop
        if rel_orient == 0.0
          rel_orient = sign(diag_cross)
        end

        push!(I, e[i])
        push!(J, e[i])
        push!(V, diag_cross * rel_orient)
      end
    end

    for p ∈ ((1,2,3), (1,3,2), (2,1,3),
             (2,3,1), (3,1,2), (3,2,1))
      val = rel_orient * sign(Val(2), s, t) * diag_dot[p[1]] *
              dot(ev[p[1]], ev[p[3]]) / crossdot(ev[p[2]], ev[p[3]])
      if val != 0.0
        push!(I, e[p[1]])
        push!(J, e[p[2]])
        push!(V, val)
      end
    end
  end
  sparse(I,J,V)
end

⋆(::Val{0}, s::AbstractDeltaDualComplex2D, ::GeometricHodge) =
  ⋆(Val(0), s, DiagonalHodge())
⋆(::Val{2}, s::AbstractDeltaDualComplex2D, ::GeometricHodge) =
  ⋆(Val(2), s, DiagonalHodge())

⋆(::Val{0}, s::AbstractDeltaDualComplex2D, form::AbstractVector, ::GeometricHodge) =
  ⋆(Val(0), s, form, DiagonalHodge())
⋆(::Val{1}, s::AbstractDeltaDualComplex2D, form::AbstractVector, ::GeometricHodge) =
  ⋆(Val(1), s, GeometricHodge()) * form
⋆(::Val{2}, s::AbstractDeltaDualComplex2D, form::AbstractVector, ::GeometricHodge) =
  ⋆(Val(2), s, form, DiagonalHodge())

⋆(::Val{n}, s::AbstractDeltaDualComplex1D, ::GeometricHodge) where n =
  ⋆(Val(n), s, DiagonalHodge())
⋆(::Val{n}, s::AbstractDeltaDualComplex1D, form::AbstractVector, ::GeometricHodge) where n =
  ⋆(Val(n), s, form, DiagonalHodge())

@inline function ⋆(::Val{0}, s::AbstractDeltaDualComplex1D, unit::Units;
                   hodge::DiscreteHodge=GeometricHodge())
  hdg = ⋆(Val(0), s, hodge)
  Diagonal(hdg.diag .* unit)
end

@inline function ⋆(::Val{0}, s::AbstractDeltaDualComplex1D,
                   form::AbstractVector, unit::Units;
                   hodge::DiscreteHodge=GeometricHodge())
  ⋆(Val(0), s, unit; hodge) * form
end

""" Alias for the Hodge star operator [`⋆`](@ref).
"""
const hodge_star = ⋆

""" Inverse Hodge star operator from dual ``N-n``-forms to primal ``n``-forms.

Confusingly, this is *not* the operator inverse of the Hodge star [`⋆`](@ref)
because it carries an extra global sign, in analogy to the smooth case
(Gillette, 2009, Notes on the DEC, Definition 2.27).
"""
@inline inv_hodge_star(n::Int, s::HasDeltaSet, args...; kw...) =
  inv_hodge_star(Val(n), s, args...; kw...)
@inline inv_hodge_star(::Val{n}, s::HasDeltaSet;
                       hodge::DiscreteHodge=GeometricHodge()) where n =
  inv_hodge_star(Val(n), s, hodge)
@inline inv_hodge_star(::Val{n}, s::HasDeltaSet, form::AbstractVector;
                       hodge::DiscreteHodge=GeometricHodge()) where n =
  inv_hodge_star(Val(n), s, form, hodge)

function inv_hodge_star(::Val{n}, s::HasDeltaSet,
                        form::AbstractVector, ::DiagonalHodge) where n
  if iseven(n*(ndims(s)-n))
    applydiag(form) do x, a; a / hodge_diag(Val(n),s,x) end
  else
    applydiag(form) do x, a; -a / hodge_diag(Val(n),s,x) end
  end
end

function inv_hodge_star(::Val{n}, s::HasDeltaSet, ::DiagonalHodge) where n
  if iseven(n*(ndims(s)-n))
    Diagonal([ 1 / hodge_diag(Val(n),s,x) for x in simplices(n,s) ])
  else
    Diagonal([ -1 / hodge_diag(Val(n),s,x) for x in simplices(n,s) ])
  end
end

function inv_hodge_star(::Val{1}, s::AbstractDeltaDualComplex2D,
                        ::GeometricHodge)
  -1 * inv(Matrix(⋆(Val(1), s, GeometricHodge())))
end
function inv_hodge_star(::Val{1}, s::AbstractDeltaDualComplex2D,
                        form::AbstractVector, ::GeometricHodge)
  -1 * (Matrix(⋆(Val(1), s, GeometricHodge())) \ form)
end

inv_hodge_star(::Val{0}, s::AbstractDeltaDualComplex2D, ::GeometricHodge) =
  inv_hodge_star(Val(0), s, DiagonalHodge())
inv_hodge_star(::Val{2}, s::AbstractDeltaDualComplex2D, ::GeometricHodge) =
  inv_hodge_star(Val(2), s, DiagonalHodge())

inv_hodge_star(::Val{0}, s::AbstractDeltaDualComplex2D,
               form::AbstractVector, ::GeometricHodge) =
  inv_hodge_star(Val(0), s, form, DiagonalHodge())
inv_hodge_star(::Val{2}, s::AbstractDeltaDualComplex2D,
               form::AbstractVector, ::GeometricHodge) =
  inv_hodge_star(Val(2), s, form, DiagonalHodge())

inv_hodge_star(::Val{n}, s::AbstractDeltaDualComplex1D,
               ::GeometricHodge) where n =
  inv_hodge_star(Val(n), s, DiagonalHodge())
inv_hodge_star(::Val{n}, s::AbstractDeltaDualComplex1D,
               form::AbstractVector, ::GeometricHodge) where n =
  inv_hodge_star(Val(n), s, form, DiagonalHodge())

@inline function inv_hodge_star(::Val{0}, s::AbstractDeltaDualComplex1D,
                                unit::Units;
                                hodge::DiscreteHodge=GeometricHodge())
  ihdg = inv_hodge_star(Val(0), s, hodge)
  Diagonal(ihdg.diag .* unit)
end

@inline function inv_hodge_star(::Val{0}, s::AbstractDeltaDualComplex1D,
                                form::AbstractVector, unit::Units;
                                hodge::DiscreteHodge=GeometricHodge())
  inv_hodge_star(Val(0), s, unit; hodge) * form
end

@inline function inv_hodge_star(::Val{1}, s::AbstractDeltaDualComplex1D,
                                unit::Units;
                                hodge::DiscreteHodge=GeometricHodge())
  ihdg = inv_hodge_star(Val(1), s, hodge)
  Diagonal(ihdg.diag .* unit)
end

@inline function inv_hodge_star(::Val{1}, s::AbstractDeltaDualComplex1D,
                                form::AbstractVector, unit::Units;
                                hodge::DiscreteHodge=GeometricHodge())
  inv_hodge_star(Val(1), s, unit; hodge) * form
end

""" Alias for the inverse Hodge star operator [`⋆⁻¹`](@ref).
"""
const ⋆⁻¹ = inv_hodge_star

""" Codifferential operator from primal ``n`` forms to primal ``n-1``-forms.
"""
δ(s::HasDeltaSet, x::SimplexForm{n}; kw...) where n =
  SimplexForm{n-1}(δ(Val(n), s, GeometricHodge(), x.data; kw...))
@inline δ(n::Int, s::HasDeltaSet, args...; kw...) =
  δ(Val(n), s, args...; kw...)
@inline δ(::Val{n}, s::HasDeltaSet; hodge::DiscreteHodge=GeometricHodge(),
          matrix_type::Type=SparseMatrixCSC{Float64}) where n =
  δ(Val(n), s, hodge, matrix_type)
@inline δ(::Val{n}, s::HasDeltaSet, form::AbstractVector;
          hodge::DiscreteHodge=GeometricHodge()) where n =
  δ(Val(n), s, hodge, form)

function δ(::Val{n}, s::HasDeltaSet, ::DiagonalHodge, args...) where n
  # The sign of δ in Gillette's notes (see test file) is simply a product of
  # the signs for the inverse hodge and dual derivative involved.
  sgn = iseven((n-1)*(ndims(s)*(n-1) + 1)) ? +1 : -1
  operator_nz(Float64, nsimplices(n-1,s), nsimplices(n,s), args...) do x
    c = hodge_diag(Val(n), s, x)
    I, V = dual_derivative_nz(Val(ndims(s)-n), s, x)
    V = map(I, V) do i, a
      sgn * c * a / hodge_diag(Val(n-1), s, i)
    end
    (I, V)
  end
end

function δ(::Val{n}, s::HasDeltaSet, ::GeometricHodge, matrix_type) where n
  inv_hodge_star(n-1, s) * dual_derivative(ndims(s)-n, s) * ⋆(n, s)
end

function δ(::Val{n}, s::HasDeltaSet, ::GeometricHodge, form::AbstractVector) where n
  Vector(inv_hodge_star(n - 1, s, dual_derivative(ndims(s)-n, s, ⋆(n, s, form))))
end

""" Alias for the codifferential operator [`δ`](@ref).
"""
const codifferential = δ

""" Laplace-Beltrami operator on discrete forms.

This linear operator on primal ``n``-forms defined by ``∇² α := -δ d α``, where
[`δ`](@ref) is the codifferential and [`d`](@ref) is the exterior derivative.

!!! note

    For following texts such as Abraham-Marsden-Ratiu, we take the sign
    convention that makes the Laplace-Beltrami operator consistent with the
    Euclidean Laplace operator (the divergence of the gradient). Other authors,
    such as (Hirani 2003), take the opposite convention, which has the advantage
    of being consistent with the Laplace-de Rham operator [`Δ`](@ref).
"""
∇²(s::HasDeltaSet, x::SimplexForm{n}; kw...) where n =
  SimplexForm{n}(∇²(Val(n), s, x.data; kw...))
@inline ∇²(n::Int, s::HasDeltaSet, args...; kw...) =
  ∇²(Val(n), s, args...; kw...)

∇²(::Val{n}, s::HasDeltaSet, form::AbstractVector; kw...) where n =
  -δ(n+1, s, d(Val(n), s, form); kw...)
∇²(::Val{n}, s::HasDeltaSet; matrix_type::Type=SparseMatrixCSC{Float64}, kw...) where n =
  -δ(n+1, s; matrix_type=matrix_type, kw...) * d(Val(n), s, matrix_type)

""" Alias for the Laplace-Beltrami operator [`∇²`](@ref).
"""
const laplace_beltrami = ∇²

""" Laplace-de Rham operator on discrete forms.

This linear operator on primal ``n``-forms is defined by ``Δ := δ d + d δ``.
Restricted to 0-forms, it reduces to the negative of the Laplace-Beltrami
operator [`∇²`](@ref): ``Δ f = -∇² f``.
"""
Δ(s::HasDeltaSet, x::SimplexForm{n}; kw...) where n =
  SimplexForm{n}(Δ(Val(n), s, x.data; kw...))
@inline Δ(n::Int, s::HasDeltaSet, args...; kw...) =
  Δ(Val(n), s, args...; kw...)

Δ(::Val{0}, s::HasDeltaSet, form::AbstractVector; kw...) =
  δ(1, s, d(Val(0), s, form); kw...)
Δ(::Val{0}, s::HasDeltaSet; matrix_type::Type=SparseMatrixCSC{Float64}, kw...) =
  δ(1, s; matrix_type=matrix_type, kw...) * d(Val(0), s, matrix_type)

Δ(::Val{n}, s::HasDeltaSet, form::AbstractVector; kw...) where n =
  δ(n+1, s, d(Val(n), s, form); kw...) + d(Val(n-1), s, δ(n, s, form; kw...))
Δ(::Val{n}, s::HasDeltaSet; matrix_type::Type=SparseMatrixCSC{Float64}, kw...) where n =
  δ(n+1, s; matrix_type=matrix_type, kw...) * d(Val(n), s, matrix_type) +
		d(Val(n-1), s, matrix_type) * δ(n, s; matrix_type=matrix_type, kw...)

Δ(::Val{1}, s::AbstractDeltaDualComplex1D, form::AbstractVector; kw...) =
  d(Val(0), s, δ(1, s, form; kw...))
Δ(::Val{1}, s::AbstractDeltaDualComplex1D; matrix_type::Type=SparseMatrixCSC{Float64}, kw...) =
  d(Val(0), s, matrix_type) * δ(1, s; matrix_type=matrix_type, kw...)

Δ(::Val{2}, s::AbstractDeltaDualComplex2D, form::AbstractVector; kw...) =
  d(Val(1), s, δ(2, s, form; kw...))
Δ(::Val{2}, s::AbstractDeltaDualComplex2D; matrix_type::Type=SparseMatrixCSC{Float64}, kw...) =
  d(Val(1), s, matrix_type) * δ(2, s; matrix_type=matrix_type, kw...)
""" Alias for the Laplace-de Rham operator [`Δ`](@ref).
"""
const laplace_de_rham = Δ

""" Flat operator converting vector fields to 1-forms.

A generic function for discrete flat operators. Currently the DPP-flat from
(Hirani 2003, Definition 5.5.2) and (Desbrun et al 2005, Definition 7.3) is
implemented,
as well as a primal-to-primal flat, which assumes linear-interpolation of the
vector field across an edge, determined solely by the values at the endpoints.

See also: the sharp operator [`♯`](@ref).
"""
♭(s::HasDeltaSet, X::DualVectorField) = EForm(♭(s, X.data, DPPFlat()))

""" Alias for the flat operator [`♭`](@ref).
"""
const flat = ♭

""" Sharp operator for converting primal 1-forms to primal vector fields.

This the primal-primal sharp from Hirani 2003, Definition 5.8.1 and Remark 2.7.2.

!!! note

    A PP-flat is also defined in (Desbrun et al 2005, Definition 7.4) but
    differs in two ways: Desbrun et al's notation suggests a *unit* normal
    vector, whereas the gradient of Hirani's primal-primal interpolation
    function is not necessarily a unit vector. More importantly, Hirani's vector
    is a normal to a different face than Desbrun et al's, with further confusion
    created by the fact that Hirani's Figure 5.7 agrees with Desbrun et al's
    description rather than his own. That being said, to the best of our
    knowledge, our implementation is the correct one and agrees with Hirani's
    description, if not his figure.

See also: [`♭`](@ref) and [`♯_mat`](@ref), which returns a matrix that encodes this operator.
"""
♯(s::HasDeltaSet2D, α::EForm) = PrimalVectorField(♯(s, α.data, PPSharp()))

""" Sharp operator for converting dual 1-forms to dual vector fields.

This dual-dual sharp uses a method of local linear least squares to provide a
tangent vector field.

See also: [`♯_mat`](@ref), which returns a matrix that encodes this operator.
"""
♯(s::HasDeltaSet2D, α::DualForm{1}) = DualVectorField(♯(s, α.data, LLSDDSharp()))

""" Alias for the sharp operator [`♯`](@ref).
"""
const sharp = ♯

"""    ♭♯_mat(s::HasDeltaSet2D)

Make a dual 1-form primal by chaining ♭ᵈᵖ♯ᵈᵈ.

This returns a matrix which can be multiplied by a dual 1-form.
See also [`♭♯`](@ref).
"""
♭♯_mat(s::HasDeltaSet2D) = only.(♭_mat(s, DPPFlat()) * ♯_mat(s, LLSDDSharp()))

"""    ♭♯(s::HasDeltaSet2D, α::SimplexForm{1})

Make a dual 1-form primal by chaining ♭ᵈᵖ♯ᵈᵈ.

This returns the given dual 1-form as a primal 1-form.
See also [`♭♯_mat`](@ref).
"""
♭♯(s::HasDeltaSet2D, α::SimplexForm{1}) = ♭♯_mat(s) * α

""" Alias for the flat-sharp dual-to-primal interpolation operator [`♭♯`](@ref).
"""
const flat_sharp = ♭♯

""" Alias for the flat-sharp dual-to-primal interpolation matrix [`♭♯_mat`](@ref).
"""
const flat_sharp_mat = ♭♯_mat


"""     p2_d2_interpolation(sd::HasDeltaSet2D)

Generates a sparse matrix that converts data on primal 2-forms into data on dual 2-forms.
"""
function p2_d2_interpolation(sd::HasDeltaSet2D)
  mat = spzeros(nv(sd), ntriangles(sd))
  for tri_id in triangles(sd)
    tri_area = sd[tri_id, :area]
    for dual_tri_id in tri_id:ntriangles(sd):nparts(sd, :DualTri)
      dual_tri_area = sd[dual_tri_id, :dual_area]

      weight = (dual_tri_area / tri_area)

      v = sd[sd[dual_tri_id, :D_∂e1], :D_∂v1]

      mat[v, tri_id] += weight
    end
  end

  mat
end

"""     p3_d3_interpolation(sd::HasDeltaSet3D)

Generates a sparse matrix that converts data on primal 3-forms into data on dual 3-forms.
"""
function p3_d3_interpolation(sd::HasDeltaSet3D)
  mat = spzeros(nv(sd), ntetrahedra(sd))
  for tet_id in tetrahedra(sd)
    tet_vol = sd[tet_id, :vol]
    for dual_tet_id in (1:24) .+ 24 * (tet_id - 1)
      dual_tet_vol = sd[dual_tet_id, :dual_vol]

      weight = (dual_tet_vol / tet_vol)

      v = sd[sd[sd[dual_tet_id, :D_∂t1], :D_∂e2], :D_∂v1]

      mat[v, tet_id] += weight
    end
  end

  mat
end


""" Wedge product of discrete forms.

The wedge product of a ``k``-form and an ``l``-form is a ``(k+l)``-form.

The DEC and related systems have several flavors of wedge product. This one is
the discrete primal-primal wedge product introduced in (Hirani, 2003, Chapter 7)
and (Desbrun et al 2005, Section 8). It depends on the geometric embedding and
requires the dual complex. Note that we diverge from Hirani in that his
formulation explicitly divides by (k+1)!. We do not do so in this computation.
"""
∧(s::HasDeltaSet, α::SimplexForm{k}, β::SimplexForm{l}) where {k,l} =
  SimplexForm{k+l}(∧(Val(k), Val(l), s, α.data, β.data))
@inline ∧(k::Int, l::Int, s::HasDeltaSet, args...) =
  ∧(Val(k), Val(l), s, args...)

function ∧(::Val{k}, ::Val{l}, s::HasDeltaSet, α, β) where {k,l}
  map(simplices(k+l, s)) do x
    ∧(Val(k), Val(l), s, α, β, x)
  end
end

∧(::Val{0}, ::Val{0}, s::HasDeltaSet, f, g, x::Int) =
  f[x]*g[x]
∧(::Val{k}, ::Val{0}, s::HasDeltaSet, α, g, x::Int) where k =
  wedge_product_zero(Val(k), s, g, α, x)
∧(::Val{0}, ::Val{k}, s::HasDeltaSet, f, β, x::Int) where k =
  wedge_product_zero(Val(k), s, f, β, x)

""" Wedge product of a 0-form and a ``k``-form.
"""
function wedge_product_zero(::Val{k}, s::HasDeltaSet,
                            f, α, x::Int) where k
  subs = subsimplices(k, s, x)
  vs = primal_vertex(k, s, subs)
  coeffs = map(x′ -> dual_volume(k,s,x′), subs) / volume(k,s,x)
  dot(coeffs, f[vs]) * α[x]
end

""" Alias for the wedge product operator [`∧`](@ref).
"""
const wedge_product = ∧

""" Interior product of a vector field (or 1-form) and a ``n``-form.

Specifically, this operation is the primal-dual interior product defined in
(Hirani 2003, Section 8.2) and (Desbrun et al 2005, Section 10). Thus it takes a
primal vector field (or primal 1-form) and a dual ``n``-forms and then returns a
dual ``(n-1)``-form.
"""
interior_product(s::HasDeltaSet, X♭::EForm, α::DualForm{n}; kw...) where n =
  DualForm{n-1}(interior_product_flat(Val(n), s, X♭.data, α.data); kw...)

""" Interior product of a 1-form and a ``n``-form, yielding an ``(n-1)``-form.

Usually, the interior product is defined for vector fields; this function
assumes that the flat operator [`♭`](@ref) (not yet implemented for primal
vector fields) has already been applied to yield a 1-form.
"""
@inline interior_product_flat(n::Int, s::HasDeltaSet, args...; kw...) =
  interior_product_flat(Val(n), s, args...; kw...)

function interior_product_flat(::Val{n}, s::HasDeltaSet,
                               X♭::AbstractVector, α::AbstractVector;
                               kw...) where n
  # TODO: Global sign `iseven(n*n′) ? +1 : -1`
  n′ = ndims(s) - n
  hodge_star(n′+1,s, wedge_product(n′,1,s, inv_hodge_star(n′,s, α; kw...), X♭); kw...)
end

""" Lie derivative of ``n``-form with respect to a vector field (or 1-form).

Specifically, this is the primal-dual Lie derivative defined in (Hirani 2003,
Section 8.4) and (Desbrun et al 2005, Section 10).
"""
ℒ(s::HasDeltaSet, X♭::EForm, α::DualForm{n}; kw...) where n =
  DualForm{n}(lie_derivative_flat(Val(n), s, X♭, α.data; kw...))

""" Alias for Lie derivative operator [`ℒ`](@ref).
"""
const lie_derivative = ℒ

""" Lie derivative of ``n``-form with respect to a 1-form.

Assumes that the flat operator [`♭`](@ref) has already been applied to the
vector field.
"""
@inline lie_derivative_flat(n::Int, s::HasDeltaSet, args...; kw...) =
  lie_derivative_flat(Val(n), s, args...; kw...)

function lie_derivative_flat(::Val{0}, s::HasDeltaSet,
                             X♭::AbstractVector, α::AbstractVector; kw...)
  interior_product_flat(1, s, X♭, dual_derivative(0, s, α); kw...)
end

function lie_derivative_flat(::Val{1}, s::HasDeltaSet,
                             X♭::AbstractVector, α::AbstractVector; kw...)
  interior_product_flat(2, s, X♭, dual_derivative(1, s, α); kw...) +
    dual_derivative(0, s, interior_product_flat(1, s, X♭, α; kw...))
end

function lie_derivative_flat(::Val{2}, s::HasDeltaSet,
                             X♭::AbstractVector, α::AbstractVector; kw...)
  dual_derivative(1, s, interior_product_flat(2, s, X♭, α; kw...))
end

function eval_constant_primal_form(s::HasDeltaSet1D, α)
  @assert length(α) == length(point(s, 1))
  EForm(map(edges(s)) do e
          dot(α, point(s, tgt(s,e)) - point(s, src(s,e))) * sign(1,s,e)
        end)
end

function eval_constant_primal_form(s::EmbeddedDeltaDualComplex2D{Bool, Float64, T} where T<:Union{Point2d, Point2D}, α::Union{Point3d, SVector{3,Float64}})
  α = SVector{2,Float64}(α[1],α[2])
  EForm(map(edges(s)) do e
          dot(α, point(s, tgt(s,e)) - point(s, src(s,e))) * sign(1,s,e)
        end)
end

# Evaluate a constant dual form
# XXX: This "left/right-hand-rule" trick only works when z=0.
# XXX: So, do not use this function to test e.g. curved surfaces.
function eval_constant_dual_form(s::EmbeddedDeltaDualComplex2D, α::Union{Point3d, SVector{3,Float64}})
  DualForm{1}(
    hodge_star(1,s) *
      eval_constant_primal_form(s, SVector{3,Float64}(α[2], -α[1], α[3])))
end

end
