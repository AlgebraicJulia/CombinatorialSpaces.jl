module Multigrid

using CombinatorialSpaces
using LinearAlgebra: I, Diagonal, Transpose
using Krylov, Catlab, SparseArrays
using ..SimplicialSets
import Catlab: dom, codom

export multigrid_vcycles, multigrid_wcycles, full_multigrid,
  repeated_subdivisions, repeated_subdivision_maps,
  binary_subdivision, binary_subdivision_map, dom, codom,
  as_matrix, MultigridData, MGData, AbstractGeometricMapSeries,
  PrimalGeometricMapSeries, finest_mesh, meshes, matrices, cubic_subdivision,
  cubic_subdivision_map, AbstractSubdivisionScheme, BinarySubdivision,
  CubicSubdivision, MeshTopology, subdivision, subdivision_map, subdivision_matrix,
  refine, propagate_points,
  AbstractMultigridMode, DirectMode, GalerkinMode,
  binary_subdivision_topo, cubic_subdivision_topo

# MeshTopology
#-------------

"""    struct MeshTopology

Boundary maps of a simplicial 2-complex as plain arrays.
"""
struct MeshTopology
  nv::Int
  ne::Int
  ntri::Int
  ∂v0::Vector{Int}
  ∂v1::Vector{Int}
  ∂e0::Vector{Int}
  ∂e1::Vector{Int}
  ∂e2::Vector{Int}
end

MeshTopology(s::EmbeddedDeltaSet1D) =
  MeshTopology(nv(s), ne(s), 0, s[:∂v0], s[:∂v1], Int[], Int[], Int[])

MeshTopology(s::EmbeddedDeltaSet2D) =
  MeshTopology(nv(s), ne(s), ntriangles(s), s[:∂v0], s[:∂v1], s[:∂e0], s[:∂e1], s[:∂e2])

"""
    topo_to_mesh(::Type{S}, topo::MeshTopology, points) -> S

Reconstitute an `EmbeddedDeltaSet` from a `MeshTopology` and point data.
"""
function topo_to_mesh(::Type{S}, topo::MeshTopology,
                      points) where S <: Union{EmbeddedDeltaSet1D, EmbeddedDeltaSet2D}
  sd = S()
  add_vertices!(sd, topo.nv)
  sd[1:topo.nv, :point] = points
  add_parts!(sd, :E, topo.ne)
  sd[1:topo.ne, :∂v0] = topo.∂v0
  sd[1:topo.ne, :∂v1] = topo.∂v1
  if S <: EmbeddedDeltaSet2D && topo.ntri > 0
    add_parts!(sd, :Tri, topo.ntri)
    sd[1:topo.ntri, :∂e0] = topo.∂e0
    sd[1:topo.ntri, :∂e1] = topo.∂e1
    sd[1:topo.ntri, :∂e2] = topo.∂e2
  end
  sd
end

# Types
#------

struct PrimalGeometricMap{D,M}
  domain::D
  codomain::D
  matrix::M
end

dom(f::PrimalGeometricMap) = f.domain
codom(f::PrimalGeometricMap) = f.codomain
as_matrix(f::PrimalGeometricMap) = f.matrix

abstract type AbstractSubdivisionScheme end
struct UnarySubdivision <: AbstractSubdivisionScheme end
struct BinarySubdivision <: AbstractSubdivisionScheme end
struct CubicSubdivision <: AbstractSubdivisionScheme end

# Binary Subdivision
#-------------------

"""
    propagate_points(::BinarySubdivision, topo, coarse_points) -> fine_points

Interpolate vertex positions for binary subdivision.
Original vertices are copied; midpoints are averaged from edge endpoints.
"""
function propagate_points(::BinarySubdivision, topo::MeshTopology,
                          coarse_points::AbstractVector)
  nv_c = topo.nv
  nv_f = nv_c + topo.ne
  fine_points = similar(coarse_points, nv_f)
  copyto!(fine_points, 1, coarse_points, 1, nv_c)
  @inbounds for e in 1:topo.ne
    fine_points[nv_c + e] =
      (coarse_points[topo.∂v0[e]] + coarse_points[topo.∂v1[e]]) / 2
  end
  fine_points
end

"""
    refine(::BinarySubdivision, topo) -> MeshTopology

Binary (medial) topology refinement: split each edge at its midpoint,
subdivide each triangle into 4.  No subdivision matrix is constructed.
"""
function refine(::BinarySubdivision, topo::MeshTopology)
  nv_c, ne_c, ntri_c = topo.nv, topo.ne, topo.ntri
  nv_f   = nv_c + ne_c
  ne_f   = 2 * ne_c + 3 * ntri_c
  ntri_f = 4 * ntri_c

  # --- Edge boundary maps ---
  ∂v0_f = Vector{Int}(undef, ne_f)
  ∂v1_f = Vector{Int}(undef, ne_f)

  @inbounds for e in 1:ne_c
    m = nv_c + e
    ∂v0_f[2 * e - 1] = m;          ∂v1_f[2 * e - 1] = topo.∂v0[e]
    ∂v0_f[2 * e]     = m;          ∂v1_f[2 * e]     = topo.∂v1[e]
  end

  # --- Triangle boundary maps ---
  #       v2
  #      /  \
  #    m1 -- m0
  #   /  \  /  \
  # v0 -- m2 -- v1
  ∂e0_f = Vector{Int}(undef, ntri_f)
  ∂e1_f = Vector{Int}(undef, ntri_f)
  ∂e2_f = Vector{Int}(undef, ntri_f)

  @inbounds for t in 1:ntri_c
    e0, e1, e2 = topo.∂e0[t], topo.∂e1[t], topo.∂e2[t]
    m0, m1, m2 = e0 + nv_c, e1 + nv_c, e2 + nv_c

    # Interior edges connecting the three midpoints.
    m0_m1 = 2 * ne_c + 3 * t - 2
    m1_m2 = 2 * ne_c + 3 * t - 1
    m0_m2 = 2 * ne_c + 3 * t

    ∂v0_f[m0_m1] = m1;  ∂v1_f[m0_m1] = m0
    ∂v0_f[m1_m2] = m2;  ∂v1_f[m1_m2] = m1
    ∂v0_f[m0_m2] = m2;  ∂v1_f[m0_m2] = m0

    # Split-edge references.
    m0_v1  = 2 * e0;      v2_m0 = 2 * e0 - 1
    m1_v0  = 2 * e1;      v2_m1 = 2 * e1 - 1
    m2_v0  = 2 * e2;      v1_m2 = 2 * e2 - 1

    # 4 child triangles.
    b = 4 * t - 3
    ∂e0_f[b]     = m1_m2;  ∂e1_f[b]     = m0_m2;  ∂e2_f[b]     = m0_m1
    ∂e0_f[b + 1] = m1_m2;  ∂e1_f[b + 1] = m2_v0;  ∂e2_f[b + 1] = m1_v0
    ∂e0_f[b + 2] = m0_m2;  ∂e1_f[b + 2] = v1_m2;  ∂e2_f[b + 2] = m0_v1
    ∂e0_f[b + 3] = m0_m1;  ∂e1_f[b + 3] = v2_m1;  ∂e2_f[b + 3] = v2_m0
  end

  MeshTopology(nv_f, ne_f, ntri_f, ∂v0_f, ∂v1_f, ∂e0_f, ∂e1_f, ∂e2_f)
end

"""
    subdivision_matrix(::BinarySubdivision, topo) -> SparseMatrixCSC

Build the binary subdivision matrix (nv_coarse × nv_fine) in direct CSC
format.  Identity block for original vertices, ½/½ averages for midpoints.
"""
function subdivision_matrix(::BinarySubdivision, topo::MeshTopology)
  nv_c, ne_c = topo.nv, topo.ne
  nv_f = nv_c + ne_c

  nnz = nv_c + 2 * ne_c
  colptr = Vector{Int32}(undef, nv_f + 1)
  rowval = Vector{Int32}(undef, nnz)
  nzval  = Vector{Float64}(undef, nnz)

  @inbounds for i in 1:nv_c
    colptr[i] = Int32(i)
    rowval[i] = Int32(i)
    nzval[i]  = 1.0
  end
  @inbounds for e in 1:ne_c
    k = nv_c + 2 * e - 1
    colptr[nv_c + e] = Int32(k)
    r0 = Int32(topo.∂v0[e])
    r1 = Int32(topo.∂v1[e])
    if r0 > r1
      r0, r1 = r1, r0
    end
    rowval[k]     = r0;  nzval[k]     = 0.5
    rowval[k + 1] = r1;  nzval[k + 1] = 0.5
  end
  colptr[nv_f + 1] = Int32(nnz + 1)

  SparseMatrixCSC(nv_c, nv_f, colptr, rowval, nzval)
end

"""
    binary_subdivision_topo(topo::MeshTopology) -> (refined::MeshTopology, mat)

Binary subdivision returning both topology and matrix.
"""
binary_subdivision_topo(topo::MeshTopology) =
  (refine(BinarySubdivision(), topo), subdivision_matrix(BinarySubdivision(), topo))

# Cubic Subdivision
#------------------

"""
    propagate_points(::CubicSubdivision, topo, coarse_points) -> fine_points

Interpolate vertex positions for cubic subdivision.
Original vertices are copied; edge points at ⅓/⅔ positions and
triangle centroids are computed from connectivity.
"""
function propagate_points(::CubicSubdivision, topo::MeshTopology,
                          coarse_points::AbstractVector)
  nv_c = topo.nv
  ne_c = topo.ne
  nv_f = nv_c + 2 * ne_c + topo.ntri
  fine_points = similar(coarse_points, nv_f)
  copyto!(fine_points, 1, coarse_points, 1, nv_c)
  @inbounds for e in 1:ne_c
    p0 = coarse_points[topo.∂v0[e]]
    p1 = coarse_points[topo.∂v1[e]]
    fine_points[nv_c + 2 * e - 1] = (2 * p0 + p1) / 3
    fine_points[nv_c + 2 * e]     = (p0 + 2 * p1) / 3
  end
  @inbounds for t in 1:topo.ntri
    pA = coarse_points[topo.∂v1[topo.∂e1[t]]]
    pB = coarse_points[topo.∂v1[topo.∂e0[t]]]
    pC = coarse_points[topo.∂v0[topo.∂e0[t]]]
    fine_points[nv_c + 2 * ne_c + t] = (pA + pB + pC) / 3
  end
  fine_points
end

"""
    refine(::CubicSubdivision, topo) -> MeshTopology

Cubic topology refinement: two interior points per edge plus a centroid per
triangle, subdividing each triangle into 9.  No subdivision matrix is constructed.
"""
function refine(::CubicSubdivision, topo::MeshTopology)
  nv_c, ne_c, ntri_c = topo.nv, topo.ne, topo.ntri
  nv_f   = nv_c + 2 * ne_c + ntri_c
  ne_f   = 3 * ne_c + 9 * ntri_c
  ntri_f = 9 * ntri_c

  # --- Edge boundary maps ---
  #  v0 -> m0 -> m1 -> v1  (3 child edges per original)
  ∂v0_f = Vector{Int}(undef, ne_f)
  ∂v1_f = Vector{Int}(undef, ne_f)

  @inbounds for e in 1:ne_c
    m0  = nv_c + 2 * e - 1
    m1  = nv_c + 2 * e
    v0e = topo.∂v0[e]
    v1e = topo.∂v1[e]
    ∂v0_f[3 * e - 2] = m0;   ∂v1_f[3 * e - 2] = v0e
    ∂v0_f[3 * e - 1] = m1;   ∂v1_f[3 * e - 1] = m0
    ∂v0_f[3 * e]     = v1e;  ∂v1_f[3 * e]     = m1
  end

  # --- Triangle boundary maps ---
  #          030
  #         ^  v
  #       021 > 120
  #      ^  ^  ^  v
  #    012< 111  >210
  #   ^  ^  ^  ^  ^  v
  # 003 >102  >201  >300
  ∂e0_f = Vector{Int}(undef, ntri_f)
  ∂e1_f = Vector{Int}(undef, ntri_f)
  ∂e2_f = Vector{Int}(undef, ntri_f)

  @inbounds for t in 1:ntri_c
    e0, e1, e2 = topo.∂e0[t], topo.∂e1[t], topo.∂e2[t]

    # Interior vertex indices (barycentric coordinate naming).
    m012 = 2 * e0 + nv_c - 1;  m021 = 2 * e0 + nv_c
    m102 = 2 * e1 + nv_c - 1;  m201 = 2 * e1 + nv_c
    m120 = 2 * e2 + nv_c - 1;  m210 = 2 * e2 + nv_c
    m111 = nv_c + 2 * ne_c + t

    # 9 interior edges.
    be = 3 * ne_c + 9 * t - 8
    ∂v0_f[be]     = m210;  ∂v1_f[be]     = m201
    ∂v0_f[be + 1] = m111;  ∂v1_f[be + 1] = m201
    ∂v0_f[be + 2] = m111;  ∂v1_f[be + 2] = m102
    ∂v0_f[be + 3] = m012;  ∂v1_f[be + 3] = m102
    ∂v0_f[be + 4] = m012;  ∂v1_f[be + 4] = m111
    ∂v0_f[be + 5] = m021;  ∂v1_f[be + 5] = m111
    ∂v0_f[be + 6] = m210;  ∂v1_f[be + 6] = m111
    ∂v0_f[be + 7] = m120;  ∂v1_f[be + 7] = m111
    ∂v0_f[be + 8] = m120;  ∂v1_f[be + 8] = m021

    # Named edge references for triangle wiring.
    m201_m210 = be;      m201_m111 = be + 1;  m102_m111 = be + 2
    m102_m012 = be + 3;  m111_m012 = be + 4;  m111_m021 = be + 5
    m111_m210 = be + 6;  m111_m120 = be + 7;  m021_m120 = be + 8

    m021_v030 = 3 * e0;      m201_v300 = 3 * e1;      m210_v300 = 3 * e2
    m012_m021 = 3 * e0 - 1;  m102_m201 = 3 * e1 - 1;  m120_m210 = 3 * e2 - 1
    v003_m012 = 3 * e0 - 2;  v003_m102 = 3 * e1 - 2;  v030_m120 = 3 * e2 - 2

    # 9 child triangles.
    bt = 9 * t - 8
    ∂e0_f[bt]     = m210_v300;  ∂e1_f[bt]     = m201_v300;  ∂e2_f[bt]     = m201_m210
    ∂e0_f[bt + 1] = m111_m210;  ∂e1_f[bt + 1] = m201_m210;  ∂e2_f[bt + 1] = m201_m111
    ∂e0_f[bt + 2] = m201_m111;  ∂e1_f[bt + 2] = m102_m111;  ∂e2_f[bt + 2] = m102_m201
    ∂e0_f[bt + 3] = m111_m012;  ∂e1_f[bt + 3] = m102_m012;  ∂e2_f[bt + 3] = m102_m111
    ∂e0_f[bt + 4] = m102_m012;  ∂e1_f[bt + 4] = v003_m012;  ∂e2_f[bt + 4] = v003_m102
    ∂e0_f[bt + 5] = m120_m210;  ∂e1_f[bt + 5] = m111_m210;  ∂e2_f[bt + 5] = m111_m120
    ∂e0_f[bt + 6] = m021_m120;  ∂e1_f[bt + 6] = m111_m120;  ∂e2_f[bt + 6] = m111_m021
    ∂e0_f[bt + 7] = m012_m021;  ∂e1_f[bt + 7] = m111_m021;  ∂e2_f[bt + 7] = m111_m012
    ∂e0_f[bt + 8] = v030_m120;  ∂e1_f[bt + 8] = m021_m120;  ∂e2_f[bt + 8] = m021_v030
  end

  MeshTopology(nv_f, ne_f, ntri_f, ∂v0_f, ∂v1_f, ∂e0_f, ∂e1_f, ∂e2_f)
end

"""
    subdivision_matrix(::CubicSubdivision, topo) -> SparseMatrixCSC

Build the cubic subdivision matrix (nv_coarse × nv_fine) in direct CSC
format.  Identity block for original vertices, ⅓/⅔ weights for edge
points, equal weights for centroids.
"""
function subdivision_matrix(::CubicSubdivision, topo::MeshTopology)
  nv_c, ne_c, ntri_c = topo.nv, topo.ne, topo.ntri
  nv_f = nv_c + 2 * ne_c + ntri_c

  nnz = nv_c + 4 * ne_c + 3 * ntri_c
  colptr = Vector{Int32}(undef, nv_f + 1)
  rowval = Vector{Int32}(undef, nnz)
  nzval  = Vector{Float64}(undef, nnz)

  @inbounds for i in 1:nv_c
    colptr[i] = Int32(i)
    rowval[i] = Int32(i)
    nzval[i]  = 1.0
  end

  # Two interior points per edge.
  @inbounds for e in 1:ne_c
    r0 = Int32(topo.∂v0[e])
    r1 = Int32(topo.∂v1[e])
    lo, hi = r0 < r1 ? (r0, r1) : (r1, r0)
    w_lo_third = r0 < r1 ? 2.0 / 3 : 1.0 / 3
    w_hi_third = r0 < r1 ? 1.0 / 3 : 2.0 / 3

    # ⅓ point column.
    k1 = nv_c + 4 * e - 3
    colptr[nv_c + 2 * e - 1] = Int32(k1)
    rowval[k1]     = lo;  nzval[k1]     = w_lo_third
    rowval[k1 + 1] = hi;  nzval[k1 + 1] = w_hi_third

    # ⅔ point column.
    k2 = nv_c + 4 * e - 1
    colptr[nv_c + 2 * e] = Int32(k2)
    rowval[k2]     = lo;  nzval[k2]     = w_hi_third
    rowval[k2 + 1] = hi;  nzval[k2 + 1] = w_lo_third
  end

  # Centroid columns.
  @inbounds for t in 1:ntri_c
    k = nv_c + 4 * ne_c + 3 * t - 2
    colptr[nv_c + 2 * ne_c + t] = Int32(k)
    vA = Int32(topo.∂v1[topo.∂e1[t]])
    vB = Int32(topo.∂v1[topo.∂e0[t]])
    vC = Int32(topo.∂v0[topo.∂e0[t]])
    vA > vB && ((vA, vB) = (vB, vA))
    vB > vC && ((vB, vC) = (vC, vB))
    vA > vB && ((vA, vB) = (vB, vA))
    rowval[k]     = vA;  nzval[k]     = 1.0 / 3
    rowval[k + 1] = vB;  nzval[k + 1] = 1.0 / 3
    rowval[k + 2] = vC;  nzval[k + 2] = 1.0 / 3
  end

  colptr[nv_f + 1] = Int32(nnz + 1)
  SparseMatrixCSC(nv_c, nv_f, colptr, rowval, nzval)
end

"""
    cubic_subdivision_topo(topo::MeshTopology) -> (refined::MeshTopology, mat)

Cubic subdivision returning both topology and matrix.
"""
cubic_subdivision_topo(topo::MeshTopology) =
  (refine(CubicSubdivision(), topo), subdivision_matrix(CubicSubdivision(), topo))

"""
    propagate_points(mat::SparseMatrixCSC, coarse_points) -> fine_points

Interpolate via column-wise SpMV on the subdivision matrix.  Exported for
external callers that have a matrix but no scheme type; all internal paths
use the scheme-dispatched methods above.
"""
function propagate_points(mat::SparseMatrixCSC, coarse_points::AbstractVector)
  nv_fine = size(mat, 2)
  fine_points = similar(coarse_points, nv_fine)
  rv = rowvals(mat)
  nz = nonzeros(mat)
  @inbounds for j in 1:nv_fine
    pt = zero(eltype(coarse_points))
    for idx in nzrange(mat, j)
      pt += nz[idx] * coarse_points[rv[idx]]
    end
    fine_points[j] = pt
  end
  fine_points
end

# ACSet Subdivision Interface
#----------------------------

"""
    subdivision(s, scheme) -> EmbeddedDeltaSet

Subdivide a mesh.  Only refines topology and propagates points — no
subdivision matrix is constructed.
"""
function subdivision(s::Union{EmbeddedDeltaSet1D, EmbeddedDeltaSet2D},
                     scheme::AbstractSubdivisionScheme)
  topo = MeshTopology(s)
  points = propagate_points(scheme, topo, s[:point])
  topo_to_mesh(typeof(s), refine(scheme, topo), points)
end

subdivision(s::EmbeddedDeltaSet2D, ::UnarySubdivision) = copy(s)
subdivision(s::EmbeddedDeltaSet2D) = subdivision(s, BinarySubdivision())

"""
    subdivision_map(s, scheme) -> PrimalGeometricMap

Subdivide and return the geometric map (mesh + subdivision matrix).
"""
function subdivision_map(s::Union{EmbeddedDeltaSet1D, EmbeddedDeltaSet2D},
                         scheme::AbstractSubdivisionScheme)
  topo = MeshTopology(s)
  mat = subdivision_matrix(scheme, topo)
  points = propagate_points(scheme, topo, s[:point])
  sd = topo_to_mesh(typeof(s), refine(scheme, topo), points)
  PrimalGeometricMap(sd, s, mat)
end

function subdivision_map(s::Union{EmbeddedDeltaSet1D, EmbeddedDeltaSet2D}, ::UnarySubdivision)
  PrimalGeometricMap(copy(s), s, I(nv(s)))
end

subdivision_map(pgm::PrimalGeometricMap, scheme::AbstractSubdivisionScheme) =
  subdivision_map(dom(pgm), scheme)

# Backward-compatible named aliases.
binary_subdivision(s) = subdivision(s, BinarySubdivision())
cubic_subdivision(s)  = subdivision(s, CubicSubdivision())
unary_subdivision(s)  = subdivision(s, UnarySubdivision())

binary_subdivision_map(s) = subdivision_map(s, BinarySubdivision())
cubic_subdivision_map(s)  = subdivision_map(s, CubicSubdivision())
unary_subdivision_map(s)  = subdivision_map(s, UnarySubdivision())

# Repeated Subdivisions and Map Series
#-------------------------------------

"""
    repeated_subdivisions(k, ss, scheme) -> Vector{EmbeddedDeltaSet}

Apply `k` subdivisions.  Returns meshes only — no subdivision matrices
are constructed.
"""
repeated_subdivisions(k, ss, scheme::AbstractSubdivisionScheme) =
  accumulate((x,_) -> subdivision(x, scheme), 1:k; init=ss)
repeated_subdivisions(k, ss) = repeated_subdivisions(k, ss, BinarySubdivision())

"""
    repeated_subdivision_maps(k, ss, subdivider) -> Vector{PrimalGeometricMap}

Apply `k` subdivisions via `subdivider`, returning geometric maps (meshes +
matrices).  Used by `PrimalGeometricMapSeries`.
"""
repeated_subdivision_maps(k, ss, scheme::AbstractSubdivisionScheme) =
  repeated_subdivision_maps(k, ss, x -> subdivision_map(x, scheme))
repeated_subdivision_maps(k, ss, subdivider::Function) =
  accumulate((x,_) -> subdivider(x), 1:k; init=ss)

abstract type AbstractGeometricMapSeries end

struct PrimalGeometricMapSeries{D<:HasDeltaSet, M<:AbstractMatrix} <: AbstractGeometricMapSeries
  meshes::AbstractVector{D}
  matrices::AbstractVector{M}
end

meshes(series::PrimalGeometricMapSeries) = series.meshes
matrices(series::PrimalGeometricMapSeries) = series.matrices
finest_mesh(series::PrimalGeometricMapSeries) = first(series.meshes)

PrimalGeometricMapSeries(s::HasDeltaSet, scheme::AbstractSubdivisionScheme, levels::Int, alg=Circumcenter()) =
  PrimalGeometricMapSeries(s, x -> subdivision_map(x, scheme), levels, alg)
PrimalGeometricMapSeries(s::HasDeltaSet, levels::Int, alg=Circumcenter()) =
  PrimalGeometricMapSeries(s, BinarySubdivision(), levels, alg)

function PrimalGeometricMapSeries(s::HasDeltaSet, subdivider::Function, levels::Int, alg = Circumcenter())
  subdivs = Iterators.reverse(repeated_subdivision_maps(levels, s, subdivider));
  meshes = [dom.(subdivs)..., s]
  dual_meshes = map(s -> dualize(s, alg), meshes)
  matrices = as_matrix.(subdivs)
  PrimalGeometricMapSeries{typeof(first(dual_meshes)), typeof(first(matrices))}(dual_meshes, matrices)
end

# MultigridData
#--------------

struct MultigridData{Mv}
  operators::Vector{SparseMatrixCSC{Float64, Int32}}
  restrictions::Mv
  prolongations::Mv
  steps::Vector{Int}
end

MultigridData(g,r,p,s) = MultigridData{typeof(r)}(g,r,p,s)
MultigridData(g,r,p,s::Int) = MultigridData(g,r,p,fill(s,length(g)))

MGData(series::PrimalGeometricMapSeries, op::Function, s::Int, ::T) where T <: AbstractSubdivisionScheme =
  MultigridData(series, op, fill(s,length(series.meshes)))
MGData(series::PrimalGeometricMapSeries, op::Function, s::Int) =
  MultigridData(series, op, fill(s,length(series.meshes)))

Base.length(md::MultigridData) = length(md.operators)

# AbstractMultigridMode
#----------------------

"""
    abstract type AbstractMultigridMode

Controls operator assembly in `MultigridData` construction.

- `DirectMode()`: dualize and discretize at every level during the walk.
- `GalerkinMode()`: only dualize the finest mesh; derive coarse operators
  via the Galerkin condition  Aₖ₊₁ = Rₖ Aₖ Pₖ.
"""
abstract type AbstractMultigridMode end
struct DirectMode   <: AbstractMultigridMode end
struct GalerkinMode <: AbstractMultigridMode end

# Restriction Normalization
#--------------------------

function row_normalize!(M)
  nrows = size(M, 1)
  row_sums = zeros(nrows)
  rvs = rowvals(M)
  nzv = nonzeros(M)
  @inbounds for k in eachindex(nzv)
    row_sums[rvs[k]] += nzv[k]
  end
  @inbounds for k in eachindex(nzv)
    nzv[k] /= row_sums[rvs[k]]
  end
  M
end

"""
    make_restriction(p::Transpose{<:Any, <:SparseMatrixCSC})

Build restriction R = row_normalize(S) directly from the subdivision matrix
S = parent(p), sharing its `colptr` and `rowval` arrays.  Only allocates a
new `nzval` vector and a temporary row-sum buffer.
"""
function make_restriction(p::Transpose{<:Any, <:SparseMatrixCSC})
  S = parent(p)
  rv = rowvals(S)
  nz = nonzeros(S)
  nrows = size(S, 1)

  row_sums = zeros(nrows)
  @inbounds for k in eachindex(nz)
    row_sums[rv[k]] += nz[k]
  end

  new_nz = Vector{Float64}(undef, length(nz))
  @inbounds for k in eachindex(nz)
    new_nz[k] = nz[k] / row_sums[rv[k]]
  end

  SparseMatrixCSC(nrows, size(S, 2), S.colptr, rv, new_nz)
end

make_restriction(p::Diagonal) = (pt = transpose(p); pt ./ sum(pt, dims=2))
make_restriction(p::AbstractMatrix) = row_normalize!(copy(transpose(p)))

normalize_restrictions(ps::AbstractVector) = map(make_restriction, ps)

# MultigridData Constructors
#---------------------------

# --- From PrimalGeometricMapSeries (legacy) ---

function MultigridData(series::PrimalGeometricMapSeries, op::Function, s::AbstractVector)
  ops = op.(meshes(series))
  ps = transpose.(matrices(series))
  rs = normalize_restrictions(ps)
  MultigridData(ops, rs, ps, s)
end

# --- From custom subdivider function (backward compat, no mode dispatch) ---

function MultigridData(s::HasDeltaSet, subdivider::Function, levels::Int,
                       op::Function, steps, alg=Circumcenter())
  steps_vec = steps isa AbstractVector ? steps : fill(steps, levels + 1)

  if levels == 0
    only_op = op(dualize(s, alg))
    return MultigridData([only_op], typeof(only_op)[], typeof(only_op)[], steps_vec)
  end

  current_primal = s
  sd = dualize(current_primal, alg)
  first_op = op(sd)
  sd = nothing

  pgm = subdivider(current_primal)
  first_mat = as_matrix(pgm)
  current_primal = dom(pgm)
  pgm = nothing

  first_p = transpose(first_mat)
  first_r = make_restriction(first_p)

  ops = Vector{typeof(first_op)}(undef, levels + 1)
  ps  = Vector{typeof(first_p)}(undef, levels)
  rs  = Vector{typeof(first_r)}(undef, levels)
  ops[levels + 1] = first_op
  ps[levels] = first_p
  rs[levels] = first_r

  for i in 2:levels
    sd = dualize(current_primal, alg)
    ops[levels - i + 2] = op(sd)
    sd = nothing

    pgm = subdivider(current_primal)
    mat = as_matrix(pgm)
    current_primal = dom(pgm)
    pgm = nothing

    idx = levels - i + 1
    ps[idx] = transpose(mat)
    rs[idx] = make_restriction(ps[idx])
  end

  ops[1] = op(dualize(current_primal, alg))
  MultigridData(ops, rs, ps, steps_vec)
end

# --- Scheme-typed entry point (dispatches on AbstractMultigridMode) ---

"""
    MultigridData(s, scheme, levels, op, steps; alg, mode)

Construct a `MultigridData` from a base mesh using topology-only subdivision.

# Keyword arguments
- `mode::AbstractMultigridMode = DirectMode()`: controls operator assembly.
- `alg = Circumcenter()`: dualization algorithm.

# Examples
```julia
md = MultigridData(s, BinarySubdivision(), 4, s -> ∇²(0, s), 3)
md = MultigridData(s, BinarySubdivision(), 4, s -> ∇²(0, s), 3; mode=GalerkinMode())
```
"""
function MultigridData(s::HasDeltaSet2D, scheme::AbstractSubdivisionScheme,
                       levels::Int, op::Function, steps; alg=Circumcenter(),
                       mode::AbstractMultigridMode=DirectMode())
  _build_multigrid(mode, s, scheme, levels, op, steps, alg)
end

# Fallback for HasDeltaSet (1D) — delegates to function-based path.
function MultigridData(s::HasDeltaSet, scheme::AbstractSubdivisionScheme,
                       levels::Int, op::Function, steps; alg=Circumcenter(),
                       mode::AbstractMultigridMode=DirectMode())
  MultigridData(s, x -> subdivision_map(x, scheme), levels, op, steps, alg)
end

# _build_multigrid
#-----------------

# The coarse→fine walk is shared:
#   1. Extract MeshTopology from the base mesh.
#   2. At each level: propagate points, subdivide topology, build p/r pair.
#
# Three phases dispatch on the mode to control operator assembly:
#   _init_ops          — before the walk
#   _record_level_op!  — after each subdivision level
#   _finalize_ops      — after the walk completes

# --- Phase 1: Pre-walk initialization ---

"""Compute the coarsest-level operator before the walk begins."""
_init_ops(::DirectMode, s, op, alg) = [op(dualize(s, alg))]

"""No operators needed before the walk in Galerkin mode."""
_init_ops(::GalerkinMode, s, op, alg) = Any[]

# --- Phase 2: Per-level operator recording ---

"""Dualize the current mesh and record its operator."""
_record_level_op!(::DirectMode, ops, S, topo, points, op, alg) =
  push!(ops, op(dualize(topo_to_mesh(S, topo, points), alg)))

"""No per-level operators in Galerkin mode."""
_record_level_op!(::GalerkinMode, ops, S, topo, points, op, alg) = ops

# --- Phase 3: Post-walk finalization ---

"""Direct mode: operators were collected coarsest-first; reverse to finest-first."""
_finalize_ops(::DirectMode, ops, S, topo, points, ps, rs, levels, op, alg) =
  reverse(ops)

"""Galerkin mode: dualize the finest mesh, derive coarse operators via Rₖ Aₖ Pₖ."""
function _finalize_ops(::GalerkinMode, _, S, topo, points, ps, rs, levels, op, alg)
  finest_op = op(dualize(topo_to_mesh(S, topo, points), alg))
  ops = Vector{typeof(finest_op)}(undef, levels + 1)
  ops[1] = finest_op
  for i in 1:levels
    ops[i + 1] = rs[i] * ops[i] * ps[i]
  end
  ops
end

# --- Main walk ---

function _build_multigrid(mode::AbstractMultigridMode, s::HasDeltaSet2D,
                          scheme::AbstractSubdivisionScheme,
                          levels::Int, op::Function, steps, alg)
  steps_vec = steps isa AbstractVector ? steps : fill(steps, levels + 1)
  S = typeof(s)

  if levels == 0
    only_op = op(dualize(s, alg))
    return MultigridData([only_op], typeof(only_op)[], typeof(only_op)[], steps_vec)
  end

  topo = MeshTopology(s)
  points = s[:point]

  # Phase 1: Pre-walk operator initialization.
  ops = _init_ops(mode, s, op, alg)

  # Phase 2: Coarse-to-fine subdivision walk.
  # First step bootstraps concrete types for ps/rs.
  points = propagate_points(scheme, topo, points)
  mat = subdivision_matrix(scheme, topo)
  topo = refine(scheme, topo)
  first_p = transpose(mat)
  first_r = make_restriction(first_p)

  ps = Vector{typeof(first_p)}(undef, levels)
  rs = Vector{typeof(first_r)}(undef, levels)
  ps[levels] = first_p
  rs[levels] = first_r
  _record_level_op!(mode, ops, S, topo, points, op, alg)

  for i in 2:levels
    points = propagate_points(scheme, topo, points)
    mat = subdivision_matrix(scheme, topo)
    topo = refine(scheme, topo)
    idx = levels + 1 - i
    ps[idx] = transpose(mat)
    rs[idx] = make_restriction(ps[idx])
    _record_level_op!(mode, ops, S, topo, points, op, alg)
  end

  # Phase 3: Finalize operator assembly.
  ops = _finalize_ops(mode, ops, S, topo, points, ps, rs, levels, op, alg)

  MultigridData(ops, rs, ps, steps_vec)
end

# Utility
#--------

function is_simplicial_complex(s::HasDeltaSet2D)
  allunique(map(x -> edge_vertices(s,x), edges(s))) &&
  allunique(map(x -> triangle_vertices(s,x), triangles(s)))
end

# Multigrid Algorithms
#---------------------

"""
Solve `Ax=b` with initial guess `u` for `cycles` V-cycles.
`alg` is a Krylov.jl method (default `cg`).
"""
multigrid_vcycles(u, b, md, cycles, alg=cg) = multigrid_μ_cycles(u, b, md, cycles, alg, 1)

"""W-cycle variant of `multigrid_vcycles`."""
multigrid_wcycles(u, b, md, cycles, alg=cg) = multigrid_μ_cycles(u, b, md, cycles, alg, 2)

function multigrid_μ_cycles(u, b, md::MultigridData, cycles, alg=cg, μ=1)
  for _ in 1:cycles
    u = _multigrid_μ_cycle(u, b, md, 1, alg, μ)
  end
  u
end

"""
Full multigrid: start at the coarsest grid and work up, applying
μ-cycles at each level (μ=1 for V, μ=2 for W).
"""
function full_multigrid(b, md::MultigridData, cycles, alg=cg, μ=1)
  z_f = zeros(size(b))
  if length(md) > 1
    r = md.restrictions[1]
    p = md.prolongations[1]
    b_c = r * b
    z_c = _full_multigrid(b_c, md, 2, cycles, alg, μ)
    z_f = p * z_c
  end
  multigrid_μ_cycles(z_f, b, md, cycles, alg, μ)
end

function _full_multigrid(b, md::MultigridData, lvl, cycles, alg, μ)
  z_f = zeros(size(b))
  if lvl < length(md)
    r = md.restrictions[lvl]
    p = md.prolongations[lvl]
    b_c = r * b
    z_c = _full_multigrid(b_c, md, lvl + 1, cycles, alg, μ)
    z_f = p * z_c
  end
  for _ in 1:cycles
    z_f = _multigrid_μ_cycle(z_f, b, md, lvl, alg, μ)
  end
  z_f
end

function _multigrid_μ_cycle(u, b, md::MultigridData, lvl, alg=cg, μ=1)
  A     = md.operators[lvl]
  steps = md.steps[lvl]
  u = steps == 0 ? u : alg(A, b, u, itmax=steps)[1]
  lvl >= length(md) && return u
  r = md.restrictions[lvl]
  p = md.prolongations[lvl]
  r_f = b - A * u
  r_c = r * r_f
  z = _multigrid_μ_cycle(zeros(size(r_c)), r_c, md, lvl + 1, alg, μ)
  if μ > 1
    z = _multigrid_μ_cycle(z, r_c, md, lvl + 1, alg, μ - 1)
  end
  u += p * z
  u = steps == 0 ? u : alg(A, b, u, itmax=steps)[1]
end

end

