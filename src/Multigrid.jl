module Multigrid

using CombinatorialSpaces
using LinearAlgebra: I, Diagonal
using Krylov, Catlab, SparseArrays, StaticArrays
using ..SimplicialSets
import Catlab: dom, codom

export multigrid_vcycles, multigrid_wcycles, full_multigrid,
  repeated_subdivisions, binary_subdivision, binary_subdivision_map, dom, codom,
  as_matrix, MultigridData, MGData, AbstractGeometricMapSeries,
  PrimalGeometricMapSeries, finest_mesh, meshes, matrices, cubic_subdivision,
  cubic_subdivision_map, AbstractSubdivisionScheme, BinarySubdivision,
  CubicSubdivision

# Types, Structs, Constructors, Getters & Setters, and Dispatch Control
#----------------------------------------------------------------------

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

subdivision(s::EmbeddedDeltaSet2D, ::UnarySubdivision) = unary_subdivision(s)
subdivision(s::EmbeddedDeltaSet2D, ::BinarySubdivision) = binary_subdivision(s)
subdivision(s::EmbeddedDeltaSet2D, ::CubicSubdivision) = cubic_subdivision(s)
subdivision(s::EmbeddedDeltaSet2D) = binary_subdivision(s, BinarySubdivision)

unary_subdivision_map(pgm::PrimalGeometricMap) = unary_subdivision_map(dom(pgm))
binary_subdivision_map(pgm::PrimalGeometricMap) = binary_subdivision_map(dom(pgm))
cubic_subdivision_map(pgm::PrimalGeometricMap) = cubic_subdivision_map(dom(pgm))

repeated_subdivisions(k, ss, ::UnarySubdivision) = repeated_subdivisions(k, ss, unary_subdivision_map)
repeated_subdivisions(k, ss, ::BinarySubdivision) = repeated_subdivisions(k, ss, binary_subdivision_map)
repeated_subdivisions(k, ss, ::CubicSubdivision) = repeated_subdivisions(k, ss, cubic_subdivision_map)
repeated_subdivisions(k, ss) = repeated_subdivisions(k, ss, BinarySubdivision)

# Different means of representing a series of complexes with maps between them should sub-type this abstract type.
# Those concrete types should then provide a constructor for `MultigridData`.
"""    abstract type AbstractGeometricMapSeries end

Organizes the mesh data that results from mesh refinement through a subdivision method.

See also: [`PrimalGeometricMapSeries`](@ref).
"""
abstract type AbstractGeometricMapSeries end

"""    struct PrimalGeometricMapSeries{D<:HasDeltaSet, M<:AbstractMatrix} <: AbstractGeometricMapSeries
Organize a series of dual complexes and maps between primal vertices between them.

See also: [`AbstractGeometricMapSeries`](@ref).
"""
struct PrimalGeometricMapSeries{D<:HasDeltaSet, M<:AbstractMatrix} <: AbstractGeometricMapSeries
  meshes::AbstractVector{D}
  matrices::AbstractVector{M}
end

meshes(series::PrimalGeometricMapSeries) = series.meshes
matrices(series::PrimalGeometricMapSeries) = series.matrices

PrimalGeometricMapSeries(s::HasDeltaSet, ::UnarySubdivision, levels::Int, alg = Circumcenter()) =
  PrimalGeometricMapSeries(s, unary_subdivision_map, levels, alg)
PrimalGeometricMapSeries(s::HasDeltaSet, ::BinarySubdivision, levels::Int, alg = Circumcenter()) =
  PrimalGeometricMapSeries(s, binary_subdivision_map, levels, alg)
PrimalGeometricMapSeries(s::HasDeltaSet, ::CubicSubdivision, levels::Int, alg = Circumcenter()) =
  PrimalGeometricMapSeries(s, cubic_subdivision_map, levels, alg)
PrimalGeometricMapSeries(s::HasDeltaSet, levels::Int, alg = Circumcenter()) =
  PrimalGeometricMapSeries(s, binary_subdivision_map, levels, alg)

"""    finest_mesh(series::PrimalGeometricMapSeries)

Return the mesh in a `PrimalGeometricMapSeries` with the highest resolution.
"""
finest_mesh(series::PrimalGeometricMapSeries) = first(series.meshes)

"""    struct MultigridData{Gv,Mv}

Contains the data required for multigrid methods. If there are
`n` grids, there are `n-1` restrictions and prolongations and `n`
step radii. This structure does not contain the solution `u` or
the right-hand side `b` because those would have to mutate.
"""
struct MultigridData{Gv,Mv}
  operators::Gv
  restrictions::Mv
  prolongations::Mv
  steps::Vector{Int}
end

MultigridData(g,r,p,s) = MultigridData{typeof(g),typeof(r)}(g,r,p,s)

# This function definition is kept for backwards compatibility.
MGData(series::PrimalGeometricMapSeries, op::Function, s::Int, ::T) where T <: AbstractSubdivisionScheme =
  MultigridData(series, op, fill(s,length(series.meshes)))

MGData(series::PrimalGeometricMapSeries, op::Function, s::Int) =
  MultigridData(series, op, fill(s,length(series.meshes)))

"""    MultigridData(g,r,p,s::Int)

Construct a `MultigridData` with a constant step radius on each grid.
"""
MultigridData(g,r,p,s::Int) = MultigridData(g,r,p,fill(s,length(g)))

"""    function car(md::MultigridData)

Get the leading grid, restriction, prolongation, and step radius.
"""
function car(md::MultigridData)
  first_or_null(x) = isempty(x) ? nothing : first(x)
  first_or_null.([md.operators, md.restrictions, md.prolongations, md.steps])
end

"""    cdr(md::MultigridData)

Remove the leading grid, restriction, prolongation, and step radius.
"""
cdr(md::MultigridData) =
  length(md) > 1 ?
    MultigridData(md.operators[2:end],md.restrictions[2:end],md.prolongations[2:end],md.steps[2:end]) :
    error("Not enough grids remaining in $md to take the cdr.")

"""    Base.length(md::MultigridData)

The length of a `MultigridData` is its number of grids.
"""
Base.length(md::MultigridData) = length(md.operators)

"""    function PrimalGeometricMapSeries(s::HasDeltaSet, subdivider::Function, levels::Int, alg = Circumcenter())

Construct a `PrimalGeometricMapSeries` given a primal mesh `s` and a subdivider function like `binary_subdivision`, `levels` times.

The `PrimalGeometricMapSeries` returned contains a list of `levels + 1` dual complexes, with `levels` matrices between the primal vertices of each.

See also: [`AbstractGeometricMapSeries`](@ref), [`finest_mesh`](@ref).
"""
function PrimalGeometricMapSeries(s::HasDeltaSet, subdivider::Function, levels::Int, alg = Circumcenter())
  subdivs = Iterators.reverse(repeated_subdivisions(levels, s, subdivider));
  meshes = [dom.(subdivs)..., s]
  dual_meshes = map(s -> dualize(s, alg), meshes)
  matrices = as_matrix.(subdivs)
  PrimalGeometricMapSeries{typeof(first(dual_meshes)), typeof(first(matrices))}(dual_meshes, matrices)
end

function normalize_restrictions(ps::Vector{T}) where T <: Diagonal
  rs = map(ps) do p
    pt = transpose(p)
    pt ./ sum(pt, dims=2)
  end
end

# XXX: Row-normalizing a sparse matrix is non-trivial.
#https://discourse.julialang.org/t/scaling-a-sparse-matrix-row-wise-and-column-wise-too-slow/115956/8
function row_normalize!(M)
  row_sums = sum(M, dims=2)
  rows = rowvals(M)
  vals = nonzeros(M)
  n = size(M, 2)
  for j in 1:n
    for i in nzrange(M, j)
      row = rows[i]
      vals[i] /= row_sums[row]
    end
  end
  M
end

function normalize_restrictions(ps::Vector{T}) where T <: AbstractMatrix
  rs = map(ps) do p
    pt = copy(transpose(p))
    row_normalize!(pt)
  end
end

function MultigridData(series::PrimalGeometricMapSeries, op::Function, s::AbstractVector)
  ops = op.(meshes(series))
  ps = transpose.(matrices(series))
  rs = normalize_restrictions(ps)
  MultigridData(ops, rs, ps, s)
end

# XXX: This function does not detect e.g. dangling edges.
function is_simplicial_complex(s::HasDeltaSet2D)
  allunique(map(x -> edge_vertices(s,x), edges(s))) &&
  allunique(map(x -> triangle_vertices(s,x), triangles(s)))
end

# Subdivision Algorithms
#-----------------------

# Subdivide each triangle into 1 via "unary" a.k.a. "trivial" subdivision,
# returning a primal simplicial complex.
function unary_subdivision(s::Union{EmbeddedDeltaSet1D, EmbeddedDeltaSet2D})
  sd = copy(s)
  sd
end

function unary_subdivision_map(s)
  sd = unary_subdivision(s)
  PrimalGeometricMap(sd, s, I(nv(s)))
end

"""
Subdivide each triangle into 4 via "binary" a.k.a. "medial" subdivision,
returning a primal simplicial complex.
"""
binary_subdivision(s::EmbeddedDeltaSet1D) = binary_subdivision_1D(s)
binary_subdivision(s::EmbeddedDeltaSet2D) = binary_subdivision_2D(s)
binary_subdivision(s::EmbeddedDeltaSet3D) = binary_subdivision_3D(s)

function binary_subdivision_1D(s)
  sd = typeof(s)()

  add_vertices!(sd,nv(s)+ne(s))
  sd[1:nv(s), :point] = s[:point]
  sd[(nv(s)+1:nv(s)+ne(s)), :point] = (s[[:∂v0,:point]] .+ s[[:∂v1,:point]])./2

  # v0 -- m -- v1
  add_parts!(sd, :E, 2*ne(s))
  for e in edges(s)
    offset = 2*e-1
    e_idxs = SVector{2}(offset, offset+1)
    m = e+nv(s)

    sd[e_idxs, :∂v0] = m,          m
    sd[e_idxs, :∂v1] = s[e, :∂v0], s[e, :∂v1]
  end
  return sd
end

function binary_subdivision_2D(s)
  sd = binary_subdivision_1D(s)

  add_parts!(sd, :E, 3*ntriangles(s))
  add_parts!(sd, :Tri, 4*ntriangles(s))
  inc_arr = SVector{3}(0,1,2)

  #       v2
  #      /  \
  #    m1 -- m0
  #   /  \  /  \
  # v0 -- m2 -- v1
  for t in triangles(s)
    es = triangle_edges(s,t)

    # Vertex indices:
    m0, m1, m2 = es .+ nv(s)

    # Edge indices:
    m0_m1, m1_m2, m0_m2 = (3t-2 + 2ne(s)) .+ inc_arr
    m0_v1, m1_v0, m2_v0 = 2es
    v2_m0, v2_m1, v1_m2 = 2es .- 1

    # Vertex × Edge:
    sd[SVector{3}(m0_m1, m1_m2, m0_m2), :∂v0] = m1, m2, m2
    sd[SVector{3}(m0_m1, m1_m2, m0_m2), :∂v1] = m0, m1, m0

    # Edge × Triangle:
    offset = 4t-3
    tri_idxs = SVector{4}(offset, offset+1, offset+2, offset+3)
    sd[tri_idxs, :∂e0] = m1_m2, m1_m2, m0_m2, m0_m1
    sd[tri_idxs, :∂e1] = m0_m2, m2_v0, v1_m2, v2_m1
    sd[tri_idxs, :∂e2] = m0_m1, m1_v0, m0_v1, v2_m0
  end
  return sd
end

# TODO: Optimize this!
function binary_subdivision_3D(s)
  sd = binary_subdivision_2D(s)

  # Unfolded tetrahedron, 1-4 are the original vertices
  # 4 -- 6 -- 3 -- 6 -- 4
  #  \  / \ /  \ /  \ /
  #   9-- 8 -- 5 -- 7
  #    \ / \  /  \ /
  #    1 -- 10 -- 2
  #     \   /\   /
  #       9 -- 7
  #       \   /
  #         4

  # 10 and 6 which are opposite points of the octahedron, are midpoints of 1-2 and 3-4
  # Knowing the midpoint idx, look back by nv to find the edge idx

  # Outermost tetrahedra
  for tet in tetrahedra(s)
    tet_vs = tetrahedron_vertices(s, tet)
    tet_edges = tetrahedron_edges(s, tet)
    tet_tris = tetrahedron_triangles(s, tet)

    for (v, face) in zip(tet_vs, tet_tris)
      tri_edges = triangle_edges(s, face)
      mids = setdiff(tet_edges, tri_edges) .+ nv(s)
      glue_sorted_tetrahedron!(sd, v, mids...)
    end

    # Inner tetrahedra
    base_edge = last(tet_edges)
    base_mid = base_edge + nv(s) # 10
    v0 = s[base_edge, :∂v0]
    v1 = s[base_edge, :∂v1]

    t0 = only(tet_tris[tet_vs .== v0])
    t1 = only(tet_tris[tet_vs .== v1])

    opp_edge = only(intersect(triangle_edges(s, t0), triangle_edges(s, t1)))
    opp_mid = opp_edge + nv(s) # 6

    for t in tet_tris
      mid_face = 4 * t - 3 # Middle face of original triangle from binary subdivision
      glue_sorted_tetrahedron!(sd, union(base_mid, opp_mid, triangle_vertices(sd, mid_face)...)...)
    end
  end
  return sd
end


function binary_subdivision_map(s)
  sd = binary_subdivision(s)

  nentries = nv(s) + 2*ne(s)

  I = zeros(Int32, nentries)
  J = zeros(Int32, nentries)
  V = ones(nentries)

  # Map old point back to same point
  for i in vertices(s) I[i]=J[i]=i; end

  # Map edge points to midpoint by average
  for i in edges(s)
    arr_i = nv(s) + 2i - 1
    shift_i = nv(s) + i

    I[arr_i], I[arr_i+1] = s[i, :∂v0], s[i, :∂v1]
    J[arr_i], J[arr_i+1] = shift_i, shift_i
    V[arr_i], V[arr_i+1] = 1/2, 1/2
  end

  PrimalGeometricMap(sd,s,sparse(I,J,V))
end

"""
Subdivide each triangle into 9 via cubic subdivision, returning a primal simplicial complex.
"""
function cubic_subdivision(s::Union{EmbeddedDeltaSet1D, EmbeddedDeltaSet2D})
  ntriangles(s::EmbeddedDeltaSet1D) = 0
  ntriangles(s::EmbeddedDeltaSet2D) = nparts(s,:Tri)
  triangles(s::EmbeddedDeltaSet1D) = 1:0
  triangles(s::EmbeddedDeltaSet2D) = parts(s,:Tri)
  sd = typeof(s)()
  add_vertices!(sd, nv(s)+2ne(s)+ntriangles(s))
  sd[1:nv(s), :point] =
    s[:point]
  sd[(nv(s)+1:nv(s)+2ne(s)), :point] = reduce(vcat, map(edges(s)) do e
    v0, v1 = s[s[e, :∂v0], :point], s[s[e, :∂v1], :point]
    [(2v0 + v1)/3, (v0 + 2v1)/3]
  end)
  sd[(nv(s)+2ne(s)+1:nv(s)+2ne(s)+ntriangles(s)), :point] =
    (s[[:∂e1,:∂v1,:point]] .+ s[[:∂e0,:∂v1,:point]] .+ s[[:∂e0,:∂v0,:point]])./3

  # v0 -> m0 -> m1 -> v1
  add_parts!(sd, :E, 3ne(s)+9ntriangles(s))
  for e in edges(s)
    offset = 3*e-2
    e_idxs = SVector{3}(offset, offset+1, offset+2)
    m0, m1 = nv(s)+2*e-1, nv(s)+2*e
    v0, v1 = s[e, :∂v0], s[e, :∂v1]

    sd[e_idxs, :∂v0] = m0, m1, v1
    sd[e_idxs, :∂v1] = v0, m0, m1
  end

  #          030
  #         ^  v
  #       021 > 120
  #      ^  ^  ^  v
  #    012< 111  >210
  #   ^  ^  ^  ^  ^  v
  # 003 >102  >201  >300
  if s isa EmbeddedDeltaSet2D
    add_parts!(sd, :Tri, 9ntriangles(s))
  end
  inc_arr = SVector{9}(0,1,2,3,4,5,6,7,8)
  for t in triangles(s)
    es = triangle_edges(s,t)

    # Vertex indices:
    m012, m102, m120 = 2*es .+ nv(s) .- 1
    m021, m201, m210 = 2*es .+ nv(s)
    m111 = nv(s)+2ne(s)+t

    # Edge indices:
    ms = (3ne(s) + 9t-8) .+ inc_arr
    m201_m210, m201_m111, m102_m111,
    m102_m012, m111_m012, m111_m021,
    m111_m210, m111_m120, m021_m120 = ms
    m021_v030, m201_v300, m210_v300 = 3es
    m012_m021, m102_m201, m120_m210 = 3es .- 1
    v003_m012, v003_m102, v030_m120 = 3es .- 2

    # Vertex × Edge:
    sd[ms, :∂v0] = m210, m111, m111, m012, m012, m021, m210, m120, m120
    sd[ms, :∂v1] = m201, m201, m102, m102, m111, m111, m111, m111, m021

    # Edge × Triangle:
    offset = 9t-8
    tri_idxs = offset .+ inc_arr
    sd[tri_idxs, :∂e0] = m210_v300, m111_m210, m201_m111, m111_m012, m102_m012, m120_m210, m021_m120, m012_m021, v030_m120
    sd[tri_idxs, :∂e1] = m201_v300, m201_m210, m102_m111, m102_m012, v003_m012, m111_m210, m111_m120, m111_m021, m021_m120
    sd[tri_idxs, :∂e2] = m201_m210, m201_m111, m102_m201, m102_m111, v003_m102, m111_m120, m111_m021, m111_m012, m021_v030
  end
  sd
end

function cubic_subdivision_map(s)
  ntriangles(s::EmbeddedDeltaSet1D) = 0
  ntriangles(s::EmbeddedDeltaSet2D) = nparts(s,:Tri)
  triangles(s::EmbeddedDeltaSet1D) = 1:0
  triangles(s::EmbeddedDeltaSet2D) = parts(s,:Tri)
  sd = cubic_subdivision(s)

  nentries = 1*nv(s) + 2*(2*ne(s)) + 3*ntriangles(s)

  I = zeros(Int32, nentries)
  J = zeros(Int32, nentries)
  V = ones(nentries)

  # Original points:
  for i in vertices(s) I[i]=J[i]=i; end

  # Points along original edges:
  for i in edges(s)
    arr_i = nv(s) + 4i - 3
    arr_idxs = SVector(arr_i, arr_i+1, arr_i+2, arr_i+3)
    shift_i = nv(s) + 2*i - 1

    I[arr_idxs] = [s[i, :∂v0], s[i, :∂v1], s[i, :∂v1], s[i, :∂v0]]
    J[arr_idxs] = [shift_i, shift_i, shift_i+1, shift_i+1]
    V[arr_idxs] = [2/3, 1/3, 2/3, 1/3]
  end

  # Points at triangle centers:
  for i in triangles(s)
    arr_i = (nv(s) + (2*2*ne(s))) + 3*i - 2
    arr_idxs = SVector(arr_i, arr_i+1, arr_i+2)
    shift_i = nv(s) + 2*ne(s) + i

    I[arr_idxs] = [s[s[i, :∂e1], :∂v1], s[s[i, :∂e0], :∂v1], s[s[i, :∂e0], :∂v0]]
    J[arr_idxs] .= shift_i
    V[arr_idxs] = [1/3, 1/3, 1/3]
  end

  PrimalGeometricMap(sd,s,sparse(I,J,V))
end

repeated_subdivisions(k, ss, subdivider) =
  accumulate((x,_) -> subdivider(x), 1:k; init=ss)

# Multigrid Algorithms
#---------------------

# TODO:
# - Smarter calculations for steps and cycles,
# - Input arbitrary iterative solver,
# - Implement weighted Jacobi and maybe Gauss-Seidel,
# - Masking for boundary condtions
# - This could use Galerkin conditions to construct As from As[1]
# - Add maxcycles and tolerances
"""
Solve `Ax=b` on `s` with initial guess `u` using , for `cycles` V-cycles, performing `md.steps` steps of the
conjugate gradient method on each mesh and going through
`cycles` total V-cycles. Everything is just matrices and vectors
at this point.

Warning:
Quoting from the Krylov.jl documentation:
> itmax: the maximum number of iterations. If itmax=0, the default number of iterations is set to 2n;

, where n is the length of the solution vector.
We diverge from this behavior and perform no iterations when the corresponding element of `md.steps` is `0`.

`alg` is a Krylov.jl method, probably either the default `cg` or
`gmres`.
"""
multigrid_vcycles(u, b, md, cycles, alg=cg) = multigrid_μ_cycles(u, b, md, cycles, alg, 1)

"""
Just the same as `multigrid_vcycles` but with W-cycles.
"""
multigrid_wcycles(u, b, md, cycles, alg=cg) = multigrid_μ_cycles(u, b, md, cycles, alg, 2)

function multigrid_μ_cycles(u, b, md::MultigridData, cycles, alg=cg, μ=1)
  cycles == 0 && return u
  u = _multigrid_μ_cycle(u,b,md,alg,μ)
  multigrid_μ_cycles(u,b,md,cycles-1,alg,μ)
end

"""
The full multigrid framework: start at the coarsest grid and
work your way up, applying V-cycles or W-cycles at each level
according as μ is 1 or 2.
"""
function full_multigrid(b, md::MultigridData, cycles, alg=cg, μ=1)
  z_f = zeros(size(b))
  if length(md) > 1
    r,p = car(md)[2:3]
    b_c = r * b
    z_c = full_multigrid(b_c, cdr(md), cycles, alg, μ)
    z_f = p * z_c
  end
  multigrid_μ_cycles(z_f,b,md,cycles,alg,μ)
end

function _multigrid_μ_cycle(u, b, md::MultigridData, alg=cg, μ=1)
  A,r,p,steps = car(md)
  # Manually perform 0 steps, unlike the 2n step default of Krylov.jl.
  u = steps == 0 ? u : alg(A,b,u,itmax=steps)[1]
  length(md) == 1 && return u
  r_f = b - A*u
  r_c = r * r_f
  z = _multigrid_μ_cycle(zeros(size(r_c)), r_c, cdr(md), alg, μ)
  if μ > 1
    z = _multigrid_μ_cycle(z, r_c, cdr(md), alg, μ-1)
  end
  u += p * z
  # Manually perform 0 steps, unlike the 2n step default of Krylov.jl.
  u = steps == 0 ? u : alg(A, b, u, itmax=steps)[1]
end

end
