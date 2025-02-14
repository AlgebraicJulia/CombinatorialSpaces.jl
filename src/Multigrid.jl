module Multigrid
using CombinatorialSpaces
using GeometryBasics:Point3, Point2
using Krylov, Catlab, SparseArrays, StaticArrays
using ..SimplicialSets
import Catlab: dom,codom
export multigrid_vcycles, multigrid_wcycles, full_multigrid, repeated_subdivisions, binary_subdivision_map, dom, codom, as_matrix, MultigridData, AbstractGeometricMapSeries, PrimalGeometricMapSeries, finest_mesh, meshes, matrices
const Point3D = Point3{Float64}
const Point2D = Point2{Float64}

struct PrimalGeometricMap{D,M}
  domain::D
  codomain::D
  matrix::M
end

dom(f::PrimalGeometricMap) = f.domain
codom(f::PrimalGeometricMap) = f.codomain
as_matrix(f::PrimalGeometricMap) = f.matrix

function is_simplicial_complex(s)
  allunique(map(1:ne(s)) do i edge_vertices(s,i) end) &&
  allunique(map(1:ntriangles(s)) do i triangle_vertices(s,i) end)
end

"""
Subdivide each triangle into 4 via "binary" a.k.a. "medial" subdivision, returning a primal simplicial complex.

Binary subdivision results in triangles that resemble the "tri-force" symbol from Legend of Zelda.
"""
# function binary_subdivision(s)
#   is_simplicial_complex(s) || error("Subdivision is supported only for simplicial complexes.")
#   sd = typeof(s)()
#   add_vertices!(sd,nv(s))
#   add_vertices!(sd,ne(s))
#   sd[:point] = [s[:point];
#                 (subpart(s,(:∂v0,:point)).+subpart(s,(:∂v1,:point)))/2]
#   succ3(i) = (i+1)%3 == 0 ? 3 : (i+1)%3
#   for t in triangles(s)
#     es = triangle_edges(s,t)
#     glue_sorted_triangle!(sd,(es.+nv(s))...)
#     for i in 1:3
#       glue_sorted_triangle!(sd,
#         triangle_vertices(s,t)[i],
#         triangle_edges(s,t)[succ3(i)]+nv(s),
#         triangle_edges(s,t)[succ3(i+1)]+nv(s))
#     end
#   end
#   sd
# end

function binary_subdivision(s::EmbeddedDeltaSet2D)
  sd = typeof(s)()
  add_vertices!(sd,nv(s)+ne(s))
  sd[1:nv(s), :point] = s[:point]
  sd[(nv(s)+1:nv(s)+ne(s)), :point] = (s[[:∂v0,:point]] .+ s[[:∂v1,:point]])./2

  succ3(i::Int) = (i+1)%3 == 0 ? 3 : (i+1)%3

  add_parts!(sd, :Tri, 4*ntriangles(s))
  for t in triangles(s)
    shift_idx = 4t-3
    es = triangle_edges(s,t)
    vs = triangle_vertices(s,t)
    for i in 1:3
      glue_sorted_triangle!(sd,shift_idx+i,
      vs[i],
      es[succ3(i)]+nv(s),
      es[succ3(i+1)]+nv(s))
    end
    glue_sorted_triangle!(sd,shift_idx,(es.+nv(s))...)
  end
  sd
end

function binary_subdivision_map(s)
  sd = binary_subdivision(s)

  nentries = nv(s) + 2*ne(s)

  I = zeros(Int32, nentries)
  J = zeros(Int32, nentries)
  V = ones(nentries)

  # Map old point back to same point
  for i in 1:nv(s) I[i]=J[i]=i; end

  # Map edge points to midpoint by average
  for i in 1:ne(s)
    arr_i = nv(s) + 2i - 1
    shift_i = nv(s) + i

    I[arr_i] = s[i, :∂v0]
    I[arr_i+1] = s[i, :∂v1]

    J[arr_i] = shift_i
    J[arr_i+1] = shift_i

    V[arr_i] = 1/2
    V[arr_i+1] = 1/2
  end

  PrimalGeometricMap(sd,s,sparse(I,J,V))
end

function repeated_subdivisions(k,ss,subdivider)
  is_simplicial_complex(ss) || error("Subdivision is supported only for simplicial complexes.")
  map(1:k) do k′
    f = subdivider(ss)
    ss = dom(f)
    f
  end
end

# Different means of representing a series of complexes with maps between them should sub-type this abstract type.
# Those concrete types should then provide a constructor for `MultigridData`.
"""
Organizes the mesh data that results from mesh refinement through a subdivision method.

See also: [`PrimalGeometricMapSeries`](@ref).
"""
abstract type AbstractGeometricMapSeries end

"""
Organize a series of dual complexes and maps between primal vertices between them.

See also: [`AbstractGeometricMapSeries`](@ref).
"""
struct PrimalGeometricMapSeries{D<:HasDeltaSet, M<:AbstractMatrix} <: AbstractGeometricMapSeries
  meshes::AbstractVector{D}
  matrices::AbstractVector{M}
end

meshes(series::PrimalGeometricMapSeries) = series.meshes
matrices(series::PrimalGeometricMapSeries) = series.matrices

"""    function PrimalGeometricMapSeries(s::HasDeltaSet, subdivider::Function, levels::Int, alg = Circumcenter())

Construct a `PrimalGeometricMapSeries` given a primal mesh `s` and a subdivider function like `binary_subdivision`, `levels` times.

The `PrimalGeometricMapSeries` returned contains a list of `levels + 1` dual complexes, with `levels` matrices between the primal vertices of each.

See also: [`AbstractGeometricMapSeries`](@ref), [`finest_mesh`](@ref).
"""
function PrimalGeometricMapSeries(s::HasDeltaSet, subdivider::Function, levels::Int, alg = Circumcenter())
  subdivs = reverse(repeated_subdivisions(levels, s, binary_subdivision_map));
  meshes = dom.(subdivs)
  push!(meshes, s)

  dual_meshes = map(s -> dualize(s, alg), meshes)

  matrices = as_matrix.(subdivs)
  PrimalGeometricMapSeries{typeof(first(dual_meshes)), typeof(first(matrices))}(dual_meshes, matrices)
end

"""    finest_mesh(series::PrimalGeometricMapSeries)

Return the mesh in a `PrimalGeometricMapSeries` with the highest resolution.
"""
finest_mesh(series::PrimalGeometricMapSeries) = first(series.meshes)

"""
Contains the data require for multigrid methods. If there are
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
"""
Construct a `MultigridData` with a constant step radius
on each grid.
"""
MultigridData(g,r,p,s::Int) = MultigridData{typeof(g),typeof(r)}(g,r,p,fill(s,length(g)))

function MultigridData(series::PrimalGeometricMapSeries, op::Function, s)
  ops = map(meshes(series)) do sd op(sd) end
  ps = transpose.(matrices(series))
  rs = transpose.(ps)./4.0 #4 is the biggest row sum that occurs for triforce, this is not clearly the correct scaling

  MultigridData(ops, rs, ps, s)
end

"""
Get the leading grid, restriction, prolongation, and step radius.
"""
car(md::MultigridData) = length(md) > 1 ? (md.operators[1],md.restrictions[1],md.prolongations[1],md.steps[1]) : length(md) > 0 ? (md.operators[1],nothing,nothing,md.steps[1]) : (nothing,nothing,nothing,nothing)

"""
Remove the leading grid, restriction, prolongation, and step radius.
"""
cdr(md::MultigridData) = length(md) > 1 ? MultigridData(md.operators[2:end],md.restrictions[2:end],md.prolongations[2:end],md.steps[2:end]) : error("Not enough grids remaining in $md to take the cdr.")

"""
The length of a `MultigridData` is its number of grids.
"""
Base.length(md::MultigridData) = length(md.operators)

"""
Decrement the number of (eg V-)cycles left to be performed.
"""
decrement_cycles(md::MultigridData) = MultigridData(md.operators,md.restrictions,md.prolongations,md.steps,md.cycles-1)

# TODO:
# - Smarter calculations for steps and cycles,
# - Input arbitrary iterative solver,
# - Implement weighted Jacobi and maybe Gauss-Seidel,
# - Masking for boundary condtions
# - This could use Galerkin conditions to construct As from As[1]
# - Add maxcycles and tolerances
"""
Solve `Ax=b` on `s` with initial guess `u` using , for `cycles` V-cycles, performing `steps` steps of the
conjugate gradient method on each mesh and going through
`cycles` total V-cycles. Everything is just matrices and vectors
at this point.

`alg` is a Krylov.jl method, probably either the default `cg` or
`gmres`.
"""
multigrid_vcycles(u,b,md,cycles,alg=cg) = multigrid_μ_cycles(u,b,md,cycles,alg,1)

"""
Just the same as `multigrid_vcycles` but with W-cycles.
"""
multigrid_wcycles(u,b,md,cycles,alg=cg) = multigrid_μ_cycles(u,b,md,cycles,alg,2)
function multigrid_μ_cycles(u,b,md::MultigridData,cycles,alg=cg,μ=1)
  cycles == 0 && return u
  u = _multigrid_μ_cycle(u,b,md,alg,μ)
  multigrid_μ_cycles(u,b,md,cycles-1,alg,μ)
end

"""
The full multigrid framework: start at the coarsest grid and
work your way up, applying V-cycles or W-cycles at each level
according as μ is 1 or 2.
"""
function full_multigrid(b,md::MultigridData,cycles,alg=cg,μ=1)
  z_f = zeros(size(b))
  if length(md) > 1
    r,p = car(md)[2:3]
    b_c = r * b
    z_c = full_multigrid(b_c,cdr(md),cycles,alg,μ)
    z_f = p * z_c
  end
  multigrid_μ_cycles(z_f,b,md,cycles,alg,μ)
end

function _multigrid_μ_cycle(u,b,md::MultigridData,alg=cg,μ=1)
  A,r,p,s = car(md)
  u = alg(A,b,u,itmax=s)[1]
  if length(md) == 1
    return u
  end
  r_f = b - A*u
  r_c = r * r_f
  z = _multigrid_μ_cycle(zeros(size(r_c)),r_c,cdr(md),alg,μ)
  if μ > 1
    z = _multigrid_μ_cycle(z,r_c,cdr(md),alg,μ-1)
  end
  u += p * z
  u = alg(A,b,u,itmax=s)[1]
end

end
