module Multigrid
using CombinatorialSpaces
using GeometryBasics:Point3, Point2
using Krylov, Catlab, SparseArrays
using ..SimplicialSets
import Catlab: dom,codom
export multigrid_vcycles, multigrid_wcycles, full_multigrid, repeated_subdivisions, binary_subdivision_map, dom, codom, as_matrix, MultigridData, PrimitiveGeometricMapSeries, finest_mesh
const Point3D = Point3{Float64}
const Point2D = Point2{Float64}

struct PrimitiveGeometricMap{D,M}
  domain::D
  codomain::D
  matrix::M
end

dom(f::PrimitiveGeometricMap) = f.domain
codom(f::PrimitiveGeometricMap) = f.codomain
as_matrix(f::PrimitiveGeometricMap) = f.matrix

function is_simplicial_complex(s)
  allunique(map(1:ne(s)) do i edge_vertices(s,i) end) &&
  allunique(map(1:ntriangles(s)) do i triangle_vertices(s,i) end)
end

"""
Subdivide each triangle into 4 via "binary" a.k.a. "medial" subdivision, returning a primal simplicial complex.

Binary subdivision results in triangles that resemble the "tri-force" symbol from Legend of Zelda.
"""
function binary_subdivision(s)
  is_simplicial_complex(s) || error("Subdivision is supported only for simplicial complexes.")
  sd = typeof(s)()
  add_vertices!(sd,nv(s))
  add_vertices!(sd,ne(s))
  sd[:point] = [s[:point];
                (subpart(s,(:∂v0,:point)).+subpart(s,(:∂v1,:point)))/2]
  succ3(i) = (i+1)%3 == 0 ? 3 : (i+1)%3
  for t in triangles(s)
    es = triangle_edges(s,t)
    glue_sorted_triangle!(sd,(es.+nv(s))...)
    for i in 1:3
      glue_sorted_triangle!(sd,
        triangle_vertices(s,t)[i],
        triangle_edges(s,t)[succ3(i)]+nv(s),
        triangle_edges(s,t)[succ3(i+1)]+nv(s))
    end
  end
  sd
end

function binary_subdivision_map(s)
  sd = binary_subdivision(s)
  mat = spzeros(nv(s),nv(sd))
  for i in 1:nv(s) mat[i,i] = 1. end
  for i in 1:ne(s) 
    x, y = s[:∂v0][i], s[:∂v1][i]
    mat[x,i+nv(s)] = 1/2
    mat[y,i+nv(s)] = 1/2
  end
  PrimitiveGeometricMap(sd,s,mat)
end

function repeated_subdivisions(k,ss,subdivider)
  map(1:k) do k′
    f = subdivider(ss) 
    ss = dom(f)
    f
  end
end

"""
This struct is meant to organize the mesh data required for multigrid methods.
"""
struct PrimitiveGeometricMapSeries{D<:HasDeltaSet, M<:AbstractMatrix}
  meshes::AbstractVector{D}
  matrices::AbstractVector{M}

  function PrimitiveGeometricMapSeries{D, M}(meshes, matrices) where {D, M}
    new(meshes, matrices)
  end

  function PrimitiveGeometricMapSeries(s, subdivider, levels, alg = Circumcenter())
    subdivs = reverse(repeated_subdivisions(levels, s, binary_subdivision_map));
    meshes = map(subdivs) do subdiv dom(subdiv) end
    push!(meshes,s)

    dual_meshes = map(meshes) do s dualize(s, alg) end
    matrices = as_matrix.(subdivs)
    PrimitiveGeometricMapSeries{typeof(first(dual_meshes)), typeof(first(matrices))}(dual_meshes, matrices)
  end
end

finest_mesh(series::PrimitiveGeometricMapSeries) = first(series.meshes)

"""
A cute little package contain your multigrid data. If there are 
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

function MultigridData(series::PrimitiveGeometricMapSeries, op::Function, s)
  ops = map(series.meshes) do sd op(sd) end
  ps = transpose.(series.matrices)
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
