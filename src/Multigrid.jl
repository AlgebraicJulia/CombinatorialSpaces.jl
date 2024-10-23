module Multigrid
using GeometryBasics:Point3, Point2
using Krylov, Catlab, SparseArrays
using ..SimplicialSets
import Catlab: dom,codom
export multigrid_vcycles, repeated_subdivisions, Point3D, Point2D, triforce_subdivision_map, dom, codom, as_matrix
Point3D = Point3{Float64}
Point2D = Point2{Float64}

struct PrimitiveGeometricMap{D,M}
  domain::D
  codomain::D
  matrix::M
end

dom(f::PrimitiveGeometricMap) = f.domain
codom(f::PrimitiveGeometricMap) = f.codomain
as_matrix(f::PrimitiveGeometricMap) = f.matrix

is_simplicial_complex(s) = true
function triforce_subdivision(s)
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

function triforce_subdivision_map(s)
  sd = triforce_subdivision(s)
  mat = spzeros(nv(s),nv(sd))
  for i in 1:nv(s) mat[i,i] = 1. end
  for i in 1:ne(s) 
    x,y = s[:∂v0][i],s[:∂v1][i]
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

# TODO: Smarter calculations for steps and cycles, input arbitrary iterative solver, implement weighted Jacobi and maybe Gauss-Seidel, masking for boundary condtions

#This could use Galerkin conditions to construct As from As[1]
#Add maxcycles and tolerances
"""
Solve `Ax=b` on `s` with initial guess `u` using fine grid 
operator `A`, restriction operators `rs`, and prolongation
operators `ps`, for V-cycles, performing `steps` steps of the 
conjugate gradient method on each mesh and going through 
`cycles` total V-cycles. Everything is just matrices and vectors
at this point.

`alg` is a Krylov.jl method, probably either the default `cg` or
`gmres`.
"""
function multigrid_vcycles(u,b,As,rs,ps,steps,cycles=1,alg=cg)
  cycles == 0 && return u 
  u = alg(As[1],b,u,itmax=steps)[1] 
  if length(As) == 1 
    return u 
  end 
  r_f = b - As[1]*u 
  r_c = rs[1] * r_f 
  z = multigrid_vcycles(zeros(size(r_c)),r_c,As[2:end],rs[2:end],ps[2:end],steps,cycles)
  u += ps[1] * z 
  u = alg(As[1],b,u,itmax=steps)[1] 
  multigrid_vcycles(u,b,As,rs,ps,steps,cycles-1) 
end

end