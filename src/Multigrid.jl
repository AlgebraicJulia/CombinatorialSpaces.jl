module Multigrid
using GeometryBasics:Point3
export multigrid_vcycles, Point3D
Point3D = Point3{Float64}

# TODO: Smarter calculations for steps and cycles, input arbitrary iterative solver, implement weighted Jacobi and maybe Gauss-Seidel, masking for boundary condtions

#This could use Galerkin conditions to construct As from As[1]
"""
Solve `Ax=b` on `s` with initial guess `u` using fine grid 
operator `A`, restriction operators `rs`, and prolongation
operators `ps`, for V-cycles, performing `steps` steps of the 
conjugate gradient method on each mesh and going through 
`cycles` total V-cycles. Everything is just matrices and vectors
at this point.
"""
function multigrid_vcycles(u,b,As,rs,ps,steps,cycles=1)
  cycles == 0 && return u 
  u = cg(As[1],b,u,itmax=steps)[1] 
  if length(As) == 1 
    return u 
  end 
  r_f = b - As[1]*u 
  r_c = rs[1] * r_f 
  z = multigrid_vcycles(zeros(size(r_c)),r_c,As[2:end],rs[2:end],ps[2:end],steps,cycles)
  u += ps[1] * z 
  u = cg(As[1],b,u,itmax=steps)[1] 
  multigrid_vcycles(u,b,As,rs,ps,steps,cycles-1) 
end


#masked(A,x,b)

using CairoMakie
s1 = triangulated_grid(1,1,1/2,1/2,Point3D)
s2 = triangulated_grid(1,1,1/4,1/4,Point3D)
fr = Figure();
ax = CairoMakie.Axis(fr[1,1],aspect=sqrt(3))
wireframe!(ax,s)
wireframe!(ax,s,color=(:orange,0.5))
fr

function triforce_subdivision(s::EmbeddedDeltaSet2D)
  sd = typeof(s)()
  add_vertices!(sd,nv(s))
  add_vertices!(sd,ne(s))
  sd[:point] = [s[:point];
                (subpart(s,(:∂v0,:point)).+subpart(s,(:∂v1,:point)))/2]
#  add_edges!(sd,1+nv(s):ne(s)+nv(s),s[:∂v0])
#  add_edges!(sd,1+nv(s):ne(s)+nv(s),s[:∂v1])
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

function triforce_subdivision_map(s::EmbeddedDeltaSet2D)
  sd = triforce_subdivision(s)
  mat = spzeros(nv(s),nv(sd))
  for i in 1:nv(s) mat[i,i] = 1. end
  for i in 1:ne(s) 
    x,y = s[:∂v0][i],s[:∂v1][i]
    mat[x,x+nv(s)] = 1/2
    mat[y,y+nv(s)] = 1/2
  end
  GeometricMap(SimplicialComplex(sd),SimplicialComplex(s),mat)
end
end