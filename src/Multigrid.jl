module Multigrid
using GeometryBasics:Point3
using Krylov
#using ..SimplicialComplexes: triforce_subdivision_map
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


#masked(A,x,b)
#=
using CairoMakie
s1 = triangulated_grid(1,1,1/2,1/2,Point3D)
s2 = triangulated_grid(1,1,1/4,1/4,Point3D)
fr = Figure();
ax = CairoMakie.Axis(fr[1,1],aspect=sqrt(3))
wireframe!(ax,s1)
wireframe!(ax,s2,color=(:orange,0.5))
fr
=#


end