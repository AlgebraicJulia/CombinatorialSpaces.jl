module Multigrid
export multigrid_vcycles

# TODO: Smarter calculations for steps and cycles, input arbitrary iterative solver, implement weighted Jacobi and maybe Gauss-Seidel, masking for boundary condtions

#This could use Galerkin conditions to construct As from As[1]
```
Solve `As[1]x=b` with initial guess `u` using multigrid 
operators `As`, restriction operators `rs`, and prolongation
operators `ps`, for V-cycles, performing `steps` steps of the 
conjugate gradient method on each mesh and going through 
`cycles` total V-cycles. Everything is just matrices and vectors
at this point.
```
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

end