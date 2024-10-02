# Solving Poisson's equation on a multiscale regular 1-D mesh

CombinatorialSpaces provides advanced capabilities for working with irregular and complex meshes
in up to three dimensions. For a first example of working across meshes of multiple scales at once,
we reproduce a 1-D Poisson equation example from Golub and van Loan's "Matrix Computations", 11.6.

## Poisson equation

In general, the Poisson equation asks for a function on a manifold ``M`` with boundary with a fixed Laplacian on the interior, satisfying
boundary conditions that may be given in various forms, such as the Dirichlet conditions:

```math
\Delta u = -f,u\mid_{\partial M} = f_0
```

In one dimension, on the interval ``[0,1]``, this specializes to the equation
```math
\frac{d^2u}{dx^2} = -f(x), u(0)=u_0, u(1)=u_1.
```

If we subdivide the interval into ``m`` congruent pieces of radius ``h=1/m``, then we get the discretized equations
```math
\frac{u((i-1)h)-2u(ih)+u((i+1)h)}{h^2}\approx -f(ih)
```
for ``i\in \{1,\ldots,m-1\}``. Since ``u(0)=u_0,u(1)=u_1`` are given by the boundary conditions, we can move them to 
the right-hand side of the first and last equations, producing the linear system ``Au=b`` for 
``u=[u(h),u(2h),\ldots,u((m-1)h)],b=[h^2f(h)+u_0,h^2f(2h),\ldots,h^2f((m-1)h),h^2f(mh)+u_1],`` and
```math
A=\left(\begin{matrix}
2&-1&0&0&\cdots&0\\
-1&2&-1&0&\cdots&0\\
0&-1&2&-1&\cdots&0\\
\vdots&&&&\vdots\\
0&\cdots&0&-1&2&-1\\
0&\cdots&0&0&-1&2
\end{matrix}\right)
```

We are thus led to consider the solution of  ``Au=b`` for this tridiagonal ``A``. Tridiagonal systems are easy to solve naively, 
of course, but this example also gives a nice illustration of the multi-grid method. The latter proceeds by mixing steps of solution
via some iterative solver with approximate corrections obtained on a coarser grid, and works particularly well for this equation
where there is a neat division between high-frequency and low-frequency contributors to the solution.

Specifically, we will proceed by restricting discretized functions from a grid of radius ``h`` to one of radius ``2h`` and
prolonging back from there, by taking the weighted average of values near a coarse-grid point, weighting the point itself double,
for restriction, and making the value at a fine-grid point not in the coarse grid average the adjacent coarse values for prolongation.
It's interesting to note that restriction after prolongation is not idempotent, but instead smears some heat around away from
where it started.

## The problem solved directly via multigrid

```@example gvl
using SparseArrays
using LinearAlgebra: norm

#The tridiagonal Laplacian discussed above, with m=2^k.
function sparse_square_laplacian(k)
  N,h = 2^k-1, 1/(2^k)
  A = spzeros(N,N)
  for i in 1:N
    A[i,i] = 2
    if i > 1 A[i,i-1] = -1 end
    if i < N A[i,i+1] = -1 end
  end
  1/h^2 * A
end
#The restriction matrix to half as fine a grid.
function sparse_restriction(k)
  N,M = 2^k-1, 2^(k-1)-1
  A = spzeros(M,N)
  for i in 1:M
    A[i,2i-1:2i+1] = [1,2,1]
  end
  1/4*A
end
#The prolongation matrix from coarse to fine.
sparse_prolongation(k) = 2*transpose(sparse_restriction(k))

sparse_square_laplacian(3)
```
```@example gvl
sparse_restriction(3)
```
```@example gvl
sparse_prolongation(3)
```

Here is the function that actually runs some multigrid v-cycles, 
using the conjugate gradient method from `Krylov.jl` to iterate toward
solution on each grid.

```@example gvl
using Krylov

function multigrid_vcycles(u,b,As,rs,ps,steps,cycles=1)
  cycles == 0 && return u
  u = cg(As[1],b,u,itmax=steps)[1]
  if length(As) == 1 #on coarsest grid
    return u
  end
  #smooth, update error, restrict, recurse, prolong, smooth
  r_f = b - As[1]*u #residual
  r_c = rs[1] * r_f #restrict
  z = multigrid_vcycles(zeros(size(r_c)),r_c,As[2:end],rs[2:end],ps[2:end],steps,cycles) #recurse
  u += ps[1] * z #prolong. 
  u = cg(As[1],b,u,itmax=steps)[1] #smooth again
  multigrid_vcycles(u,b,As,rs,ps,steps,cycles-1)
end
```

Here is a function that sets up and runs a v-cycle for the 
Poisson problem on a mesh with ``2^k+1`` points, on all
meshes down to ``3`` points,
smoothing using ``s`` steps of the Krylov method on each mesh,
with a random target vector,
and continuing through the entire cycle ``c`` times. 

In the example, we are solving the Poisson equation on a grid
with ``2^15+1`` points using just ``15*7*3`` steps of
the conjugate gradient method. This is, honestly, pretty crazy.

```@example gvl
function test_vcycle_1D(k,s,c)
  b=rand(2^k-1)
  N = 2^k-1 
  ls = reverse([sparse_square_laplacian(k′) for k′ in 1:k])
  is = reverse([sparse_restriction(k′) for k′ in 2:k])
  ps = reverse([sparse_prolongation(k′) for k′ in 2:k])
  u = zeros(N)
  norm(ls[1]*multigrid_vcycles(u,b,ls,is,ps,s,c)-b)/norm(b)
end
@time test_vcycle_1D(15,7,3)
```


Let's examine some particular cases of these equations. For both, we need a mesh and some discrete differential operators.

```@example euler
using CairoMakie, CombinatorialSpaces, StaticArrays
using CombinatorialSpaces.DiscreteExteriorCalculus: eval_constant_primal_form
using GeometryBasics: Point3d
using LinearAlgebra: norm

s = triangulated_grid(100,100,5,5,Point3d);
sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}(s);
subdivide_duals!(sd, Barycenter());

f = Figure()
ax = CairoMakie.Axis(f[1,1])
wireframe!(ax, s)

f
```

Now that we have our mesh, let's allocate our discrete differential operators:

```@example euler
d0 = dec_dual_derivative(0, sd)
d1 = dec_differential(1, sd);
s1 = dec_hodge_star(1, sd);
s2 = dec_hodge_star(2, sd);
ι1 = interior_product_dd(Tuple{1,1}, sd)
ι2 = interior_product_dd(Tuple{1,2}, sd)
ℒ1 = ℒ_dd(Tuple{1,1}, sd);
```

```@setup euler
using ACSets

sharp_dd = ♯_mat(sd, LLSDDSharp())
function plot_dvf(sd, X; ls=1f0, title="Dual Vector Field")
  X♯ = sharp_dd * X
  # Makie will throw an error if the colorrange end points are not unique:
  f = Figure()
  ax = Axis(f[1, 1], title=title)
  wireframe!(ax, sd, color=:gray95)
  extX = extrema(norm.(X♯))
  if (abs(extX[1] - extX[2]) > 1e-4)
    range = extX
    scatter!(ax, getindex.(sd[sd[:tri_center], :dual_point],1), getindex.(sd[sd[:tri_center], :dual_point],2), color = norm.(X♯), colorrange=range)
    Colorbar(f[1,2], limits=range)
  end
  arrows!(ax, getindex.(sd[sd[:tri_center], :dual_point],1), getindex.(sd[sd[:tri_center], :dual_point],2), getindex.(X♯,1), getindex.(X♯,2), lengthscale=ls)
  hidedecorations!(ax)
  f
end

sharp_pp = ♯_mat(sd, AltPPSharp())
function plot_vf(sd, X; ls=1f0, title="Primal Vector Field")
  X♯ = sharp_pp * X
  # Makie will throw an error if the colorrange end points are not unique:
  f = Figure()
  ax = Axis(f[1, 1], title=title)
  wireframe!(ax, sd, color=:gray95)
  extX = extrema(norm.(X♯))
  if (abs(extX[1] - extX[2]) > 1e-4)
    range = extX
    scatter!(ax, getindex.(sd[:point],1), getindex.(sd[:point],2), color = norm.(X♯), colorrange=range)
    Colorbar(f[1,2], limits=range)
  end
  arrows!(ax, getindex.(sd[:point],1), getindex.(sd[:point],2), getindex.(X♯,1), getindex.(X♯,2), lengthscale=ls)
  hidedecorations!(ax)
  f
end

function plot_dual0form(sd, f0; title="Dual 0-form")
  ps  = (stack(sd[sd[:tri_center], :dual_point])[[1,2],:])'
  f = Figure(); ax = CairoMakie.Axis(f[1,1], title=title);
  if (minimum(f0) ≈ maximum(f0))
    sct = scatter!(ax, ps)
  else
    sct = scatter!(ax, ps,
        color=f0);
    Colorbar(f[1,2], sct)
  end
  f
end

function boundary_inds(::Type{Val{0}}, s)
  ∂1_inds = boundary_inds(Val{1}, s)
  unique(vcat(s[∂1_inds,:∂v0],s[∂1_inds,:∂v1]))
end
function boundary_inds(::Type{Val{1}}, s)
  collect(findall(x -> x != 0, boundary(Val{2},s) * fill(1,ntriangles(s))))
end
function boundary_inds(::Type{Val{2}}, s)
  ∂1_inds = boundary_inds(Val{1}, s)
  inds = map([:∂e0, :∂e1, :∂e2]) do esym
    vcat(incident(s, ∂1_inds, esym)...)
  end
  unique(vcat(inds...))
end
```

## First Case

In this first case, we will explicitly provide initial values for `u`. We will solve for pressure and the time derivative of `u` and check that they are what we expect. Note that we will set the mass budget, `b`, to 0.

Let's provide a flow field of unit magnitude, static throughout the domain. We want to store this as a 1-form. We can create a 1-form by "flattening" a vector field, performing many line integrals to store values on the edges of the mesh. Since we want to store our flow as a "dual" 1-form (on the edges of the dual mesh), we can use the Hodge star operator to convert from a primal 1-form to a dual 1-form. Since the values of a 1-form can be unintuitive, we will "sharpen" the 1-form back to a vector field when visualizing.

```@example euler
X♯ = SVector{3,Float64}(1/√2,1/√2,0)
u = s1 * eval_constant_primal_form(sd, X♯)

plot_dvf(sd, u, title="Flow")
```

Let's look at the self-advection term, in which we take the lie derivative of `u` along itself, and subtract half of the gradient of its inner product. (See Marsden, Ratiu, and Abraham for a derivation.) Recall that our flow `u` is static throughout the domain, so we should expect this term to be 0 throughout the interior of the domain, where it is not affected by boundary conditions.

The Lie derivative encodes how a differential form changes along a vector field. For our case of many parallel streamlines, and in which the magnitude is identical everywhere, we expect such a quantity to be 0. However, when discretizing, we have to make some assumptions about what is happening "outside" of the domain, and these assumptions have implications on the data stored on the boundary of the domain. In our discretization, we assume the flow outside the domain is 0. Thus, our Lie derivative along the boundary points inward:

```@example euler
lie_u_u = ℒ1(u,u)

plot_dvf(sd, lie_u_u, title="Lie Derivative of Flow with Itself")
```

```@example euler
selfadv = ℒ1(u,u) - 0.5*d0*ι1(u,u)

plot_dvf(sd, selfadv, title="Self-Advection")
```

Now, let's solve for pressure. We can set up a Poisson problem on the divergence of the self-advection term we computed. Recall that divergence can be computed as ``\star d \star``, and the Laplacian as ``d \star d \star``. To solve a Poisson problem, we reverse the order of the operations, and take advantage of the fact that solving the inverse hodge star is equivalent to multiplying by the hodge star.

```@example euler
div(x) = s2 * d1 * (s1 \ x);
solveΔ(x) = float.(d0) \ (s1 * (float.(d1) \ (s2 \ x)))

p = (solveΔ ∘ div)(selfadv)

plot_dual0form(sd, p, title="Pressure")
```

We see that we have a nonzero pressure of exactly 2 across the interior of the domain.

```@example euler
dp = d0*p

plot_dvf(sd, dp, title="Pressure Gradient")
```

Based on our initial conditions and the way that we computed pressure, we expect that the time derivative of `u` should be 0 on the interior of the domain, where it is not affected by boundary conditions.

```@example euler
∂ₜu = -selfadv - dp;

plot_dvf(sd, ∂ₜu, title="Time Derivative")
```

We see that we do indeed find zero-vectors throughout the interior of the domain as expected.

## Second Case

For this second case, we will specify that the time derivative of `u` is 0. We will assume a constant pressure, and then analyze the steady-states of `u`. We will again ignore any mass budget, `b`, and recall the gradient of a constant function (here, pressure) is 0. Recall our formula:

```math
\frac{\partial \textbf{u}^\flat}{\partial t} + \pounds_u \textbf{u}^\flat - \frac{1}{2} \textbf{d}(\textbf{u}^\flat(\textbf{u})) = - \frac{1}{\rho} \textbf{d} p + \textbf{b}^\flat.
```

Setting appropriate terms to 0, we have:

```math
\pounds_u \textbf{u}^\flat - \frac{1}{2} \textbf{d}(\textbf{u}^\flat(\textbf{u})) = 0.
```

We already allocated our discrete differential operators. Let us solve.

```@setup euler
println("Solving")
```

```@example euler
using NLsolve

steady_flow(u) = ℒ1(u,u) - 0.5*d0*ι1(u,u)

starting_state = s1 * eval_constant_primal_form(sd, X♯)
sol = nlsolve(steady_flow, starting_state)

plot_dvf(sd, sol.zero, title="Steady State")
```

```@setup euler
println("Solved")
```

We note that this steady flow of all zero-vectors does indeed satisfy the constraints that we set.

## Third Case

For this third case, we will again solve for `u`. However, we will set a Gaussian bubble of pressure at the center of the domain, and use Euler's method to solve Euler's equations.

```math
\frac{\partial \textbf{u}^\flat}{\partial t} = - \pounds_u \textbf{u}^\flat + \frac{1}{2} \textbf{d}(\textbf{u}^\flat(\textbf{u})) - \frac{1}{\rho} \textbf{d} p.
```

### Case 3.1: Euler's method

```@example euler
center = [50.0, 50.0, 0.0]
gauss(pnt) = 2 + 50/(√(2*π*10))*ℯ^(-(norm(center-pnt)^2)/(2*10))
p = gauss.(sd[sd[:tri_center], :dual_point])

u = s1 * eval_constant_primal_form(sd, X♯)
du = copy(u)

function euler_equation!(du,u,p)
  du .= - ℒ1(u,u) + 0.5*d0*ι1(u,u) - d0*p
end

dt = 1e-3
function eulers_method()
  for _ in 0:dt:1
    euler_equation!(du,u,p)
    u .+= du * dt
  end
  u
end

eulers_method()

plot_dvf(sd, u, title="Flow")
```

### Case 3.2: Euler's method with Projection

In Case 3.1, we solved Euler's equation directly using the method of lines. However, we assume that our flow, `u`, is incompressible. That is, ``\delta u = 0``. In our finite updates, we did not check that the self-advection term is divergence free! One way to resolve this discrepancy is the "Projection method", and this is intimately related to the Hodge decomposition of the flow. (See the [Wikipedia entry](https://en.wikipedia.org/wiki/Projection_method_(fluid_dynamics)) on the projection method, for example.) Let's employ this method here.

```@example euler
u = s1 * eval_constant_primal_form(sd, X♯)
du = copy(u)

dt = 1e-3
u_int = zeros(ne(sd))
p_next = zeros(ntriangles(sd))

function euler_equation_with_projection!(u)
  u_int .= u .+ (- ℒ1(u,u) + 0.5*d0*ι1(u,u))*dt
  p_next .= (solveΔ ∘ div)(u_int/dt)
  u .= u_int - dt*(d0*p_next)
end

function eulers_method()
  for _ in 0:dt:1
    euler_equation_with_projection!(u)
  end
  u
end

eulers_method()

plot_dvf(sd, u, title="Flow, with Projection Method")
```

