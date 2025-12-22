# Steady-State Euler Equations

CombinatorialSpaces provides meshes and discrete operators amenable for solving many types of physics and multi-physics problems. For example, CombinatorialSpaces powers the [Decapodes](https://github.com/AlgebraicJulia/Decapodes.jl) library provides a DSL for generating initial-value problem simulations, such as [co-rotating vortices on a sphere governed by Navier-Stokes](https://algebraicjulia.github.io/Decapodes.jl/dev/navier_stokes/ns/).

On this page, we will use CombinatorialSpaces directly to solve a steady-state problem.

## Euler Equations

The Euler equations are a concise model of fluid flow, but this model still demonstrates some interesting differential operators:

```math
\frac{\partial \textbf{u}^\flat}{\partial t} + \pounds_u \textbf{u}^\flat - \frac{1}{2} \textbf{d}(\textbf{u}^\flat(\textbf{u})) = - \frac{1}{\rho} \textbf{d} p + \textbf{b}^\flat.
```

See Marsden, Ratiu, and Abraham's "Manifolds, Tensor Analysis, and Applications" for an overview in the exterior calculus.

Here, we see an exterior derivative, [`d`](@ref), a Lie derivative, [`ℒ`](@ref), and an interior product, [`interior_product`](@ref).

## Discretizing

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

sharp_pd = ♯_mat(sd, AltPPSharp())
function plot_vf(sd, X; ls=1f0, title="Primal Vector Field")
  X♯ = sharp_pd * X
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

