# Steady-State Euler Equations

CombinatorialSpaces provides meshes and discrete operators amenable for solving many types of physics and multi-physics problems. For example, CombinatorialSpaces powers the [Decapodes](https://github.com/AlgebraicJulia/Decapodes.jl) library provides a DSL for generating initial-value problem simulations, such as [co-rotating vortices on a sphere governed by Navier-Stokes](https://algebraicjulia.github.io/Decapodes.jl/dev/navier_stokes/ns/).

On this page, we will use CombinatorialSpaces directly to solve a steady-state problem.

## Euler Equations

The Euler equations are a concise model of fluid flow, but this model still demonstrates some interesting differential operators:

```math
\frac{\partial \textbf{u}^\flat}{\partial t} + \pounds_u \textbf{u}^\flat - \frac{1}{2} \textbf{d}(\textbf{u}^\flat(\textbf{u})) = - \frac{1}{\rho} \textbf{d} p + \textbf{b}^\flat.
```

Here, we see an exterior derivative, [`d`](@ref), a Lie derivative, [`ℒ`](@ref), and an interior product, [`interior_product`](@ref).

## Discretizing

Let's examine some particular cases of these equations. In this first case, we will provide a mesh, and explicitly provide initial values for `u`. We will solve for pressure and the time derivative of `u` and check that they are what we expect.

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
sharp_dd = ♯_mat(sd, LLSDDSharp())
function plot_dvf(sd, X; ls=1f0, title="Dual Vector Field")
  X♯ = sharp_dd * X
  # Makie will throw an error if the colorrange end points are not unique:
  extX = extrema(norm.(X♯))
  range = extX[1] ≈ extX[2] ? (0,extX[2]) : extX
  f = Figure()
  ax = Axis(f[1, 1], title=title)
  wireframe!(ax, sd, color=:gray95)
  scatter!(ax, getindex.(sd[sd[:tri_center], :dual_point],1), getindex.(sd[sd[:tri_center], :dual_point],2), color = norm.(X♯), colorrange=range)
  arrows!(ax, getindex.(sd[sd[:tri_center], :dual_point],1), getindex.(sd[sd[:tri_center], :dual_point],2), getindex.(X♯,1), getindex.(X♯,2), lengthscale=ls)
  Colorbar(f[1,2], limits=range)
  hidedecorations!(ax)
  f
end

function plot_dual0form(sd, f0)
  ps  = (stack(sd[sd[:tri_center], :dual_point])[[1,2],:])'
  f = Figure(); ax = CairoMakie.Axis(f[1,1]);
  sct = scatter!(ax, ps,
      color=f0);
  Colorbar(f[1,2], sct)
  f
end
```
Let's provide a flow field of unit magnitude, static throughout the domain.

```@example euler
X♯ = SVector{3,Float64}(1/√2,1/√2,0)
u = s1 * eval_constant_primal_form(sd, X♯)

plot_dvf(sd, u, title="Flow")
```

Let's look at the self-advection term, in which we take the lie derivative of `u` along itself, and subtract half of the gradient of its inner product. Recall that our flow `u` is static throughout the domain, so we should expect this term to be 0 throughout the interior of the domain, where it is not affected by boundary conditions:

```@example euler
selfadv = ℒ1(u,u) - 0.5*d0*ι1(u,u)

plot_dvf(sd, selfadv, title="Self-Advection")
```

Now, let's solve for pressure. We can set up a Poisson problem on the divergence of the self-advection term we computed. Recall that divergence can be computed as `\star d \star`, and the Laplacian as `d \star d \star`.

```@example euler
div(x) = s2 * d1 * (s1 \ x);
solveΔ(x) = float.(d0) \ (s1 * (float.(d1) \ (s2 \ x)))

p = (solveΔ ∘ div)(selfadv)
dp = d0*p

plot_dvf(sd, dp, title="Pressure Gradient")
```

Based on our initial conditions and the way that we computed pressure, we expect that the time derivative of `u` should be 0 on the interior of the domain, where it is not affected by boundary conditions.

```@example euler
∂ₜu = -selfadv - dp;

plot_dvf(sd, ∂ₜu, title="Time Derivative")
```
