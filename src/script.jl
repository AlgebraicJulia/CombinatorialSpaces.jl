using CombinatorialSpaces
using GeometryBasics
using Test
using CairoMakie
using StaticArrays
using LinearAlgebra
import CombinatorialSpaces.DiscreteExteriorCalculus: eval_constant_primal_form

#TODO: Remove me before merging

s=loadmesh(Rectangle_30x10())
# s = triangulated_grid(128,128,1,1,Point3{Float64});
sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3{Float64}}(s);
subdivide_duals!(sd, Circumcenter());

function my_form(s)
  EForm(map(edges(s)) do e
          dot(SVector{3}(point(sd)[src(s,e)][1]^2, 0, 0), point(s, tgt(s,e)) - point(s, src(s,e))) * sign(1,s,e)
        end)
end

alt_ex1 = my_form(sd)

test_form = map(p->-2*p[1],sd[:point])
f = Figure();
ax = CairoMakie.Axis(f[1,1]; title = "Exact")
msh = mesh!(ax, s, color=test_form, colormap=:jet)
Colorbar(f[1,2], msh)
f

ex1 = map(edges(s)) do e
  1/3 * ((point(s, tgt(s,e))[1])^3 - (point(s, src(s,e))[1])^3)
end

d1 = dec_differential(1,sd)
mat = p2_d2_interpolation(sd)
inv_hdg_0 = dec_inv_hodge_star(0, sd)
res = inv_hdg_0 * mat * d1 * ex1;

f = Figure();
ax = CairoMakie.Axis(f[1,1]; title = "Interpolated")
msh = mesh!(ax, s, color=res, colormap=:jet)
Colorbar(f[1,2], msh)
f

diff = res - test_form

f = Figure();
ax = CairoMakie.Axis(f[1,1]; title = "Difference")
msh = mesh!(ax, s, color=diff, colormap=:jet)
Colorbar(f[1,2], msh)
f
