module TestMeshOptimization

using CombinatorialSpaces
using Random
using Statistics
using Test
Random.seed!(0)

let # Test the symmetric mirrored mesh, useful for porous convection.
s = binary_subdivision(mirrored_mesh(40.0, 20.0));
orient!(s);
s[:edge_orientation] = false;
eqs = optimize_mesh!(s, SimulatedAnnealing(ϵ=1e-3, epochs=5000))

@test eqs[end] < eqs[begin]
@test all(<(1e-5), abs.(diff(eqs)))
end

let # Test a refinement of the porous convection mesh.
s = binary_subdivision(binary_subdivision(mirrored_mesh(40.0, 20.0)));
orient!(s);
s[:edge_orientation] = false;
eqs = optimize_mesh!(s, SimulatedAnnealing(ϵ=1e-3, epochs=200))

@test eqs[end] < eqs[begin]
@test all(<(1e-4), abs.(diff(eqs)))
end

let # Test the default triangulated_grid by moving points along z.
s = triangulated_grid(40,20,5,5,Point3d);
eqs = optimize_mesh!(s, SimulatedAnnealing(ϵ=1e-2, epochs=500, jitter3D=true))

@test eqs[end] < eqs[begin]
@test all(<(1e-5), abs.(diff(eqs)))
end

let # Test the triangulated_grid explicitly using 2D points.
s = triangulated_grid(40,20,5,5,Point2d);
eqs = optimize_mesh!(s, SimulatedAnnealing(ϵ=1e-2, epochs=500, hold_boundaries=false))

@test eqs[end] < eqs[begin]
# The bulk of the change in equilaterality is performed in the first 250 epochs.
@test sum(eqs[251:500]) < (sum(eqs[1:250]) / 100)
end

let # Test optimizing the UV sphere, (typically bad for DEC).
s, npi, spi = makeSphere(0, 90, 20, 0, 360, 20, 1);
orient!(s);
s[:edge_orientation] = false;
eqs = optimize_mesh!(s,
  SimulatedAnnealing(ϵ=1e-3, epochs=200, anneal=false, jitter3D=true, spherical=true))

@test eqs[end] < eqs[begin]
@test all(<(1e-3), abs.(diff(eqs)))
# The rate of improvement tends to decrease:
@test 0 < mean(diff(diff(eqs)))
end

let # Test optimizing a ribbon of UV-points.
s, npi, spi = makeSphere(70, 110, 10, 0, 360, 20, 1);
orient!(s);
s[:edge_orientation] = false;
eqs = optimize_mesh!(s,
  SimulatedAnnealing(ϵ=1e-3, epochs=200, jitter3D=true, spherical=true))

@test eqs[end] < eqs[begin]
@test all(<(1e-3), abs.(diff(eqs)))
end

let # Test optimizing the icosphere, which is mostly equilateral.
s = loadmesh(Icosphere(3));
s[:edge_orientation] = false;
eqs = optimize_mesh!(s,
  SimulatedAnnealing(ϵ=1e-3, epochs=25, anneal=false, jitter3D=true, spherical=true))

@test eqs[end] < eqs[begin]
@test all(<(1e-3), abs.(diff(eqs)))
# The rate of improvement tends to decrease:
@test 0 < mean(diff(diff(eqs)))
end

#############################
# Plots for debugging tests #
#############################
#=
function wf(s_orig, s)
  f = Figure()
  ax = CairoMakie.Axis(f[1,1],
    xlabel="x-coordinate [m]",
    ylabel="y-coordinate [m]")
  wireframe!(ax, s_orig)
  wireframe!(ax, s)
  f
end

function wf(s)
  f = Figure()
  ax = CairoMakie.Axis(f[1,1],
    xlabel="x-coordinate [m]",
    ylabel="y-coordinate [m]")
  wireframe!(ax, s)
  f
end

function eqs_dist(s_orig, s)
  f = Figure()
  ax = CairoMakie.Axis(f[1,1])
  hist!(ax, e_eqs(e_lens(s, ∂₂)))
  hist!(ax, e_eqs(e_lens(s_orig, ∂₂)))
  f
end

function compare_integrals()
  f = Figure();
  ax = CairoMakie.Axis(f[1,1],
    title="Error in integral over dome surface",
    xlabel="Time [s]",
    ylabel="Error");
  lines!(ax, range(0,tₑ;length=100), map(range(0,tₑ;length=100)) do x
    abs(sum(s0 * soln(x).C) - sum(s0 * soln(0).C))
  end,
    label="Original")
  lines!(ax, range(0,tₑ;length=100), map(range(0,tₑ;length=100)) do x
    abs(sum(s0_orig * soln_orig(x).C) - sum(s0_orig * soln_orig(0).C))
  end,
    label="Optimized")
  axislegend(ax, position=:lt)
  f
end

function viz(msh, msh_soln, msh_title)
  f = Figure()
  ax = CairoMakie.LScene(f[1, 1], scenekw=(lights=[],))
  mesh!(ax, msh, color=msh_soln(tₑ).C)
  f
end

function viz(s)
  sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}(msh)
  subdivide_duals!(sd, Circumcenter())
  s0 = dec_hodge_star(0, sd)
  f = Figure()
  ax = CairoMakie.LScene(f[1, 1], scenekw=(lights=[],))
  scatter!(ax, sd[sd[:tri_center], :dual_point], color=e_eqs(e_lens(s, ∂₂)))
  f
end
=#

end # module TestMeshOptimization

