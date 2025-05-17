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

let # Test the effect of mesh optimization on simple heat equation dynamics.
#=
function dome_comparison(sim)
  s, npi, spi = makeSphere(0, 90, 20, 0, 360, 20, 1);
  orient!(s);
  s[:edge_orientation] = false;
  s_orig = copy(s);
  ∂₂ = ∂(2,s)
  eqs = optimize_mesh!(s, SimulatedAnnealing(ϵ=1e-3, epochs=200, jitter3D=true, spherical=true))

  #s = loadmesh(Icosphere(3))
  #s[:edge_orientation] = false;
  #s_orig = copy(s);
  #∂₂ = ∂(2,s)
  #eqs = optimize_mesh!(s, SimulatedAnnealing(ϵ=1e-3, epochs=100, anneal=false, jitter3D=true, spherical=true))

  tₑ = 1e5
  function solve_sim(msh)
    sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}(msh)
    #subdivide_duals!(sd, Circumcenter())
    subdivide_duals!(sd, Barycenter())

    # TODO: Simulate the heat equation (via the midpoint method).
    #f = sim(sd, nothing, GeometricHodge())

    u₀ = ComponentArray(C = getindex.(point(msh), 3))
    cs_ps = (k = 1e-1,)

    prob = ODEProblem(f, u₀, (0, tₑ), cs_ps)
    soln = solve(prob, Tsit5(); progress=true, progress_steps=1);
    s0 = dec_hodge_star(0, sd)
    s0, soln
  end

  @info "Solving original mesh"
  s0_orig, soln_orig = solve_sim(s_orig)
  @info "Solving optimized mesh"
  s0, soln = solve_sim(s)

  #=
  viz(s_orig, soln_orig, "Original Error Distribution")
  viz(s, soln, "Optimized Error Distribution")
  wf(s)
  wf(s_orig)
  eqs_dist(s_orig, s)
  =#
end

int_diff = map(range(0,tₑ;length=100)) do x
  abs(sum(s0 * soln(x).C) - sum(s0 * soln(0).C))
end
int_diff_orig = map(range(0,tₑ;length=100)) do x
  abs(sum(s0_orig * soln_orig(x).C) - sum(s0_orig * soln_orig(0).C))
end
=#
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

