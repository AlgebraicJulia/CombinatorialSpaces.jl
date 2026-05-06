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

let # Test that BoltzmannAcceptance can be toggled via multiple dispatch.
# BoltzmannAcceptance uses exp(-(new_cost - orig_cost) / temperature), so the
# magnitude of the cost increase and the current temperature both influence
# the acceptance of a worse solution.
# Use exponential_cooling_schedule so temperature is on the same scale as cost diffs.
s = triangulated_grid(40,20,5,5,Point2d);
eqs = optimize_mesh!(s,
  SimulatedAnnealing(ϵ=1e-2, epochs=100, hold_boundaries=false,
    acceptance=BoltzmannAcceptance(),
    cooling_schedule=(e,ep)->exponential_cooling_schedule(e,ep)))
# BoltzmannAcceptance still runs and returns cost values per epoch.
@test length(eqs) == 100
# With a calibrated schedule, BoltzmannAcceptance should improve the mesh.
@test eqs[end] < eqs[begin]
end

let # Test that DirectAcceptance and BoltzmannAcceptance achieve comparable results
# when BoltzmannAcceptance uses its optimal exponential cooling schedule.
Random.seed!(42)
s_d = triangulated_grid(40,20,5,5,Point3d);
eqs_d = optimize_mesh!(s_d,
  SimulatedAnnealing(ϵ=1e-2, epochs=200, jitter3D=true,
    acceptance=DirectAcceptance()))
Random.seed!(42)
s_b = triangulated_grid(40,20,5,5,Point3d);
eqs_b = optimize_mesh!(s_b,
  SimulatedAnnealing(ϵ=1e-2, epochs=200, jitter3D=true,
    acceptance=BoltzmannAcceptance(),
    cooling_schedule=(e,ep)->exponential_cooling_schedule(e,ep, T_init=1e-7, T_final=1e-10)))
# Both strategies should improve the mesh.
@test eqs_d[end] < eqs_d[begin]
@test eqs_b[end] < eqs_b[begin]
# With a calibrated schedule, the final costs should be within 10% of each other.
@test abs(eqs_d[end] - eqs_b[end]) / eqs_d[begin] < 0.1
end

let # Test that logarithmic cooling with BoltzmannAcceptance improves the mesh.
# Logarithmic cooling (c / log(1+k)) decays very slowly, providing thorough
# exploration of the solution space before settling.
Random.seed!(42)
s = triangulated_grid(40,20,5,5,Point2d);
eqs = optimize_mesh!(s,
  SimulatedAnnealing(ϵ=1e-2, epochs=100, hold_boundaries=false,
    acceptance=BoltzmannAcceptance(),
    cooling_schedule=(e,ep)->logarithmic_cooling_schedule(e,ep)))
@test length(eqs) == 100
@test eqs[end] < eqs[begin]
end

end # module TestMeshOptimization
