module TestMeshes

using Catlab, Catlab.CategoricalAlgebra
using CombinatorialSpaces
using Test

magnitude = (sqrt ∘ (x -> foldl(+, x*x)))
unit_radius = 1
euler_characteristic(p) = nv(p) - ne(p) + nparts(p, :Tri)

# Unit Icospheres are of the proper dimensions, are spheres, and have the right
# Euler characteristic.
unit_icosphere1 = loadmesh(Icosphere(1))
@test nv(unit_icosphere1) == 12
@test ne(unit_icosphere1) == 30
@test nparts(unit_icosphere1, :Tri) == 20
ρ = magnitude(unit_icosphere1[:point][begin])
@test all(isapprox.(magnitude.(unit_icosphere1[:point]), ρ))
@test ρ == unit_radius
@test euler_characteristic(unit_icosphere1) == 2

unit_icosphere2 = loadmesh(Icosphere(2))
@test nv(unit_icosphere2) == 42
@test ne(unit_icosphere2) == 120
@test nparts(unit_icosphere2, :Tri) == 80
ρ = magnitude(unit_icosphere2[:point][begin])
@test all(isapprox.(magnitude.(unit_icosphere2[:point]), ρ))
@test ρ == unit_radius
@test euler_characteristic(unit_icosphere2) == 2

unit_icosphere3 = loadmesh(Icosphere(3))
@test nv(unit_icosphere3) == 162
@test ne(unit_icosphere3) == 480
@test nparts(unit_icosphere3, :Tri) == 320
ρ = magnitude(unit_icosphere3[:point][begin])
@test all(isapprox.(magnitude.(unit_icosphere3[:point]), ρ))
@test ρ == unit_radius
@test euler_characteristic(unit_icosphere3) == 2

unit_icosphere4 = loadmesh(Icosphere(4))
@test nv(unit_icosphere4) == 642
@test ne(unit_icosphere4) == 1920
@test nparts(unit_icosphere4, :Tri) == 1280
ρ = magnitude(unit_icosphere4[:point][begin])
@test all(isapprox.(magnitude.(unit_icosphere4[:point]), ρ))
@test ρ == unit_radius
@test euler_characteristic(unit_icosphere4) == 2

unit_icosphere5 = loadmesh(Icosphere(5))
@test nv(unit_icosphere5) == 2562
@test ne(unit_icosphere5) == 7680
@test nparts(unit_icosphere5, :Tri) == 5120
ρ = magnitude(unit_icosphere5[:point][begin])
@test all(isapprox.(magnitude.(unit_icosphere5[:point]), ρ))
@test ρ == unit_radius
@test euler_characteristic(unit_icosphere5) == 2

unit_icosphere6 = loadmesh(Icosphere(6))
@test nv(unit_icosphere6) == 10242
@test ne(unit_icosphere6) == 30720
@test nparts(unit_icosphere6, :Tri) == 20480
ρ = magnitude(unit_icosphere6[:point][begin])
@test all(isapprox.(magnitude.(unit_icosphere6[:point]), ρ))
@test ρ == unit_radius
@test euler_characteristic(unit_icosphere6) == 2

unit_icosphere7 = loadmesh(Icosphere(7))
@test nv(unit_icosphere7) == 40962
@test ne(unit_icosphere7) == 122880
@test nparts(unit_icosphere7, :Tri) == 81920
ρ = magnitude(unit_icosphere7[:point][begin])
@test all(isapprox.(magnitude.(unit_icosphere7[:point]), ρ))
@test ρ == unit_radius
@test euler_characteristic(unit_icosphere7) == 2

unit_icosphere8 = loadmesh(Icosphere(8))
@test nv(unit_icosphere8) == 163842
@test ne(unit_icosphere8) == 491520
@test nparts(unit_icosphere8, :Tri) == 327680
ρ = magnitude(unit_icosphere8[:point][begin])
@test all(isapprox.(magnitude.(unit_icosphere8[:point]), ρ))
@test ρ == unit_radius
@test euler_characteristic(unit_icosphere8) == 2

# Testing the radius parameter.
thermosphere_radius = 6371 + 90

thermo_icosphere5 = loadmesh(Icosphere(5, thermosphere_radius))
@test nv(thermo_icosphere5) == 2562
@test ne(thermo_icosphere5) == 7680
@test nparts(thermo_icosphere5, :Tri) == 5120
ρ = magnitude(thermo_icosphere5[:point][begin])
@test all(isapprox.(magnitude.(thermo_icosphere5[:point]), ρ))
@test ρ == thermosphere_radius
@test euler_characteristic(thermo_icosphere5) == 2

# Tests for the SphericalMeshes
ρ = 6371+90
s, npi, spi = makeSphere(0, 180, 5, 0, 360, 5, ρ)
magnitude = (sqrt ∘ (x -> foldl(+, x*x)))

# The definition of a discretization of a sphere of unspecified radius.
ρ′ = magnitude(s[:point][begin])
@test all(isapprox.(magnitude.(s[:point]), ρ′))

# The definition of a discretization of a sphere of radius ρ.
@test all(isapprox.(magnitude.(s[:point]), ρ))

# Some properties of a regular octahedron.
◀▶, npi, spi = makeSphere(0, 180, 90, 0, 360, 90, 1)
@test nv(◀▶) == 6
@test ne(◀▶) == 12
@test length(triangles(◀▶)) == 8

end
