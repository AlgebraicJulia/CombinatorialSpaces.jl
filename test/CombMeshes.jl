module TestMeshes

using Catlab, Catlab.CategoricalAlgebra
using CombinatorialSpaces
using Test

magnitude = (sqrt ∘ (x -> foldl(+, x*x)))
unit_radius = 1
euler_characteristic(p) = nv(p) - ne(p) + nparts(p, :Tri)
euler_characteristic(p::HasDeltaSet3D) = nv(p) - ne(p) + nparts(p, :Tri) - nparts(p, :Tet)

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

# Testing the Triangulated Grid
function test_trigrid_size(s, max_x, max_y, dx, dy)
  nx = length(0:dx:max_x)
  ny = length(0:dy:max_y)

  @test nparts(s, :V) == nx * ny
  @test nparts(s, :Tri) == 2 * (nx - 1) * (ny - 1)
end

s = triangulated_grid(1, 1, 1, 1, Point2{Float64}, false)
test_trigrid_size(s, 1, 1, 1, 1)
@test s[:point] == [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]]
@test s[:tri_orientation] == [true, false]
@test orient!(s)

s = triangulated_grid(1, 1, 1, 1, Point2{Float64}, true)
test_trigrid_size(s, 1, 1, 1, 1)
@test s[:point] == [[0.0, 0.0], [2/3, 0.0], [1/3, 1.0], [1.0, 1.0]]
@test s[:tri_orientation] == [true, false]
@test orient!(s)

s = triangulated_grid(2, 2, 1, 1, Point2{Float64}, false)
test_trigrid_size(s, 2, 2, 1, 1)
@test s[:point] == [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                    [0.5, 1.0], [1.5, 1.0], [2.5, 1.0],
                    [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]]
@test s[:tri_orientation] == [true, false, true, false, false, true, false, true]
@test orient!(s)

s = triangulated_grid(1, 1, 0.49, 0.78, Point2{Float64}, false)
test_trigrid_size(s, 1, 1, 0.49, 0.78)
@test orient!(s)

s = triangulated_grid(1.6, 3.76, 0.49, 0.78, Point2{Float64}, true)
test_trigrid_size(s, 1.6, 3.76, 0.49, 0.78)
@test orient!(s)

lx = 25
s = triangulated_grid(lx, 2, 1, 0.99, Point2{Float64}, true)
test_trigrid_size(s, lx, 2, 1, 0.99)
@test maximum(getindex.(s[:point], 1)) == lx

@test triangulated_grid(10, 10, 1, 1, Point3d) == triangulated_grid(10, 10; nx=10, ny=10)
@test triangulated_grid(10, 10, 1, 1, Point3d, false) == triangulated_grid(10, 10; nx=10, ny=10, compress = false)
@test triangulated_grid(10, 10, 1, 1, Point2d) == triangulated_grid(10, 10; nx=10, ny=10, point_type = Point2d)
@test triangulated_grid(10, 10, 1, 1, Point3d) == triangulated_grid(10, 10; dx=1, dy=1, point_type = Point3d)
@test triangulated_grid(10, 10, 1, 1, Point3d) == triangulated_grid(10, 10; nx=10, dy=1, point_type = Point3d)

nx = 15; ny = 20
@test ntriangles(triangulated_grid(10, 10; nx = nx, ny = ny)) == 2*nx*ny
@test ntriangles(triangulated_grid(1, 5; nx = nx, ny = ny)) == 2*nx*ny

@test_throws AssertionError triangulated_grid(-10, -10; nx = nx, ny = ny)
@test_throws AssertionError triangulated_grid(10, 10; nx = -nx, ny = -ny)
@test_throws AssertionError triangulated_grid(10, 10; dx = -1, dy = -1)

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

# Testing the Parallelepiped

s = parallelepiped()
@test orient!(s)
# Max 1-norm is 3 for unit cube
@test maximum(map(p -> sum(p), s[:point])) == 3
@test euler_characteristic(s) == 1
@test sum(map(tet -> volume(3, s, tet), 1:ntetrahedra(s))) ≈ 1

s = parallelepiped(lx = 10, ly = 5, lz = 3, dx = 3, dy = 5)
@test orient!(s)
@test maximum(map(p -> sum(p), s[:point])) == (10+5+3) + (3+5)
@test euler_characteristic(s) == 1
@test sum(map(tet -> volume(3, s, tet), 1:ntetrahedra(s))) ≈ 10*5*3

end
