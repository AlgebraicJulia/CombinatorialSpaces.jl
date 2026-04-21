using Test
# using CairoMakie

include("../../src/CubicalCode/UniformMesh.jl")

@testset "UniformCubicalComplex2D - No Halo" begin

  # Create a uniform grid with no halo points, 5x5 points, and a domain of 10x10
  s = UniformCubicalComplex2D(6, 6, 10.0, 10.0)

  # Test basic properties
  @test nx(s) == 6
  @test ny(s) == 6
  @test dx(s) == 10.0 / 5
  @test dy(s) == 10.0 / 5

  # Test point generation
  p = point(s, 1, 1)
  @test p == Point3d(0.0, 0.0, 0.0)
  p = point(s, 6, 6)
  @test p == Point3d(10.0, 10.0, 0.0)

  # Test counts (assuming same as EmbeddedCubicalComplex2D)
  @test nv(s) == 36
  @test ne(s) == 60
  @test nquads(s) == 25

  # Test edge source and target
  @test src(s, 1, 1, X_ALIGN) == 1
  @test tgt(s, 1, 1, X_ALIGN) == 2
  @test src(s, 1, 1, Y_ALIGN) == 1
  @test tgt(s, 1, 1, Y_ALIGN) == 7

  # Test quad vertices
  @test coord_to_quad(s, 1, 1) == 1
  @test coord_to_quad(s, 5, 5) == 25
  @test coord_to_quad(s, 2, 1) == 2
  @test coord_to_quad(s, 1, 2) == 6

  @test quad_vertices(s, 1, 1) == (1, 2, 8, 7)
  @test quad_vertices(s, 5, 5) == (29, 30, 36, 35)

  # Test quad to edge mapping
  @test quad_edges(s, 1, 1) == (1, 32, 6, 31)
  @test quad_edges(s, 5, 5) == (25, 60, 30, 59)

  # Test quad areas
  @test quad_area(s) == dx(s) * dy(s)

  # Test dual points
  dp = dual_point(s, 1, 1)
  @test dp == Point3d(1.0, 1.0, 0.0)

  dp = dual_point(s, 5, 5)
  @test dp == Point3d(9.0, 9.0, 0.0)

  # Test dual edge lengths
  @test dual_edge_len(s, 1, 1, X_ALIGN) == dy(s) / 2
  @test dual_edge_len(s, 1, 1, Y_ALIGN) == dx(s) / 2

  @test dual_edge_len(s, 1, 2, X_ALIGN) == dy(s)
  @test dual_edge_len(s, 2, 1, Y_ALIGN) == dx(s)

  @test dual_edge_len(s, 1, 6, X_ALIGN) == dy(s) / 2
  @test dual_edge_len(s, 6, 1, Y_ALIGN) == dx(s) / 2

  # Test dual quad areas
  @test dual_quad_area(s, 1, 1) == dx(s) * dy(s) / 4
  @test dual_quad_area(s, 1, 2) == dx(s) * dy(s) / 2
  @test dual_quad_area(s, 2, 1) == dx(s) * dy(s) / 2
  @test dual_quad_area(s, 2, 2) == dx(s) * dy(s)
end

# # Test plotting (if supported)
# fig = Figure()
# ax = CairoMakie.Axis(fig[1,1])
# wireframe!(ax, s)
# save("imgs/UniformGrid.png", fig)

@testset "UniformCubicalComplex2D - With Halo" begin

  # Create a uniform grid with halo points, 5x5 real points, and a domain of 10x10
  s = UniformCubicalComplex2D(6, 6, 10.0, 10.0, halo_x=1, halo_y=1)

  # Get the total number of points, which should be (6 + 2) * (6 + 2) = 64
  @test nv(s) == 64
  @test nxr(s) == 6
  @test nyr(s) == 6

  @test dx(s) == 10.0 / 5
  @test dy(s) == 10.0 / 5

  @test s.halo_x == 1
  @test s.halo_y == 1

  # Test point generation with halo points
  p = point(s, 1, 1)
  @test p == Point3d(-2.0, -2.0, 0.0)
  p = point(s, 8, 8)
  @test p == Point3d(12.0, 12.0, 0.0)

  # Test point generation in the interior (should be same as before)
  p = point(s, 2, 2)
  @test p == Point3d(0.0, 0.0, 0.0)
  p = point(s, 7, 7)
  @test p == Point3d(10.0, 10.0, 0.0)

  # Test counts (assuming same as EmbeddedCubicalComplex2D)
  @test nv(s) == 64

  @test nxedges(s) == 56
  @test nyedges(s) == 56
  @test ne(s) == 112

  @test nquads(s) == 49

  # Test edge source and target
  @test src(s, 1, 1, X_ALIGN) == 1
  @test tgt(s, 1, 1, X_ALIGN) == 2
  @test src(s, 1, 1, Y_ALIGN) == 1
  @test tgt(s, 1, 1, Y_ALIGN) == 9

  # Test quad vertices
  @test coord_to_quad(s, 1, 1) == 1
  @test coord_to_quad(s, 7, 7) == 49

  @test quad_vertices(s, 1, 1) == (1, 2, 10, 9)
  @test quad_vertices(s, 7, 7) == (55, 56, 64, 63)

  # Test quad to edge mapping
  @test quad_edges(s, 1, 1) == (1, 58, 8, 57)
  @test quad_edges(s, 7, 7) == (49, 112, 56, 111)

  # Test quad areas
  @test quad_area(s) == dx(s) * dy(s)

  # Test dual points with halo points
  dp = dual_point(s, 1, 1)
  @test dp == Point3d(-1.0, -1.0, 0.0)

  dp = dual_point(s, 7, 7)
  @test dp == Point3d(11.0, 11.0, 0.0)

  # Test dual points in the interior (should be same as before)
  dp = real_dual_point(s, 1, 1)
  @test dp == Point3d(1.0, 1.0, 0.0)

  dp = real_dual_point(s, 5, 5)
  @test dp == Point3d(9.0, 9.0, 0.0)
end

@testset "Tiny Mesh" begin
  s = UniformCubicalComplex2D(2, 2, 1.0, 1.0)

  @test nv(s) == 4
  @test ne(s) == 4

  @test nquads(s) == 1
  @test all(is_boundary_vert.(Ref(s), [1, 1, 2, 2], [1, 2, 1, 2]))
  @test all(is_boundary_edge.(Ref(s), [1, 1, 1, 2], [1, 2, 1, 1], [X_ALIGN, X_ALIGN, Y_ALIGN, Y_ALIGN]))

  @test is_left_edge(s, 1, 1, Y_ALIGN) == true
  @test is_right_edge(s, 1, 1, Y_ALIGN) == false

  @test is_left_edge(s, 2, 1, Y_ALIGN) == false
  @test is_right_edge(s, 2, 1, Y_ALIGN) == true

  @test is_top_edge(s, 1, 1, X_ALIGN) == false
  @test is_bottom_edge(s, 1, 1, X_ALIGN) == true

  @test is_top_edge(s, 1, 2, X_ALIGN) == true
  @test is_bottom_edge(s, 1, 2, X_ALIGN) == false
end
