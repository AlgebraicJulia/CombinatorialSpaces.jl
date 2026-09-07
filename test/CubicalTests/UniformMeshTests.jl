module TestUniformMesh

using Test
using CombinatorialSpaces

@testset "No Halo" begin

    # 5x5 real points, 10x10 domain, no halo
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

    # Test counts
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

@testset "With Halo" begin

    # 5x5 real points, 10x10 domain, halo=1
    s = UniformCubicalComplex2D(6, 6, 10.0, 10.0; halo_x = 1, halo_y = 1)

    # total points = (6+2)*(6+2) = 64
    @test nv(s) == 64
    @test nxr(s) == 6
    @test nyr(s) == 6

    @test dx(s) == 10.0 / 5
    @test dy(s) == 10.0 / 5

    @test halo_west(s) == 1
    @test halo_south(s) == 1

    @test halo_east(s) == 1
    @test halo_north(s) == 1

    # Test point generation with halo points
    p = point(s, 1, 1)
    @test p == Point3d(-2.0, -2.0, 0.0)
    p = point(s, 8, 8)
    @test p == Point3d(12.0, 12.0, 0.0)

    # interior points match the no-halo case
    p = point(s, 2, 2)
    @test p == Point3d(0.0, 0.0, 0.0)
    p = point(s, 7, 7)
    @test p == Point3d(10.0, 10.0, 0.0)

    # Test counts
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

    # interior dual points match the no-halo case
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

# TODO: Verify these expectations are correct. They pin the offset helpers to
# src/tgt and quad_edges rather than to hardcoded indices, but the intended
# meaning of "offset" was inferred from the implementation and its comments.
@testset "Edge Vertex and Quad Edge Offsets" begin
    for s in (UniformCubicalComplex2D(4, 5, 1.0, 1.0),
              UniformCubicalComplex2D(4, 5, 1.0, 1.0; halo_x = 1, halo_y = 1))

        # An edge's two endpoints: offset 0 is the source, offset 1 the target.
        for e in edges(s)
            x, y, align = edge_to_coord(s, e)
            @test edge_vertex_offset(s, x, y, align, 0) == src(s, x, y, align)
            @test edge_vertex_offset(s, x, y, align, 1) == tgt(s, x, y, align)
        end

        # quad_edges returns (bottom, right, top, left); offset 0 selects the
        # bottom/left edge and offset 1 the top/right one.
        for q in quads(s)
            x, y = quad_to_coord(s, q)
            bottom, right, top, left = quad_edges(s, x, y)

            @test quad_edge_offset(s, x, y, X_ALIGN, 0) == bottom
            @test quad_edge_offset(s, x, y, X_ALIGN, 1) == top
            @test quad_edge_offset(s, x, y, Y_ALIGN, 0) == left
            @test quad_edge_offset(s, x, y, Y_ALIGN, 1) == right

            # The offset never changes the alignment of the edge returned.
            @test is_edge_X_aligned(quad_edge_offset(s, x, y, X_ALIGN, 0), s)
            @test is_edge_X_aligned(quad_edge_offset(s, x, y, X_ALIGN, 1), s)
            @test is_edge_Y_aligned(quad_edge_offset(s, x, y, Y_ALIGN, 0), s)
            @test is_edge_Y_aligned(quad_edge_offset(s, x, y, Y_ALIGN, 1), s)
        end

        @test_throws ArgumentError quad_edge_offset(s, 1, 1, Z_ALIGN, 0)
    end
end

@testset "Pseudo Mesh Element Counting" begin
    s = PseudoCubicalMesh2D(10, 8)
    s_h = PseudoCubicalMesh2D(10, 8; halo_x = 2, halo_y = 3)

    # Real counts
    @test nxr(s) == 10
    @test nyr(s) == 8
    @test nxr(s_h) == 10
    @test nyr(s_h) == 8

    # Halo accessors
    @test halo_west(s) == 0
    @test halo_south(s) == 0
    @test halo_west(s_h) == 2
    @test halo_south(s_h) == 3

    # Total (halo-inclusive) counts
    @test nx(s) == 10
    @test ny(s) == 8
    @test nx(s_h) == 14
    @test ny(s_h) == 14

    # Vertex counts
    @test nv(s) == 10 * 8
    @test nvr(s) == 10 * 8
    @test nv(s_h) == 14 * 14
    @test nvr(s_h) == 10 * 8

    # Edge counts
    @test nxedges(s) == 9 * 8
    @test nyedges(s) == 10 * 7
    @test ne(s) == nxedges(s) + nyedges(s)

    @test nxedges(s_h) == 13 * 14
    @test nyedges(s_h) == 14 * 13
    @test ne(s_h) == nxedges(s_h) + nyedges(s_h)

    # Quad counts
    @test nxq(s) == 9
    @test nyq(s) == 7
    @test nquads(s) == 9 * 7

    @test nxq(s_h) == 13
    @test nyq(s_h) == 13
    @test nquads(s_h) == 13 * 13

    # Real quad counts
    @test nxqr(s) == 9
    @test nyqr(s) == 7
    @test nquadsr(s) == 9 * 7

    @test nxqr(s_h) == 9
    @test nyqr(s_h) == 7
    @test nquadsr(s_h) == 9 * 7

    # Indexing
    @test coord_to_vert(s, 1, 1) == 1
    @test coord_to_vert(s, 10, 8) == nv(s)

    @test coord_to_quad(s, 1, 1) == 1
    @test coord_to_quad(s, 9, 7) == nquads(s)

    @test coord_to_edge(s, 1, 1, X_ALIGN) == 1
    @test coord_to_edge(s, 1, 1, Y_ALIGN) == nxedges(s) + 1

    # Halo flags
    @test is_halo_vert(s_h, 1, 5) == true
    @test is_halo_vert(s_h, 3, 5) == false
    @test is_halo_quad(s_h, 2, 4) == true
    @test is_halo_quad(s_h, 3, 4) == false
end

@testset "Show" begin
    s = UniformCubicalComplex2D(6, 6, 10.0, 10.0)
    s_h = UniformCubicalComplex2D(6, 6, 10.0, 10.0; halo_x = 1, halo_y = 1)
    @test isnothing(show(IOBuffer(), s))
    @test isnothing(show(IOBuffer(), s_h))
end

end
