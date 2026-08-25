module TestUniformMesh3D

using Test
using GeometryBasics
using CombinatorialSpaces

@testset "Basic Cube" begin
    s = UniformCubicalComplex3D(5, 5, 5, 10.0, 10.0, 10.0)

    # Vertex tests
    @test nxr(s) == 5 && nyr(s) == 5 && nzr(s) == 5
    @test nx(s) == 5 && ny(s) == 5 && nz(s) == 5
    @test nv(s) == 125
    @test nvr(s) == 125
    @test vertices(s) == 1:125

    # Edge tests
    @test nxe(s) == 4 && nye(s) == 4 && nze(s) == 4
    @test nxedges(s) == 100 && nyedges(s) == 100 && nzedges(s) == 100
    @test ne(s) == 300
    @test edges(s) == 1:300

    # Quad tests
    @test nxq(s) == 4 && nyq(s) == 4 && nzq(s) == 4
    @test nxyq(s) == 16 && nxzq(s) == 16 && nyzq(s) == 16
    @test nxyquads(s) == 80 && nxzquads(s) == 80 && nyzquads(s) == 80
    @test nquads(s) == 240
    @test quads(s) == 1:240

    # Boid tests
    @test nxb(s) == 4 && nyb(s) == 4 && nzb(s) == 4
    @test nxyb(s) == 16
    @test nboids(s) == 64
    @test boids(s) == 1:64

    # Spacing and Halo tests
    @test dx(s) == 2.5 && dy(s) == 2.5 && dz(s) == 2.5
    @test halo_west(s) == 0 && halo_south(s) == 0 && halo_down(s) == 0
end

@testset "Rectangular Prism" begin
    nx_ = 2
    ny_ = 5
    nz_ = 6
    s = UniformCubicalComplex3D(nx_, ny_, nz_, 10.0, 40.0, 50.0)

    @test nv(s) == 60

    @test nxe(s) == 1
    @test nye(s) == 4
    @test nze(s) == 5

    @test nxedges(s) == 30
    @test nyedges(s) == 48
    @test nzedges(s) == 50

    @test ne(s) == 128

    @test dx(s) == dy(s) == dz(s) == 10.0

    @test coord_to_vert(s, 1, 1, 1) == 1
    @test coord_to_vert(s, 2, 1, 1) == 2
    @test coord_to_vert(s, 1, 5, 1) == 9
    @test coord_to_vert(s, 1, 1, 6) == 51
    @test coord_to_vert(s, 2, 5, 6) == nv(s)

    @test coord_to_boid(s, 1, 1, 1) == 1
    @test coord_to_boid(s, 1, 4, 1) == 4
    @test coord_to_boid(s, 1, 1, 5) == 17
    @test coord_to_boid(s, 1, 4, 5) == nboids(s)
end

@testset "Rectangular Prism with Halo" begin
        s = UniformCubicalComplex3D(2, 5, 6, 10.0, 40.0, 50.0; halo_x = 1, halo_y = 1, halo_z = 1)

    # Halo tests
    @test halo_west(s) == 1 && halo_south(s) == 1 && halo_down(s) == 1

    # Vertex tests
    @test nxr(s) == 2 && nyr(s) == 5 && nzr(s) == 6 # Real vertices
    @test nx(s) == 4 && ny(s) == 7 && nz(s) == 8 # With halo
    @test nv(s) == 4 * 7 * 8 == 224
    @test nvr(s) == 2 * 5 * 6 == 60

    # Edge tests
    @test nxe(s) == 3 && nye(s) == 6 && nze(s) == 7
    @test nxedges(s) == 3 * 7 * 8 == 168
    @test nyedges(s) == 4 * 6 * 8 == 192
    @test nzedges(s) == 4 * 7 * 7 == 196
    @test ne(s) == 168 + 192 + 196 == 556

    # Quad tests
    @test nxq(s) == 3 && nyq(s) == 6 && nzq(s) == 7
    @test nxyq(s) == 18 && nxzq(s) == 21 && nyzq(s) == 42
    @test nxyquads(s) == 18 * 8 == 144
    @test nxzquads(s) == 21 * 7 == 147
    @test nyzquads(s) == 42 * 4 == 168
    @test nquads(s) == 144 + 147 + 168 == 459

    # Boid tests
    @test nxb(s) == 3 && nyb(s) == 6 && nzb(s) == 7
    @test nboids(s) == 3 * 6 * 7 == 126

    # Spacing should be based on real dimensions
    @test dx(s) == 10.0 / (2 - 1) == 10.0
    @test dy(s) == 40.0 / (5 - 1) == 10.0
    @test dz(s) == 50.0 / (6 - 1) == 10.0
end

@testset "Reverse Coordinate Mappings" begin
    s = UniformCubicalComplex3D(3, 4, 5, 10.0, 10.0, 10.0)

    # Vertices
    # First vertex
    @test vert_to_coord(s, 1) == (1, 1, 1)
    # Last vertex: nx*ny*nz = 3*4*5 = 60
    @test vert_to_coord(s, nv(s)) == (3, 4, 5)
    # Intermediate vertex: z=2 (adds 12), y=2 (adds 3), x=2 => 1 + 12 + 3 + 1 = 17
    @test vert_to_coord(s, 17) == (2, 2, 2)

    # Boids
    # Boid dimensions: nxb=2, nyb=3, nzb=4. Total = 24
    @test boid_to_coord(s, 1) == (1, 1, 1)
    @test boid_to_coord(s, nboids(s)) == (2, 3, 4)
    # Intermediate boid: z=2 (adds 6), y=2 (adds 2), x=1 => 1 + 6 + 2 + 0 = 9
    @test boid_to_coord(s, 9) == (1, 2, 2)

    # Edges
    # X-aligned limits: nxe=2, ny=4, nz=5. Total X-edges = 40
    @test edge_to_coord(s, 1) == (1, 1, 1, X_ALIGN)
    @test edge_to_coord(s, 40) == (2, 4, 5, X_ALIGN)
    # Y-aligned limits: nx=3, nye=3, nz=5. Total Y-edges = 45. Starts at 41, ends at 85
    @test edge_to_coord(s, 41) == (1, 1, 1, Y_ALIGN)
    @test edge_to_coord(s, 85) == (3, 3, 5, Y_ALIGN)
    # Z-aligned limits: nx=3, ny=4, nze=4. Total Z-edges = 48. Starts at 86, ends at 133
    @test edge_to_coord(s, 86) == (1, 1, 1, Z_ALIGN)
    @test edge_to_coord(s, ne(s)) == (3, 4, 4, Z_ALIGN)

    # Quads
    # Z-aligned limits: nxb=2, nyb=3, nz=5. Total Z-quads = 30
    @test quad_to_coord(s, 1) == (1, 1, 1, Z_ALIGN)
    @test quad_to_coord(s, 30) == (2, 3, 5, Z_ALIGN)
    # Y-aligned limits: nxb=2, ny=4, nzb=4. Total Y-quads = 32. Starts at 31, ends at 62
    @test quad_to_coord(s, 31) == (1, 1, 1, Y_ALIGN)
    @test quad_to_coord(s, 62) == (2, 4, 4, Y_ALIGN)
    # X-aligned limits: nx=3, nyb=3, nzb=4. Total X-quads = 36. Starts at 63, ends at 98
    @test quad_to_coord(s, 63) == (1, 1, 1, X_ALIGN)
    @test quad_to_coord(s, nquads(s)) == (3, 3, 4, X_ALIGN)
end

@testset "Reverse Coordinate Mappings with Halo" begin
    # 1-cell halo in all dimensions: real (2,5,6) -> total (4,7,8), boids (3,6,7)
    s = UniformCubicalComplex3D(2, 5, 6, 10.0, 40.0, 50.0; halo_x = 1, halo_y = 1, halo_z = 1)

    # Vertices (total 4*7*8 = 224)
    @test vert_to_coord(s, 1) == (1, 1, 1)
    @test vert_to_coord(s, nv(s)) == (4, 7, 8)

    # Check intermediate index for (2, 3, 4)
    # index = (4 - 1) * (4 * 7) + (3 - 1) * 4 + 2 = 84 + 8 + 2 = 94
    @test vert_to_coord(s, 94) == (2, 3, 4)

    # Boids (total 3*6*7 = 126)
    @test boid_to_coord(s, 1) == (1, 1, 1)
    @test boid_to_coord(s, nboids(s)) == (3, 6, 7)

    # Check intermediate index for (2, 3, 4)
    # index = (4 - 1) * (3 * 6) + (3 - 1) * 3 + 2 = 54 + 6 + 2 = 62
    @test boid_to_coord(s, 62) == (2, 3, 4)

    # Edges (168 X + 192 Y + 196 Z = 556)
    # 1. X-aligned (Grid: nxe=3, ny=7, nz=8. Total: 168)
    @test edge_to_coord(s, 1) == (1, 1, 1, X_ALIGN)
    # Intermediate index for (2, 3, 4): (4-1)*(3*7) + (3-1)*3 + 2 = 63 + 6 + 2 = 71
    @test edge_to_coord(s, 71) == (2, 3, 4, X_ALIGN)
    @test edge_to_coord(s, 168) == (3, 7, 8, X_ALIGN)

    # 2. Y-aligned (Grid: nx=4, nye=6, nz=8. Total: 192. Starts at 169)
    @test edge_to_coord(s, 169) == (1, 1, 1, Y_ALIGN)
    # Intermediate index for (2, 3, 4): 168 + [(4-1)*(4*6) + (3-1)*4 + 2] = 168 + 82 = 250
    @test edge_to_coord(s, 250) == (2, 3, 4, Y_ALIGN)
    @test edge_to_coord(s, 168 + 192) == (4, 6, 8, Y_ALIGN)

    # 3. Z-aligned (Grid: nx=4, ny=7, nze=7. Total: 196. Starts at 361)
    @test edge_to_coord(s, 361) == (1, 1, 1, Z_ALIGN)
    # Intermediate index for (2, 3, 4): 360 + [(4-1)*(4*7) + (3-1)*4 + 2] = 360 + 94 = 454
    @test edge_to_coord(s, 454) == (2, 3, 4, Z_ALIGN)
    @test edge_to_coord(s, ne(s)) == (4, 7, 7, Z_ALIGN)

    # Quads (144 Z + 147 Y + 168 X = 459)
    # 1. Z-aligned (Grid: nxb=3, nyb=6, nz=8. Total: 144)
    @test quad_to_coord(s, 1) == (1, 1, 1, Z_ALIGN)
    # Intermediate index for (2, 3, 4): (4-1)*(3*6) + (3-1)*3 + 2 = 54 + 6 + 2 = 62
    @test quad_to_coord(s, 62) == (2, 3, 4, Z_ALIGN)
    @test quad_to_coord(s, 144) == (3, 6, 8, Z_ALIGN)

    # 2. Y-aligned (Grid: nxb=3, ny=7, nzb=7. Total: 147. Starts at 145)
    @test quad_to_coord(s, 145) == (1, 1, 1, Y_ALIGN)
    # Intermediate index for (2, 3, 4): 144 + [(4-1)*(3*7) + (3-1)*3 + 2] = 144 + 71 = 215
    @test quad_to_coord(s, 215) == (2, 3, 4, Y_ALIGN)
    @test quad_to_coord(s, 144 + 147) == (3, 7, 7, Y_ALIGN)

    # 3. X-aligned (Grid: nx=4, nyb=6, nzb=7. Total: 168. Starts at 292)
    @test quad_to_coord(s, 292) == (1, 1, 1, X_ALIGN)
    # Intermediate index for (2, 3, 4): 291 + [(4-1)*(4*6) + (3-1)*4 + 2] = 291 + 82 = 373
    @test quad_to_coord(s, 373) == (2, 3, 4, X_ALIGN)
    @test quad_to_coord(s, nquads(s)) == (4, 6, 7, X_ALIGN)
end

@testset "Quad Coordinate Round-Trip and Stride" begin

    # cubic grid (baseline)
    s = UniformCubicalComplex3D(3, 3, 3, 1.0, 1.0, 1.0)

    # Z_ALIGN: stride is nxb(s)=2, nxyb(s)=4
    @test coord_to_quad(s, 1, 1, 1, Z_ALIGN) == 1
    @test coord_to_quad(s, 2, 1, 1, Z_ALIGN) == 2
    @test coord_to_quad(s, 1, 2, 1, Z_ALIGN) == 3
    @test coord_to_quad(s, 1, 1, 2, Z_ALIGN) == 5
    @test quad_to_coord(s, coord_to_quad(s, 2, 2, 2, Z_ALIGN)) == (2, 2, 2, Z_ALIGN)

    # Y_ALIGN: stride is nxb(s)=2, ny(s)=3
    @test coord_to_quad(s, 1, 1, 1, Y_ALIGN) == nxyquads(s) + 1
    @test coord_to_quad(s, 2, 1, 1, Y_ALIGN) == nxyquads(s) + 2
    @test coord_to_quad(s, 1, 2, 1, Y_ALIGN) == nxyquads(s) + 3
    @test coord_to_quad(s, 1, 1, 2, Y_ALIGN) == nxyquads(s) + nxb(s) * ny(s) + 1
    @test quad_to_coord(s, coord_to_quad(s, 2, 2, 2, Y_ALIGN)) == (2, 2, 2, Y_ALIGN)

    # X_ALIGN: stride is nx(s)=3, nyb(s)=2
    @test coord_to_quad(s, 1, 1, 1, X_ALIGN) == nxyquads(s) + nxzquads(s) + 1
    @test coord_to_quad(s, 2, 1, 1, X_ALIGN) == nxyquads(s) + nxzquads(s) + 2
    @test coord_to_quad(s, 3, 1, 1, X_ALIGN) == nxyquads(s) + nxzquads(s) + 3
    @test coord_to_quad(s, 1, 2, 1, X_ALIGN) == nxyquads(s) + nxzquads(s) + nx(s) + 1
    @test coord_to_quad(s, 1, 1, 2, X_ALIGN) == nxyquads(s) + nxzquads(s) + nx(s) * nyb(s) + 1
    @test quad_to_coord(s, coord_to_quad(s, 2, 2, 2, X_ALIGN)) == (2, 2, 2, X_ALIGN)

    s2 = UniformCubicalComplex3D(3, 4, 5, 1.0, 1.0, 1.0)

    # Z_ALIGN
    @test quad_to_coord(s2, 1) == (1, 1, 1, Z_ALIGN)
    @test quad_to_coord(s2, nxyquads(s2)) == (nxb(s2), nyb(s2), nz(s2), Z_ALIGN)
    @test quad_to_coord(s2, coord_to_quad(s2, 2, 3, 4, Z_ALIGN)) == (2, 3, 4, Z_ALIGN)

    # Y_ALIGN
    @test quad_to_coord(s2, nxyquads(s2) + 1) == (1, 1, 1, Y_ALIGN)
    @test quad_to_coord(s2, nxyquads(s2) + nxzquads(s2)) == (nxb(s2), ny(s2), nzb(s2), Y_ALIGN)
    @test quad_to_coord(s2, coord_to_quad(s2, 2, 3, 4, Y_ALIGN)) == (2, 3, 4, Y_ALIGN)

    # X_ALIGN
    @test quad_to_coord(s2, nxyquads(s2) + nxzquads(s2) + 1) == (1, 1, 1, X_ALIGN)
    @test quad_to_coord(s2, nquads(s2)) == (nx(s2), nyb(s2), nzb(s2), X_ALIGN)
    @test quad_to_coord(s2, coord_to_quad(s2, 2, 3, 4, X_ALIGN)) == (2, 3, 4, X_ALIGN)
    @test quad_to_coord(s2, coord_to_quad(s2, 3, 1, 1, X_ALIGN)) == (3, 1, 1, X_ALIGN)
    @test quad_to_coord(s2, coord_to_quad(s2, 1, 3, 1, X_ALIGN)) == (1, 3, 1, X_ALIGN)
    @test quad_to_coord(s2, coord_to_quad(s2, 1, 1, 4, X_ALIGN)) == (1, 1, 4, X_ALIGN)

    q_x1 = coord_to_quad(s2, 1, 1, 1, X_ALIGN)
    q_x2 = coord_to_quad(s2, 2, 1, 1, X_ALIGN)
    q_x3 = coord_to_quad(s2, 3, 1, 1, X_ALIGN)
    @test q_x2 - q_x1 == 1
    @test q_x3 - q_x2 == 1

    q_y1 = coord_to_quad(s2, 1, 1, 1, X_ALIGN)
    q_y2 = coord_to_quad(s2, 1, 2, 1, X_ALIGN)
    @test q_y2 - q_y1 == nx(s2)

    # Full round trip tests for all quads
    for z in 1:nzb(s2), y in 1:nyb(s2), x in 1:nx(s2)
        idx = coord_to_quad(s2, x, y, z, X_ALIGN)
        @test quad_to_coord(s2, idx) == (x, y, z, X_ALIGN)
    end
    for z in 1:nz(s2), y in 1:nyb(s2), x in 1:nxb(s2)
        idx = coord_to_quad(s2, x, y, z, Z_ALIGN)
        @test quad_to_coord(s2, idx) == (x, y, z, Z_ALIGN)
    end
    for z in 1:nzb(s2), y in 1:ny(s2), x in 1:nxb(s2)
        idx = coord_to_quad(s2, x, y, z, Y_ALIGN)
        @test quad_to_coord(s2, idx) == (x, y, z, Y_ALIGN)
    end

    # halo mesh: strides still hold
    s3 = UniformCubicalComplex3D(3, 4, 5, 1.0, 1.0, 1.0; halo_x = 1, halo_y = 1, halo_z = 1)

    for z in 1:nzb(s3), y in 1:nyb(s3), x in 1:nx(s3)
        idx = coord_to_quad(s3, x, y, z, X_ALIGN)
        @test quad_to_coord(s3, idx) == (x, y, z, X_ALIGN)
    end
    for z in 1:nz(s3), y in 1:nyb(s3), x in 1:nxb(s3)
        idx = coord_to_quad(s3, x, y, z, Z_ALIGN)
        @test quad_to_coord(s3, idx) == (x, y, z, Z_ALIGN)
    end
    for z in 1:nzb(s3), y in 1:ny(s3), x in 1:nxb(s3)
        idx = coord_to_quad(s3, x, y, z, Y_ALIGN)
        @test quad_to_coord(s3, idx) == (x, y, z, Y_ALIGN)
    end
end

@testset "Incidence Relations" begin
    s = UniformCubicalComplex3D(2, 2, 2, 10.0, 10.0, 10.0)

    # X-Aligned edges
    @test src(s, 1, 1, 1, X_ALIGN) == 1
    @test tgt(s, 1, 1, 1, X_ALIGN) == 2

    @test src(s, 1, 2, 1, X_ALIGN) == 3
    @test tgt(s, 1, 2, 1, X_ALIGN) == 4

    @test src(s, 1, 1, 2, X_ALIGN) == 5
    @test tgt(s, 1, 1, 2, X_ALIGN) == 6

    @test src(s, 1, 2, 2, X_ALIGN) == nv(s) - 1
    @test tgt(s, 1, 2, 2, X_ALIGN) == nv(s)

    # Y-Aligned edges
    @test src(s, 1, 1, 1, Y_ALIGN) == 1
    @test tgt(s, 1, 1, 1, Y_ALIGN) == 3

    @test src(s, 2, 1, 1, Y_ALIGN) == 2
    @test tgt(s, 2, 1, 1, Y_ALIGN) == 4

    @test src(s, 1, 1, 2, Y_ALIGN) == 5
    @test tgt(s, 1, 1, 2, Y_ALIGN) == 7

    @test src(s, 2, 1, 2, Y_ALIGN) == nv(s) - 2
    @test tgt(s, 2, 1, 2, Y_ALIGN) == nv(s)

    # Z-Aligned edges
    @test src(s, 1, 1, 1, Z_ALIGN) == 1
    @test tgt(s, 1, 1, 1, Z_ALIGN) == 5

    @test src(s, 2, 1, 1, Z_ALIGN) == 2
    @test tgt(s, 2, 1, 1, Z_ALIGN) == 6

    @test src(s, 1, 2, 1, Z_ALIGN) == 3
    @test tgt(s, 1, 2, 1, Z_ALIGN) == 7

    @test src(s, 2, 2, 1, Z_ALIGN) == 4
    @test tgt(s, 2, 2, 1, Z_ALIGN) == nv(s)

    # Quads to vertices
    @test quad_vertices(s, 1, 1, 1, Z_ALIGN) == (1, 2, 4, 3)
    @test quad_vertices(s, 1, 1, 2, Z_ALIGN) == (5, 6, 8, 7)
    @test quad_vertices(s, 1, 1, 1, Y_ALIGN) == (1, 5, 6, 2)
    @test quad_vertices(s, 1, 2, 1, Y_ALIGN) == (3, 7, 8, 4)
    @test quad_vertices(s, 1, 1, 1, X_ALIGN) == (1, 3, 7, 5)
    @test quad_vertices(s, 2, 1, 1, X_ALIGN) == (2, 4, 8, 6)

    # Quads to edges
    @test quad_edges(s, 1, 1, 1, Z_ALIGN) == (1, 6, 2, 5)
    @test quad_edges(s, 1, 1, 2, Z_ALIGN) == (3, 8, 4, 7)
    @test quad_edges(s, 1, 1, 1, Y_ALIGN) == (9, 3, 10, 1)
    @test quad_edges(s, 1, 2, 1, Y_ALIGN) == (11, 4, 12, 2)
    @test quad_edges(s, 1, 1, 1, X_ALIGN) == (5, 11, 7, 9)
    @test quad_edges(s, 2, 1, 1, X_ALIGN) == (6, 12, 8, 10)

    # Cuboid to vertices
    @test boid_vertices(s, 1, 1, 1) == (1, 2, 4, 3, 5, 6, 8, 7)

    # Cuboid to face
    @test boid_quads(s, 1, 1, 1) == (1, 2, 3, 4, 5, 6)

    @test all(boid_edges(s, 1, 1, 1) .== collect(1:12))
end

@testset "Area and Volume Metrics" begin
    s = UniformCubicalComplex3D(5, 5, 5, 10.0, 20.0, 40.0)

    # Z-aligned (XY plane) = 2.5 * 5.0 = 12.5
    @test quad_area(s, Z_ALIGN) == 12.5
    @test quad_area(s, Z_ALIGN) == 12.5

    # Y-aligned (XZ plane) = 2.5 * 10.0 = 25.0
    @test quad_area(s, Y_ALIGN) == 25.0

    # X-aligned (YZ plane) = 5.0 * 10.0 = 50.0
    @test quad_area(s, X_ALIGN) == 50.0

    # Volume = 2.5 * 5.0 * 10.0 = 125.0
    @test boid_volume(s) == 125.0
    @test boid_volume(s) == 125.0
end

@testset "Dual Edge Lengths" begin
    # dx = 30.0 / 3 = 10.0
    # dy = 60.0 / 3 = 20.0
    # dz = 90.0 / 3 = 30.0
    s = UniformCubicalComplex3D(4, 4, 4, 30.0, 60.0, 90.0)

    # Z-aligned quads (normal to Z)
    # Boundary lengths should be dz / 2 = 15.0
    @test dual_edge_len(s, 2, 2, 1, Z_ALIGN) == 15.0
    @test dual_edge_len(s, 2, 2, 4, Z_ALIGN) == 15.0
    # Interior lengths should be dz = 30.0
    @test dual_edge_len(s, 2, 2, 2, Z_ALIGN) == 30.0
    @test dual_edge_len(s, 2, 2, 3, Z_ALIGN) == 30.0

    # Y-aligned quads (normal to Y)
    # Boundary lengths should be dy / 2 = 10.0
    @test dual_edge_len(s, 2, 1, 2, Y_ALIGN) == 10.0
    @test dual_edge_len(s, 2, 4, 2, Y_ALIGN) == 10.0
    # Interior lengths should be dy = 20.0
    @test dual_edge_len(s, 2, 2, 2, Y_ALIGN) == 20.0
    @test dual_edge_len(s, 2, 3, 2, Y_ALIGN) == 20.0

    # X-aligned quads (normal to X)
    # Boundary lengths should be dx / 2 = 5.0
    @test dual_edge_len(s, 1, 2, 2, X_ALIGN) == 5.0
    @test dual_edge_len(s, 4, 2, 2, X_ALIGN) == 5.0
    # Interior lengths should be dx = 10.0
    @test dual_edge_len(s, 2, 2, 2, X_ALIGN) == 10.0
    @test dual_edge_len(s, 3, 2, 2, X_ALIGN) == 10.0
end

@testset "Dual Boid Volume and Dual Quad Area" begin
    # dx = 30.0 / 3 = 10.0
    # dy = 60.0 / 3 = 20.0
    # dz = 90.0 / 3 = 30.0
    # Full volume = 6000.0
    s = UniformCubicalComplex3D(4, 4, 4, 30.0, 60.0, 90.0)

    # Dual boid volumes
    # 1. Interior vertex (no boundaries): 10 * 20 * 30
    @test dual_boid_volume(s, 2, 2, 2) == 6000.0
    # 2. Face boundary vertex (X boundary): 5 * 20 * 30
    @test dual_boid_volume(s, 1, 2, 2) == 3000.0
    # 3. Edge boundary vertex (X, Y boundaries): 5 * 10 * 30
    @test dual_boid_volume(s, 1, 1, 2) == 1500.0
    # 4. Corner boundary vertex (X, Y, Z boundaries): 5 * 10 * 15
    @test dual_boid_volume(s, 1, 1, 1) == 750.0
    @test dual_boid_volume(s, nx(s), ny(s), nz(s)) == 750.0

    # Dual quad areas

    # X-Aligned Edge (Dual quad in YZ plane, Full area = 20 * 30 = 600)
    # Note: Edge's 'x' coordinate doesn't affect YZ area
    @test dual_quad_area(s, 2, 2, 2, X_ALIGN) == 600.0 # Interior
    @test dual_quad_area(s, 1, 1, 2, X_ALIGN) == 300.0 # Y boundary
    @test dual_quad_area(s, 1, 2, 1, X_ALIGN) == 300.0 # Z boundary
    @test dual_quad_area(s, 1, 1, 1, X_ALIGN) == 150.0 # Y and Z boundary

    # Y-Aligned Edge (Dual quad in XZ plane, Full area = 10 * 30 = 300)
    @test dual_quad_area(s, 2, 2, 2, Y_ALIGN) == 300.0 # Interior
    @test dual_quad_area(s, 1, 2, 2, Y_ALIGN) == 150.0 # X boundary
    @test dual_quad_area(s, 2, 1, 4, Y_ALIGN) == 150.0 # Z boundary
    @test dual_quad_area(s, 4, 1, 4, Y_ALIGN) == 75.0  # X and Z boundary

    # Z-Aligned Edge (Dual quad in XY plane, Full area = 10 * 20 = 200)
    @test dual_quad_area(s, 2, 2, 2, Z_ALIGN) == 200.0 # Interior
    @test dual_quad_area(s, 1, 2, 2, Z_ALIGN) == 100.0 # X boundary
    @test dual_quad_area(s, 2, 4, 2, Z_ALIGN) == 100.0 # Y boundary
    @test dual_quad_area(s, 4, 4, 3, Z_ALIGN) == 50.0  # X and Y boundary
end

@testset "Quad to Incident Boids (Boundary Duplication)" begin
    # 3x3x3 vertices -> 2x2x2 boids (Total boids = 8)
    s = UniformCubicalComplex3D(3, 3, 3, 10.0, 10.0, 10.0)

    # Z-aligned (XY) quads
    # Interior: z=2 (Lower boid index 1, Higher boid index 5)
    @test quad_boids(s, 1, 1, 2, Z_ALIGN) == ((1, 5), (true, true))

    # Boundary (bottom): z=1 (No lower boid)
    @test quad_boids(s, 1, 1, 1, Z_ALIGN) == ((0, 1), (false, true))

    # Boundary (top): z=3 (No higher boid)
    @test quad_boids(s, 1, 1, 3, Z_ALIGN) == ((5, 0), (true, false))

    # Y-aligned (XZ) quads
    # Interior: y=2 (Higher boid index 3, Lower boid index 1)
    @test quad_boids(s, 1, 2, 1, Y_ALIGN) == ((1, 3), (true, true))

    # Boundary (back): y=1 (No lower boid)
    @test quad_boids(s, 1, 1, 1, Y_ALIGN) == ((0, 1), (false, true))

    # Boundary (front): y=3 (No higher boid)
    @test quad_boids(s, 1, 3, 1, Y_ALIGN) == ((3, 0), (true, false))

    # X-aligned (YZ) quads
    # Interior: x=2 (Higher boid index 2, Lower boid index 1)
    @test quad_boids(s, 2, 1, 1, X_ALIGN) == ((1, 2), (true, true))

    # Boundary (left): x=1 (No lower boid)
    @test quad_boids(s, 1, 1, 1, X_ALIGN) == ((0, 1), (false, true))

    # Boundary (right): x=3 (No higher boid)
    @test quad_boids(s, 3, 1, 1, X_ALIGN) == ((2, 0), (true, false))
end

@testset "Edge to Incident Quads" begin
    # Ground truth, independent of edge_quads' own ordering: every quad whose
    # own edge list (quad_edges) contains the edge in question.
    quads_touching_edge(s, e) = sort([
        q for q in quads(s) if e in quad_edges(s, quad_to_coord(s, q)...)
    ])

    for s in (UniformCubicalComplex3D(3, 4, 5, 1.0, 1.0, 1.0),
              UniformCubicalComplex3D(3, 4, 5, 1.0, 1.0, 1.0; halo_x = 1, halo_y = 1, halo_z = 1))
        for e in edges(s)
            x, y, z, align = edge_to_coord(s, e)
            (q_idx, q_valid) = edge_quads(s, x, y, z, align)
            got = sort([q_idx[i] for i in 1:4 if q_valid[i]])
            @test got == quads_touching_edge(s, e)
        end
    end

    # A concrete worked example, independently confirmed by the loop above.
    s = UniformCubicalComplex3D(3, 3, 3, 10.0, 10.0, 10.0)
    x, y, z = 2, 2, 2
    @test edge_quads(s, x, y, z, X_ALIGN) == ((6, 16, 8, 22), (true, true, true, true))
    @test edge_quads(s, x, y, z, Y_ALIGN) == ((29, 7, 35, 8), (true, true, true, true))
    @test edge_quads(s, x, y, z, Z_ALIGN) == ((21, 32, 22, 35), (true, true, true, true))
end

@testset "Edge to Incident Boids" begin
        s = UniformCubicalComplex3D(3, 3, 3, 1.0, 1.0, 1.0)

    idx, valid = edge_boids(s, 2, 2, 2, Z_ALIGN)
    @test valid == (true, true, true, true)
    @test idx[1] == coord_to_boid(s, 1, 1, 2)
    @test idx[2] == coord_to_boid(s, 2, 1, 2)
    @test idx[3] == coord_to_boid(s, 2, 2, 2)
    @test idx[4] == coord_to_boid(s, 1, 2, 2)

    idx, valid = edge_boids(s, 2, 1, 2, X_ALIGN)
    @test valid == (false, true, true, false)
    @test idx[1] == 0
    @test idx[2] == coord_to_boid(s, 2, 1, 1)
    @test idx[3] == coord_to_boid(s, 2, 1, 2)
    @test idx[4] == 0

    idx, valid = edge_boids(s, 1, 2, 1, Y_ALIGN)
    @test valid == (false, false, false, true)
    @test idx[1] == 0
    @test idx[2] == 0
    @test idx[3] == 0
    @test idx[4] == coord_to_boid(s, 1, 2, 1)
end

# TODO: Check this code to make sure it is working as intended
@testset "Vertex to Incident Edges (Explicit Indices)" begin
    s = UniformCubicalComplex3D(3, 3, 3, 10.0, 10.0, 10.0)
    @test vertex_edges(s, 2, 2, 2) == ((41, 50, 26, 29, 9, 10), (true, true, true, true, true, true))
end

@testset "Quad Edge and Boid Quad Offsets" begin
    for s in (UniformCubicalComplex3D(3, 4, 5, 1.0, 1.0, 1.0),
              UniformCubicalComplex3D(3, 4, 5, 1.0, 1.0, 1.0; halo_x = 1, halo_y = 1, halo_z = 1))

        # An offset moves along the quad's other in-plane axis, so offset 0/1 pick
        # out the low/high edge of the parallel pair returned by quad_edges.
        for q in quads(s)
            x, y, z, align = quad_to_coord(s, q)
            e1, e2, e3, e4 = quad_edges(s, x, y, z, align)

            if align == Z_ALIGN
                @test quad_edge_offset_3D(s, x, y, z, Z_ALIGN, X_ALIGN, 0) == e1
                @test quad_edge_offset_3D(s, x, y, z, Z_ALIGN, X_ALIGN, 1) == e3
                @test quad_edge_offset_3D(s, x, y, z, Z_ALIGN, Y_ALIGN, 0) == e4
                @test quad_edge_offset_3D(s, x, y, z, Z_ALIGN, Y_ALIGN, 1) == e2
            elseif align == Y_ALIGN
                @test quad_edge_offset_3D(s, x, y, z, Y_ALIGN, X_ALIGN, 0) == e4
                @test quad_edge_offset_3D(s, x, y, z, Y_ALIGN, X_ALIGN, 1) == e2
                @test quad_edge_offset_3D(s, x, y, z, Y_ALIGN, Z_ALIGN, 0) == e1
                @test quad_edge_offset_3D(s, x, y, z, Y_ALIGN, Z_ALIGN, 1) == e3
            else # X_ALIGN
                @test quad_edge_offset_3D(s, x, y, z, X_ALIGN, Y_ALIGN, 0) == e1
                @test quad_edge_offset_3D(s, x, y, z, X_ALIGN, Y_ALIGN, 1) == e3
                @test quad_edge_offset_3D(s, x, y, z, X_ALIGN, Z_ALIGN, 0) == e4
                @test quad_edge_offset_3D(s, x, y, z, X_ALIGN, Z_ALIGN, 1) == e2
            end
        end

        # Offset 0/1 pick out the low/high face on each axis of a boid.
        for b in boids(s)
            x, y, z = boid_to_coord(s, b)
            zf1, zf2, yf1, yf2, xf1, xf2 = boid_quads(s, x, y, z)

            @test boid_quad_offset(s, x, y, z, Z_ALIGN, 0) == zf1
            @test boid_quad_offset(s, x, y, z, Z_ALIGN, 1) == zf2
            @test boid_quad_offset(s, x, y, z, Y_ALIGN, 0) == yf1
            @test boid_quad_offset(s, x, y, z, Y_ALIGN, 1) == yf2
            @test boid_quad_offset(s, x, y, z, X_ALIGN, 0) == xf1
            @test boid_quad_offset(s, x, y, z, X_ALIGN, 1) == xf2
        end
    end
end

# TODO: Verify these expectations are correct. Each selector is checked against
# a brute-force scan of every edge/quad in the mesh, so the index arithmetic is
# independently verified, but which face each name refers to (and that "tangent"
# means "lying in the face") was inferred from the implementation and comments.
@testset "Boundary Tangent Edges" begin
    # Ground truth: every edge with the given alignment whose fixed coordinate
    # sits on the face. axis 1/2/3 selects x/y/z from edge_to_coord.
    edge_truth(s, axis, val, align) =
        [e for e in edges(s) if (c = edge_to_coord(s, e); c[4] == align && c[axis] == val)]

    for s in (UniformCubicalComplex3D(3, 4, 5, 1.0, 1.0, 1.0),
              UniformCubicalComplex3D(3, 4, 5, 1.0, 1.0, 1.0; halo_x = 1, halo_y = 1, halo_z = 1))

        # Each face fixes one coordinate and admits only the two alignments
        # that lie within it; the third alignment is normal to the face.
        for (got, want) in (
            (down_tangent_edges(s, Val(X_ALIGN)),  edge_truth(s, 3, 1,     X_ALIGN)),
            (down_tangent_edges(s, Val(Y_ALIGN)),  edge_truth(s, 3, 1,     Y_ALIGN)),
            (up_tangent_edges(s, Val(X_ALIGN)),    edge_truth(s, 3, nz(s), X_ALIGN)),
            (up_tangent_edges(s, Val(Y_ALIGN)),    edge_truth(s, 3, nz(s), Y_ALIGN)),
            (south_tangent_edges(s, Val(X_ALIGN)), edge_truth(s, 2, 1,     X_ALIGN)),
            (south_tangent_edges(s, Val(Z_ALIGN)), edge_truth(s, 2, 1,     Z_ALIGN)),
            (north_tangent_edges(s, Val(X_ALIGN)), edge_truth(s, 2, ny(s), X_ALIGN)),
            (north_tangent_edges(s, Val(Z_ALIGN)), edge_truth(s, 2, ny(s), Z_ALIGN)),
            (west_tangent_edges(s, Val(Y_ALIGN)),  edge_truth(s, 1, 1,     Y_ALIGN)),
            (west_tangent_edges(s, Val(Z_ALIGN)),  edge_truth(s, 1, 1,     Z_ALIGN)),
            (east_tangent_edges(s, Val(Y_ALIGN)),  edge_truth(s, 1, nx(s), Y_ALIGN)),
            (east_tangent_edges(s, Val(Z_ALIGN)),  edge_truth(s, 1, nx(s), Z_ALIGN)),
        )
            @test !isempty(want)
            @test sort(got) == want
            @test allunique(got)
        end

        # A face's tangent edges never include the alignment normal to it.
        @test !any(e -> is_edge_Z_aligned(e, s), down_tangent_edges(s))
        @test !any(e -> is_edge_Z_aligned(e, s), up_tangent_edges(s))
        @test !any(e -> is_edge_Y_aligned(e, s), south_tangent_edges(s))
        @test !any(e -> is_edge_Y_aligned(e, s), north_tangent_edges(s))
        @test !any(e -> is_edge_X_aligned(e, s), west_tangent_edges(s))
        @test !any(e -> is_edge_X_aligned(e, s), east_tangent_edges(s))

        # Per-face and all-face collections are the concatenations of their parts.
        @test down_tangent_edges(s) ==
            vcat(down_tangent_edges(s, Val(X_ALIGN)), down_tangent_edges(s, Val(Y_ALIGN)))
        @test west_tangent_edges(s) ==
            vcat(west_tangent_edges(s, Val(Y_ALIGN)), west_tangent_edges(s, Val(Z_ALIGN)))
        @test boundary_tangent_edges(s) ==
            vcat(down_tangent_edges(s), up_tangent_edges(s),
                 south_tangent_edges(s), north_tangent_edges(s),
                 west_tangent_edges(s), east_tangent_edges(s))
    end
end

# TODO: Verify these expectations are correct. See the note above; the same
# brute-force cross-check is used for the boundary quads.
@testset "Boundary Quads" begin
    quad_truth(s, axis, val, align) =
        [q for q in quads(s) if (c = quad_to_coord(s, q); c[4] == align && c[axis] == val)]

    for s in (UniformCubicalComplex3D(3, 4, 5, 1.0, 1.0, 1.0),
              UniformCubicalComplex3D(3, 4, 5, 1.0, 1.0, 1.0; halo_x = 1, halo_y = 1, halo_z = 1))

        # Each face carries the quads whose normal is the face's own axis.
        for (got, want) in (
            (down_quads(s),  quad_truth(s, 3, 1,     Z_ALIGN)),
            (up_quads(s),    quad_truth(s, 3, nz(s), Z_ALIGN)),
            (south_quads(s), quad_truth(s, 2, 1,     Y_ALIGN)),
            (north_quads(s), quad_truth(s, 2, ny(s), Y_ALIGN)),
            (west_quads(s),  quad_truth(s, 1, 1,     X_ALIGN)),
            (east_quads(s),  quad_truth(s, 1, nx(s), X_ALIGN)),
        )
            @test !isempty(want)
            @test sort(got) == want
            @test allunique(got)
        end

        @test boundary_quads(s) ==
            vcat(down_quads(s), up_quads(s), south_quads(s),
                 north_quads(s), west_quads(s), east_quads(s))
    end
end

@testset "Primal Boundary Extraction" begin
    s = UniformCubicalComplex3D(2, 2, 2, 1.0, 1.0, 1.0)

    # Expected vertices
    west_v_expected = [1, 3, 5, 7]
    east_v_expected = [2, 4, 6, 8]
    south_v_expected = [1, 2, 5, 6]
    north_v_expected = [3, 4, 7, 8]
    down_v_expected = [1, 2, 3, 4]
    up_v_expected = [5, 6, 7, 8]

    west_v, east_v = primal_boundary_vertices(s, EASTWEST)
    @test sort(west_v) == sort(west_v_expected)
    @test sort(east_v) == sort(east_v_expected)

    south_v, north_v = primal_boundary_vertices(s, NORTHSOUTH)
    @test sort(south_v) == sort(south_v_expected)
    @test sort(north_v) == sort(north_v_expected)

    down_v, up_v = primal_boundary_vertices(s, UPDOWN)
    @test sort(down_v) == sort(down_v_expected)
    @test sort(up_v) == sort(up_v_expected)

    # Expected quads
    west_q_expected = [5]
    east_q_expected = [6]
    south_q_expected = [3]
    north_q_expected = [4]
    down_q_expected = [1]
    up_q_expected = [2]

    east_q, west_q = primal_boundary_quads(s, EASTWEST)
    @test west_q == west_q_expected
    @test east_q == east_q_expected

    north_q, south_q = primal_boundary_quads(s, NORTHSOUTH)
    @test south_q == south_q_expected
    @test north_q == north_q_expected

    up_q, down_q = primal_boundary_quads(s, UPDOWN)
    @test down_q == down_q_expected
    @test up_q == up_q_expected

    # Expected boids
    boid_expected = [1]

    west_b, east_b = primal_boundary_boids(s, EASTWEST)
    @test west_b == boid_expected
    @test east_b == boid_expected

    south_b, north_b = primal_boundary_boids(s, NORTHSOUTH)
    @test south_b == boid_expected
    @test north_b == boid_expected

    down_b, up_b = primal_boundary_boids(s, UPDOWN)
    @test down_b == boid_expected
    @test up_b == boid_expected
end

@testset "Interior" begin
    nx_r, ny_r, nz_r = 3, 3, 3
    h = 1
    s = UniformCubicalComplex3D(nx_r, ny_r, nz_r, 1.0, 1.0, 1.0; halo_x = h, halo_y = h, halo_z = h)

    @testset "Boids" begin
        # Size checks
        @test nboids(s) == (nx_r + 2h - 1) * (ny_r + 2h - 1) * (nz_r + 2h - 1)
        @test nxbr(s) == nx_r - 1
        @test nybr(s) == ny_r - 1
        @test nzbr(s) == nz_r - 1

        # Halo exclusion: real boids = 1, halo = 0
        u = zeros(Float64, nboids(s))
        for rz in 1:nzbr(s), ry in 1:nybr(s), rx in 1:nxbr(s)
            u[coord_to_boid(s, rx + halo_west(s), ry + halo_south(s), rz + halo_down(s))] = 1.0
        end
        result = interior(Val(3), u, s)

        @test length(result) == nboidsr(s)
        @test all(result .== 1.0)
        @test !any(result .== 0.0)

        # Axis ordering: encode position so any axis swap is unambiguous
        u_ord = zeros(Float64, nboids(s))
        for rz in 1:nzbr(s), ry in 1:nybr(s), rx in 1:nxbr(s)
            u_ord[coord_to_boid(s, rx + halo_west(s), ry + halo_south(s), rz + halo_down(s))] = rx + 100 * ry + 10000 * rz
        end
        result_ord = reshape(interior(Val(3), u_ord, s), nxbr(s), nybr(s), nzbr(s))

        for rz in 1:nzbr(s), ry in 1:nybr(s), rx in 1:nxbr(s)
            @test result_ord[rx, ry, rz] == rx + 100 * ry + 10000 * rz
        end
    end

    @testset "Val(2) Quads" begin
        # Expected real counts per family:
        #   XY (z-aligned): nxq * nyq * nzr = 2 * 2 * 3 = 12
        #   XZ (y-aligned): nxq * nyr * nzq = 2 * 3 * 2 = 12
        #   YZ (x-aligned): nxr * nyq * nzq = 3 * 2 * 2 = 12
        #   Total: 36

        # Halo exclusion: real quads = 1, halo = 0
        q = zeros(Float64, nquads(s))
        for rz in 1:nzbr(s)+1, ry in 1:nybr(s), rx in 1:nxbr(s)  # XY family
            q[coord_to_quad(s, rx + halo_west(s), ry + halo_south(s), rz + halo_down(s), Z_ALIGN)] = 1.0
        end
        for rz in 1:nzbr(s), ry in 1:nybr(s)+1, rx in 1:nxbr(s)  # XZ family
            q[coord_to_quad(s, rx + halo_west(s), ry + halo_south(s), rz + halo_down(s), Y_ALIGN)] = 1.0
        end
        for rz in 1:nzbr(s), ry in 1:nybr(s), rx in 1:nxbr(s)+1  # YZ family
            q[coord_to_quad(s, rx + halo_west(s), ry + halo_south(s), rz + halo_down(s), X_ALIGN)] = 1.0
        end
        result = interior(Val(2), q, s)

        @test length(result) == 36
        @test all(result .== 1.0)
        @test !any(result .== 0.0)

        # Axis ordering per family — encode with rx + 100*ry + 10000*rz
        q_ord = zeros(Float64, nquads(s))
        for rz in 1:nzbr(s)+1, ry in 1:nybr(s), rx in 1:nxbr(s)
            q_ord[coord_to_quad(s, rx + halo_west(s), ry + halo_south(s), rz + halo_down(s), Z_ALIGN)] =
                rx + 100*ry + 10000*rz
        end
        for rz in 1:nzbr(s), ry in 1:nybr(s)+1, rx in 1:nxbr(s)
            q_ord[coord_to_quad(s, rx + halo_west(s), ry + halo_south(s), rz + halo_down(s), Y_ALIGN)] =
                rx + 100*ry + 10000*rz
        end
        for rz in 1:nzbr(s), ry in 1:nybr(s), rx in 1:nxbr(s)+1
            q_ord[coord_to_quad(s, rx + halo_west(s), ry + halo_south(s), rz + halo_down(s), X_ALIGN)] =
                rx + 100*ry + 10000*rz
        end
        result_ord = interior(Val(2), q_ord, s)

        # XY family: dims (nxbr, nybr, nzbr+1)
        xy = result_ord[1 : nxbr(s)*nybr(s)*(nzbr(s)+1)]
        xy_3d = reshape(xy, nxbr(s), nybr(s), nzbr(s)+1)
        for rz in 1:nzbr(s)+1, ry in 1:nybr(s), rx in 1:nxbr(s)
            @test xy_3d[rx, ry, rz] == rx + 100*ry + 10000*rz
        end

        # XZ family: dims (nxbr, nybr+1, nzbr)
        xz_offset = nxbr(s)*nybr(s)*(nzbr(s)+1)
        xz = result_ord[xz_offset+1 : xz_offset + nxbr(s)*(nybr(s)+1)*nzbr(s)]
        xz_3d = reshape(xz, nxbr(s), nybr(s)+1, nzbr(s))
        for rz in 1:nzbr(s), ry in 1:nybr(s)+1, rx in 1:nxbr(s)
            @test xz_3d[rx, ry, rz] == rx + 100*ry + 10000*rz
        end

        # YZ family: dims (nxbr+1, nybr, nzbr)
        yz_offset = xz_offset + nxbr(s)*(nybr(s)+1)*nzbr(s)
        yz = result_ord[yz_offset+1 : end]
        yz_3d = reshape(yz, nxbr(s)+1, nybr(s), nzbr(s))
        for rz in 1:nzbr(s), ry in 1:nybr(s), rx in 1:nxbr(s)+1
            @test yz_3d[rx, ry, rz] == rx + 100*ry + 10000*rz
        end
    end
end

@testset "Pseudo Mesh Element Counting" begin
    s = PseudoCubicalMesh3D(10, 8, 6)
    s_h = PseudoCubicalMesh3D(10, 8, 6; halo_x = 2, halo_y = 3, halo_z = 1)

    # Real counts
    @test nxr(s) == 10
    @test nyr(s) == 8
    @test nzr(s) == 6
    @test nxr(s_h) == 10
    @test nyr(s_h) == 8
    @test nzr(s_h) == 6

    # Halo accessors
    @test halo_west(s) == 0
    @test halo_south(s) == 0
    @test halo_down(s) == 0
    @test halo_west(s_h) == 2
    @test halo_south(s_h) == 3
    @test halo_down(s_h) == 1

    # Total (halo-inclusive) counts
    @test nx(s) == 10
    @test ny(s) == 8
    @test nz(s) == 6
    @test nx(s_h) == 14
    @test ny(s_h) == 14
    @test nz(s_h) == 8

    # Vertex counts
    @test nv(s) == 10 * 8 * 6
    @test nvr(s) == 10 * 8 * 6
    @test nv(s_h) == 14 * 14 * 8
    @test nvr(s_h) == 10 * 8 * 6

    # Edge counts
    @test nxedges(s) == 9 * 8 * 6
    @test nyedges(s) == 10 * 7 * 6
    @test nzedges(s) == 10 * 8 * 5
    @test ne(s) == nxedges(s) + nyedges(s) + nzedges(s)

    @test nxedges(s_h) == 13 * 14 * 8
    @test nyedges(s_h) == 14 * 13 * 8
    @test nzedges(s_h) == 14 * 14 * 7
    @test ne(s_h) == nxedges(s_h) + nyedges(s_h) + nzedges(s_h)

    # Quad counts
    @test nxyquads(s) == 9 * 7 * 6
    @test nxzquads(s) == 9 * 8 * 5
    @test nyzquads(s) == 10 * 7 * 5
    @test nquads(s) == nxyquads(s) + nxzquads(s) + nyzquads(s)

    @test nxyquads(s_h) == 13 * 13 * 8
    @test nxzquads(s_h) == 13 * 14 * 7
    @test nyzquads(s_h) == 14 * 13 * 7
    @test nquads(s_h) == nxyquads(s_h) + nxzquads(s_h) + nyzquads(s_h)

    # Boid counts
    @test nboids(s) == 9 * 7 * 5
    @test nboidsr(s) == 9 * 7 * 5
    @test nboids(s_h) == 13 * 13 * 7
    @test nboidsr(s_h) == 9 * 7 * 5

    # Indexing
    @test coord_to_vert(s, 1, 1, 1) == 1
    @test coord_to_vert(s, 10, 8, 6) == nv(s)

    @test coord_to_boid(s, 1, 1, 1) == 1
    @test coord_to_boid(s, 9, 7, 5) == nboids(s)

    @test coord_to_edge(s, 1, 1, 1, X_ALIGN) == 1
    @test coord_to_edge(s, 1, 1, 1, Y_ALIGN) == nxedges(s) + 1
    @test coord_to_edge(s, 1, 1, 1, Z_ALIGN) == nxedges(s) + nyedges(s) + 1

    @test coord_to_quad(s, 1, 1, 1, Z_ALIGN) == 1
    @test coord_to_quad(s, 1, 1, 1, Y_ALIGN) == nxyquads(s) + 1
    @test coord_to_quad(s, 1, 1, 1, X_ALIGN) == nxyquads(s) + nxzquads(s) + 1

    # Halo flags
    @test valid_boid(s_h, 1, 1, 1) == true
    @test valid_boid(s_h, 13, 13, 7) == true
    @test valid_boid(s_h, 14, 1, 1) == false
    @test valid_boid(s_h, 1, 14, 1) == false
    @test valid_boid(s_h, 1, 1, 8) == false
end

@testset "Show" begin
    s = UniformCubicalComplex3D(5, 5, 5, 10.0, 10.0, 10.0)
    s_h = UniformCubicalComplex3D(5, 5, 5, 10.0, 10.0, 10.0; halo_x = 1, halo_y = 1, halo_z = 1)
    @test isnothing(show(IOBuffer(), s))
    @test isnothing(show(IOBuffer(), s_h))
end

end
