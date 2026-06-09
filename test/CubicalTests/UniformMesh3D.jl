using Test
using GeometryBasics

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMesh3D.jl")

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
    @test hx(s) == 0 && hy(s) == 0 && hz(s) == 0
end

@testset "Rectangular Prism" begin
    nx_ = 2; ny_ = 5; nz_ = 6;
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
    # Using a different constructor signature
    s = UniformCubicalComplex3D(2, 5, 6, 10.0, 40.0, 50.0; halo_x=1, halo_y=1, halo_z=1)

    # Halo tests
    @test hx(s) == 1 && hy(s) == 1 && hz(s) == 1

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

    # --- Vertices ---
    # First vertex
    @test vert_to_coord(s, 1) == (1, 1, 1)
    # Last vertex: nx*ny*nz = 3*4*5 = 60
    @test vert_to_coord(s, nv(s)) == (3, 4, 5)
    # Intermediate vertex: z=2 (adds 12), y=2 (adds 3), x=2 => 1 + 12 + 3 + 1 = 17
    @test vert_to_coord(s, 17) == (2, 2, 2)

    # --- Boids ---
    # Boid dimensions: nxb=2, nyb=3, nzb=4. Total = 24
    @test boid_to_coord(s, 1) == (1, 1, 1)
    @test boid_to_coord(s, nboids(s)) == (2, 3, 4)
    # Intermediate boid: z=2 (adds 6), y=2 (adds 2), x=1 => 1 + 6 + 2 + 0 = 9
    @test boid_to_coord(s, 9) == (1, 2, 2)

    # --- Edges ---
    # X-aligned limits: nxe=2, ny=4, nz=5. Total X-edges = 40
    @test edge_to_coord(s, 1) == (1, 1, 1, X_ALIGN)
    @test edge_to_coord(s, 40) == (2, 4, 5, X_ALIGN)
    # Y-aligned limits: nx=3, nye=3, nz=5. Total Y-edges = 45. Starts at 41, ends at 85
    @test edge_to_coord(s, 41) == (1, 1, 1, Y_ALIGN)
    @test edge_to_coord(s, 85) == (3, 3, 5, Y_ALIGN)
    # Z-aligned limits: nx=3, ny=4, nze=4. Total Z-edges = 48. Starts at 86, ends at 133
    @test edge_to_coord(s, 86) == (1, 1, 1, Z_ALIGN)
    @test edge_to_coord(s, ne(s)) == (3, 4, 4, Z_ALIGN)

    # --- Quads ---
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
    # Construct a mesh with a 1-cell halo padding in all three dimensions
    # Real sizes: nxr=2, nyr=5, nzr=6
    # Halo padding adds +2 to each dimension: 
    # Total sizes: nx=4, ny=7, nz=8
    # Boid dimensions: nxb=3, nyb=6, nzb=7
    s = UniformCubicalComplex3D(2, 5, 6, 10.0, 40.0, 50.0; halo_x=1, halo_y=1, halo_z=1)

    # --- Vertices (Total: 4 * 7 * 8 = 224) ---
    @test vert_to_coord(s, 1) == (1, 1, 1)
    @test vert_to_coord(s, nv(s)) == (4, 7, 8)
    
    # Check intermediate index for (2, 3, 4)
    # index = (4 - 1) * (4 * 7) + (3 - 1) * 4 + 2 = 84 + 8 + 2 = 94
    @test vert_to_coord(s, 94) == (2, 3, 4)


    # --- Boids (Total: 3 * 6 * 7 = 126) ---
    @test boid_to_coord(s, 1) == (1, 1, 1)
    @test boid_to_coord(s, nboids(s)) == (3, 6, 7)
    
    # Check intermediate index for (2, 3, 4)
    # index = (4 - 1) * (3 * 6) + (3 - 1) * 3 + 2 = 54 + 6 + 2 = 62
    @test boid_to_coord(s, 62) == (2, 3, 4)


    # --- Edges (Total: 168 X-edges, 192 Y-edges, 196 Z-edges = 556) ---
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


    # --- Quads (Total: 144 Z-quads, 147 Y-quads, 168 X-quads = 459) ---
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

    # --- Z-Aligned Quads (Normal to Z-axis) ---
    # Boundary lengths should be dz / 2 = 15.0
    @test dual_edge_length(s, 2, 2, 1, Z_ALIGN) == 15.0
    @test dual_edge_length(s, 2, 2, 4, Z_ALIGN) == 15.0
    # Interior lengths should be dz = 30.0
    @test dual_edge_length(s, 2, 2, 2, Z_ALIGN) == 30.0
    @test dual_edge_length(s, 2, 2, 3, Z_ALIGN) == 30.0

    # --- Y-Aligned Quads (Normal to Y-axis) ---
    # Boundary lengths should be dy / 2 = 10.0
    @test dual_edge_length(s, 2, 1, 2, Y_ALIGN) == 10.0
    @test dual_edge_length(s, 2, 4, 2, Y_ALIGN) == 10.0
    # Interior lengths should be dy = 20.0
    @test dual_edge_length(s, 2, 2, 2, Y_ALIGN) == 20.0
    @test dual_edge_length(s, 2, 3, 2, Y_ALIGN) == 20.0

    # --- X-Aligned Quads (Normal to X-axis) ---
    # Boundary lengths should be dx / 2 = 5.0
    @test dual_edge_length(s, 1, 2, 2, X_ALIGN) == 5.0
    @test dual_edge_length(s, 4, 2, 2, X_ALIGN) == 5.0
    # Interior lengths should be dx = 10.0
    @test dual_edge_length(s, 2, 2, 2, X_ALIGN) == 10.0
    @test dual_edge_length(s, 3, 2, 2, X_ALIGN) == 10.0
end

@testset "Dual Boid Volume and Dual Quad Area" begin
    # dx = 30.0 / 3 = 10.0
    # dy = 60.0 / 3 = 20.0
    # dz = 90.0 / 3 = 30.0
    # Full volume = 6000.0
    s = UniformCubicalComplex3D(4, 4, 4, 30.0, 60.0, 90.0)

    # --- Dual Boid Volumes ---
    # 1. Interior vertex (no boundaries): 10 * 20 * 30
    @test dual_boid_volume(s, 2, 2, 2) == 6000.0
    # 2. Face boundary vertex (X boundary): 5 * 20 * 30
    @test dual_boid_volume(s, 1, 2, 2) == 3000.0
    # 3. Edge boundary vertex (X, Y boundaries): 5 * 10 * 30
    @test dual_boid_volume(s, 1, 1, 2) == 1500.0
    # 4. Corner boundary vertex (X, Y, Z boundaries): 5 * 10 * 15
    @test dual_boid_volume(s, 1, 1, 1) == 750.0
    @test dual_boid_volume(s, nx(s), ny(s), nz(s)) == 750.0


    # --- Dual Quad Areas ---
    
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

    # --- Z-Aligned (XY) Quads ---
    # Interior: z=2 (Lower boid index 1, Higher boid index 5)
    @test quad_boids(s, 1, 1, 2, Z_ALIGN) == ((1, 5), (true, true))

    # Boundary (bottom): z=1 (No lower boid)
    @test quad_boids(s, 1, 1, 1, Z_ALIGN) == ((0, 1), (false, true))

    # Boundary (top): z=3 (No higher boid)
    @test quad_boids(s, 1, 1, 3, Z_ALIGN) == ((5, 0), (true, false))


    # --- Y-Aligned (XZ) Quads ---
    # Interior: y=2 (Higher boid index 3, Lower boid index 1)
    @test quad_boids(s, 1, 2, 1, Y_ALIGN) == ((1, 3), (true, true))

    # Boundary (back): y=1 (No lower boid)
    @test quad_boids(s, 1, 1, 1, Y_ALIGN) == ((0, 1), (false, true))

    # Boundary (front): y=3 (No higher boid)
    @test quad_boids(s, 1, 3, 1, Y_ALIGN) == ((3, 0), (true, false))


    # --- X-Aligned (YZ) Quads ---
    # Interior: x=2 (Higher boid index 2, Lower boid index 1)
    @test quad_boids(s, 2, 1, 1, X_ALIGN) == ((1, 2), (true, true))

    # Boundary (left): x=1 (No lower boid)
    @test quad_boids(s, 1, 1, 1, X_ALIGN) == ((0, 1), (false, true))

    # Boundary (right): x=3 (No higher boid)
    @test quad_boids(s, 3, 1, 1, X_ALIGN) == ((2, 0), (true, false))
end

# TODO: Check this code to make sure it is working as intended
@testset "Edge to Incident Quads" begin
    s = UniformCubicalComplex3D(3, 3, 3, 10.0, 10.0, 10.0)

    # Base coordinate for our interior edges
    x, y, z = 2, 2, 2

    @test edge_quads(s, x, y, z, X_ALIGN) == ((6, 16, 8, 22), (true, true, true, true))
    @test edge_quads(s, x, y, z, Y_ALIGN) == ((29, 7, 35, 8), (true, true, true, true))
    @test edge_quads(s, x, y, z, Z_ALIGN) == ((21, 32, 22, 35), (true, true, true, true))
end

@testset "Edge to Incident Boids" begin
    # Small mesh for index testing
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

@testset "Primal Boundary Extraction" begin
    s = UniformCubicalComplex3D(2, 2, 2, 1.0, 1.0, 1.0)

    # --- Expected Vertices ---
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

    # --- Expected Quads ---
    west_q_expected = [5]
    east_q_expected = [6]
    south_q_expected = [3]
    north_q_expected = [4]
    down_q_expected = [1]
    up_q_expected = [2]
    
    east_q, west_q= primal_boundary_quads(s, EASTWEST)
    @test west_q == west_q_expected
    @test east_q == east_q_expected

    north_q, south_q= primal_boundary_quads(s, NORTHSOUTH)
    @test south_q == south_q_expected
    @test north_q == north_q_expected

    up_q, down_q = primal_boundary_quads(s, UPDOWN)
    @test down_q == down_q_expected
    @test up_q == up_q_expected

    # --- Expected Boids ---
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
