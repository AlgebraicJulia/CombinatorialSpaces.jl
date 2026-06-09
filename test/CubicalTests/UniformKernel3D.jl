using Test
using KernelAbstractions

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMesh3D.jl")
include("../../src/CubicalCode/UniformKernelDEC3D.jl")

@testset "Exterior Derivative Kernels" begin
    s = UniformCubicalComplex3D(2, 2, 2, 1.0, 1.0, 1.0)
    FT = Float64

    @testset "dd == 0" begin
        f0 = rand(FT, nv(s))
        f1 = exterior_derivative(Val(0), s, f0)
        f2 = exterior_derivative(Val(1), s, f1)
        @test all(isapprox.(f2, 0, atol=1e-12))

        f1 = rand(FT, ne(s))
        f2 = exterior_derivative(Val(1), s, f1)
        f3 = exterior_derivative(Val(2), s, f2)
        @test all(isapprox.(f3, 0, atol=1e-12))
    end
end

@testset "Hodge Star Operators" begin
    s = UniformCubicalComplex3D(3, 4, 5, 10.0, 20.0, 30.0)
    FT = Float64
    
    @testset "Hodge and Hodge Inverse" begin
        f0 = rand(FT, nv(s))
        f0_rec = inv_hodge_star(Val(0), s, hodge_star(Val(0), s, f0))
        @test f0 ≈ f0_rec
        
        f1 = rand(FT, ne(s))
        f1_rec = inv_hodge_star(Val(1), s, hodge_star(Val(1), s, f1))
        @test f1 ≈ f1_rec
        
        f2 = rand(FT, nquads(s))
        f2_rec = inv_hodge_star(Val(2), s, hodge_star(Val(2), s, f2))
        @test f2 ≈ f2_rec
        
        f3 = rand(FT, nboids(s))
        f3_rec = inv_hodge_star(Val(3), s, hodge_star(Val(3), s, f3))
        @test f3 ≈ f3_rec
    end

    s = UniformCubicalComplex3D(2, 2, 2, 2.0, 3.0, 4.0)
    FT = Float64

    @testset "Positivity" begin
        f0 = ones(FT, nv(s))
        star_f0 = hodge_star(Val(0), s, f0)
        @test all(star_f0 .> 0)

        f1 = ones(FT, ne(s))
        star_f1 = hodge_star(Val(1), s, f1)
        @test all(star_f1 .> 0)

        f2 = ones(FT, nquads(s))
        star_f2 = hodge_star(Val(2), s, f2)
        @test all(star_f2 .> 0)

        f3 = ones(FT, nboids(s))
        star_f3 = hodge_star(Val(3), s, f3)
        @test all(star_f3 .> 0)
    end

    @testset "Numerical Accuracy" begin
        # For a 2x2x2 mesh, all dual cells are the same since there's only one boid.
        # Primal edge lengths: dx=2, dy=3, dz=4
        # Primal quad areas: dx*dy=6 (Z), dx*dz=8 (Y), dy*dz=12 (X)
        # Primal boid volume: dx*dy*dz = 24
        # Dual volumes are half at boundaries, but this mesh is all boundary.
        # Dual boid volume for each vertex: (dx/2)*(dy/2)*(dz/2) = 1*1.5*2 = 3
        # Dual quad area for each edge: (dy/2)*(dz/2)=3 (X-align), (dx/2)*(dz/2)=2 (Y-align), (dx/2)*(dy/2)=1.5 (Z-align)
        # Dual edge length for each quad: dz/2=2 (Z-align), dy/2=1.5 (Y-align), dx/2=1 (X-align)
        
        f0 = FT[1, 2, 3, 4, 5, 6, 7, 8]
        star_f0 = hodge_star(Val(0), s, f0)
        expected_star_f0 = f0 .* 3.0
        @test star_f0 ≈ expected_star_f0

        s_333 = UniformCubicalComplex3D(3, 3, 3, 2.0, 3.0, 4.0)
        f0_center = zeros(FT, nv(s_333)); f0_center[14] = 1.0; # Center vertex
        star_f0_center = hodge_star(Val(0), s_333, f0_center)
        # Only the central boid (idx 8) should be non-zero
        @test star_f0_center[14] ≈ 1.0 * (1.0*1.5*2.0) # interior dual boid volume

        f1 = ones(FT, ne(s))
        star_f1 = hodge_star(Val(1), s, f1)
        # Expected ratio: dual_quad_area / primal_edge_len
        # X-aligned edges (4 of them): 3 / 2 = 1.5
        # Y-aligned edges (4 of them): 2 / 3
        # Z-aligned edges (4 of them): 1.5 / 4 = 0.375
        @test star_f1[1:4] ≈ ones(4) .* 1.5
        @test star_f1[5:8] ≈ ones(4) .* (2/3)
        @test star_f1[9:12] ≈ ones(4) .* 0.375

        f2 = ones(FT, nquads(s))
        star_f2 = hodge_star(Val(2), s, f2)
        # Expected ratio: dual_edge_len / primal_quad_area
        # Z-aligned quads (2 of them): 2 / 6 = 1/3
        # Y-aligned quads (2 of them): 1.5 / 8 = 0.1875
        # X-aligned quads (2 of them): 1 / 12
        @test star_f2[1:2] ≈ ones(2) .* (1/3)
        @test star_f2[3:4] ≈ ones(2) .* 0.1875
        @test star_f2[5:6] ≈ ones(2) .* (1/12)

        f3 = FT[10.0]
        star_f3 = hodge_star(Val(3), s, f3)
        expected_star_f3 = [10.0 / 24.0]
        @test star_f3 ≈ expected_star_f3
    end
end

@testset "Dual Exterior Derivative Kernels" begin
    s = UniformCubicalComplex3D(3, 3, 3, 1.0, 1.0, 1.0)
    FT = Float64

    @testset "Numerical Checks" begin
        f0_dual = [i for i in boids(s)]
        f1_dual = dual_exterior_derivative(Val(0), s, f0_dual)

        # Interior only
        @test all(f1_dual[5:8] .== 4)

        @test all(f1_dual[15:16] .== 2)
        @test all(f1_dual[21:22] .== 2)

        @test all(f1_dual[[26, 29, 32, 35]] .== 1)

        # XY-quads (Z_ALIGN)
        f1_dual_xy = zeros(nquads(s))
        f1_dual_xy[5] = 1.0; f1_dual_xy[[6,7]] .= 2.0; f1_dual_xy[8] = 3.0
        f2_dual_xy = dual_exterior_derivative(Val(1), s, f1_dual_xy)
        @test f2_dual_xy[9] == 1.0
        @test f2_dual_xy[10] == 1.0

        @test f2_dual_xy[26] == -1.0
        @test f2_dual_xy[29] == -1.0

        # XZ-quads (Y_ALIGN)
        f1_dual_xz = zeros(FT, nquads(s))
        f1_dual_xz[15] = 1.0; f1_dual_xz[[16, 21]] .= 2.0; f1_dual_xz[22] = 3.0
        f2_dual_xz = dual_exterior_derivative(Val(1), s, f1_dual_xz)
        
        @test f2_dual_xz[9] == -1.0
        @test f2_dual_xz[10] == -1.0

        @test f2_dual_xz[41] == 1.0
        @test f2_dual_xz[50] == 1.0

        # YZ-quads (X_ALIGN)
        f1_dual_yz = zeros(FT, nquads(s))
        f1_dual_yz[26] = 1.0; f1_dual_yz[[29, 32]] .= 2.0; f1_dual_yz[35] = 3.0
        f2_dual_yz = dual_exterior_derivative(Val(1), s, f1_dual_yz)
        
        @test f2_dual_yz[26] == 1.0
        @test f2_dual_yz[29] == 1.0

        @test f2_dual_yz[41] == -1.0
        @test f2_dual_yz[50] == -1.0

        # TODO: Add tests for dual deriv 2
    end

    @testset "dd == 0" begin
        f0_dual = rand(FT, nboids(s))
        f1_dual = dual_exterior_derivative(Val(0), s, f0_dual)
        f2_dual = dual_exterior_derivative(Val(1), s, f1_dual)
        @test all(isapprox.(f2_dual[[9, 10, 25, 28, 39, 48]], 0, atol=1e-12))

        f1_dual = rand(FT, nquads(s))
        f2_dual = dual_exterior_derivative(Val(1), s, f1_dual)
        f3_dual = dual_exterior_derivative(Val(2), s, f2_dual)
        @test all(isapprox.(f3_dual, 0, atol=1e-12))        
    end
end

# Test_UniformMesh3D.txt additions

@testset "Wedge Product Kernels" begin
    s = UniformCubicalComplex3D(3, 3, 3, 1.0, 1.0, 1.0)
    FT = Float64

    @testset "Wedge 11" begin
        # Linear dependence means zero
        f1 = ones(FT, ne(s)) .* 2.0
        g1 = ones(FT, ne(s)) .* 3.0
        w11 = wedge_product(Val(1), Val(1), s, f1, g1)
        @test all(isapprox.(w11, 0.0, atol=1e-12))

        # XY Quads (Z_ALIGN)
        f1 .= 0.0; g1 .= 0.0
        xedges(s, f1) .= 2.0
        yedges(s, g1) .= 3.0
        w11_xy = wedge_product(Val(1), Val(1), s, f1, g1)
        @test all(isapprox.(xyquads(s, w11_xy), 6.0, atol=1e-12))
        @test all(isapprox.(xzquads(s, w11_xy), 0.0, atol=1e-12))
        @test all(isapprox.(yzquads(s, w11_xy), 0.0, atol=1e-12))

        w11_xy = wedge_product(Val(1), Val(1), s, g1, f1)
        @test all(isapprox.(xyquads(s, w11_xy), -6.0, atol=1e-12))
        @test all(isapprox.(xzquads(s, w11_xy), 0.0, atol=1e-12))
        @test all(isapprox.(yzquads(s, w11_xy), 0.0, atol=1e-12))

        # XZ Quads (Y_ALIGN)
        f1 .= 0.0; g1 .= 0.0
        xedges(s, f1) .= 2.0
        zedges(s, g1) .= 4.0
        w11_xz = wedge_product(Val(1), Val(1), s, f1, g1)
        @test all(isapprox.(xyquads(s, w11_xz), 0.0, atol=1e-12))
        @test all(isapprox.(xzquads(s, w11_xz), 8.0, atol=1e-12))
        @test all(isapprox.(yzquads(s, w11_xz), 0.0, atol=1e-12))

        w11_xz = wedge_product(Val(1), Val(1), s, g1, f1)
        @test all(isapprox.(xyquads(s, w11_xz), 0.0, atol=1e-12))
        @test all(isapprox.(xzquads(s, w11_xz), -8.0, atol=1e-12))
        @test all(isapprox.(yzquads(s, w11_xz), 0.0, atol=1e-12))


        # YZ Quads (X_ALIGN)
        f1 .= 0.0; g1 .= 0.0
        yedges(s, f1) .= 5.0
        zedges(s, g1) .= 4.0
        w11_yz = wedge_product(Val(1), Val(1), s, f1, g1)
        @test all(isapprox.(xyquads(s, w11_yz), 0.0, atol=1e-12))
        @test all(isapprox.(xzquads(s, w11_yz), 0.0, atol=1e-12))
        @test all(isapprox.(yzquads(s, w11_yz), 20.0, atol=1e-12))

        w11_yz = wedge_product(Val(1), Val(1), s, g1, f1)
        @test all(isapprox.(xyquads(s, w11_yz), 0.0, atol=1e-12))
        @test all(isapprox.(xzquads(s, w11_yz), 0.0, atol=1e-12))
        @test all(isapprox.(yzquads(s, w11_yz), -20.0, atol=1e-12))

    end

    @testset "Wedge 12" begin
        # This should return the volume of the boid as 6.0 = 2.0 * 3.0
        f1 = ones(FT, ne(s)) .* 2.0
        g2 = ones(FT, nquads(s)) .* 3.0
        w12 = wedge_product(Val(1), Val(2), s, f1, g2)
        @test all(isapprox.(w12, 6.0, atol=1e-12))

        # 2 * 3 - 4 * 5 + 6 * 7 = 6 - 20 + 42 = 28
        f1 = zeros(FT, ne(s))
        xedges(s, f1) .= 2.0
        yedges(s, f1) .= 4.0
        zedges(s, f1) .= 6.0

        g2 = zeros(FT, nquads(s))
        xyquads(s, g2) .= 7.0
        xzquads(s, g2) .= 5.0
        yzquads(s, g2) .= 3.0

        w12 = wedge_product(Val(1), Val(2), s, f1, g2)
        @test all(isapprox.(w12, 28.0, atol=1e-12))
    end
end

@testset "Dual Wedge Product Kernels" begin
    s = UniformCubicalComplex3D(3, 3, 3, 1.0, 1.0, 1.0)
    FT = Float64
    
    @testset "Wedge DD 01" begin
        f = ones(FT, nboids(s)) .* 2.0
        a = ones(FT, nquads(s)) .* 3.0
        w01 = wedge_product_dd(Val(0), Val(1), s, f, a)
        
        # Whether on boundary (1 valid boid -> just take value) or interior (average of identical values),
        # the result should be 2.0 * 3.0 = 6.0.
        @test all(w01 .≈ 6.0)

        # Check a specific boundary case explicitly: 
        # Z-aligned quad on z=1 (boundary)
        f_grad = FT.(1:nboids(s))
        a_ones = ones(FT, nquads(s))
        w01_grad = wedge_product_dd(Val(0), Val(1), s, f_grad, a_ones)
        
        q_idx = coord_to_quad(s, 1, 1, 1, Z_ALIGN)
        b_indices, b_valid = quad_boids(s, 1, 1, 1, Z_ALIGN)
        @test b_valid == (false, true)
        @test w01_grad[q_idx] ≈ FT.(b_indices[2]) * 1.0

        q_idx = coord_to_quad(s, 1, 1, 3, Z_ALIGN)
        b_indices, b_valid = quad_boids(s, 1, 1, 3, Z_ALIGN)
        @test b_valid == (true, false)
        @test w01_grad[q_idx] ≈ FT.(b_indices[1]) * 1.0

        q_idx = coord_to_quad(s, 1, 1, 2, Z_ALIGN)
        b_indices, b_valid = quad_boids(s, 1, 1, 2, Z_ALIGN)
        @test b_valid == (true, true)
        @test w01_grad[q_idx] ≈ FT.(sum(b_indices)/2) * 1.0
    end
end

@testset "Sharp and Flat Operators" begin
    @testset "Sharp DD" begin
        s = UniformCubicalComplex3D(4, 4, 4, 1.0, 2.0, 4.0)
        FT = Float64

        f = zeros(FT, nquads(s))
        
        # --- Test 1: Interior Boid (2,2,2) ---
        boid_idx = coord_to_boid(s, 2, 2, 2)
        q_z1, q_z2, q_y1, q_y2, q_x1, q_x2 = boid_quads(s, 2, 2, 2)
        
        f[q_x1] = 1.0; f[q_x2] = 3.0
        f[q_y1] = 5.0; f[q_y2] = 5.0
        f[q_z1] = 8.0; f[q_z2] = 6.0

        X, Y, Z = sharp_dd(s, f)
        
        @test X[boid_idx] ≈ 2.0 / dx(s)
        @test Y[boid_idx] ≈ 5.0 / dy(s)
        @test Z[boid_idx] ≈ 7.0 / dz(s)

        # --- Test 2: Boundary Corner Boid (1,1,1) ---
        f .= 0.0
        boid_idx_corner = coord_to_boid(s, 1, 1, 1)
        q_z1, q_z2, q_y1, q_y2, q_x1, q_x2 = boid_quads(s, 1, 1, 1)

        f[q_x1] = 1.0; f[q_x2] = 3.0
        f[q_y1] = 2.0; f[q_y2] = 4.0
        f[q_z1] = 5.0; f[q_z2] = 6.0
        
        X_c, Y_c, Z_c = sharp_dd(s, f)
        
        @test X_c[boid_idx_corner] ≈ 2.5 / dx(s)
        @test Y_c[boid_idx_corner] ≈ 4.0 / dy(s)
        @test Z_c[boid_idx_corner] ≈ 8.0 / dz(s)
        
        # --- Test 3: Boid on Edge (1, 2, nzb(s)) ---
        f .= 0.0
        boid_idx_edge = coord_to_boid(s, 1, 2, nzb(s))
        q_z1, q_z2, q_y1, q_y2, q_x1, q_x2 = boid_quads(s, 1, 2, nzb(s))

        f[q_x1] = 1.0; f[q_x2] = 3.0 # West Boundary
        f[q_y1] = 2.0; f[q_y2] = 4.0 # Interior
        f[q_z1] = 5.0; f[q_z2] = 6.0 # Up Boundary
        
        X_e, Y_e, Z_e = sharp_dd(s, f)
        
        @test X_e[boid_idx_edge] ≈ 2.5 / dx(s)
        @test Y_e[boid_idx_edge] ≈ 3.0 / dy(s)
        @test Z_e[boid_idx_edge] ≈ 8.5 / dz(s)
    end

    @testset "Flat DP" begin
        s = UniformCubicalComplex3D(3, 3, 3, 1.0, 2.0, 4.0)
        FT = Float64

        # Constant vector field
        C_x, C_y, C_z = 1.5, 2.5, 3.5
        X_const = fill(FT(C_x), nboids(s))
        Y_const = fill(FT(C_y), nboids(s))
        Z_const = fill(FT(C_z), nboids(s))

        f_const = flat_dp(s, X_const, Y_const, Z_const)

        # --- Test 1: Interior Edge ---
        edge_idx_int = coord_to_edge(s, 2, 2, 2, X_ALIGN)
        @test f_const[edge_idx_int] ≈ C_x * dx(s)

        # --- Test 2: Boundary Edge on a face ---
        edge_idx_face = coord_to_edge(s, 2, 3, 2, Z_ALIGN)
        @test f_const[edge_idx_face] ≈ C_z * dz(s)

        # --- Test 3: Boundary Edge on a corner ---
        edge_idx_corner = coord_to_edge(s, 1, 1, 1, Y_ALIGN)
        @test f_const[edge_idx_corner] ≈ C_y * dy(s)

        # Varying vector field
        boid_indices = FT.(1:nboids(s))
        X = boid_indices
        Y = 2 .* boid_indices
        Z = 3 .* boid_indices

        f = flat_dp(s, X, Y, Z)

        # --- Test 1: Interior Edge ---
        edge_idx = coord_to_edge(s, 2, 2, 2, X_ALIGN)
        b_indices, b_valid = edge_boids(s, 2, 2, 2, X_ALIGN)
        @test all(b_valid)
        
        avg_X = (X[b_indices[1]] + X[b_indices[2]] + X[b_indices[3]] + X[b_indices[4]]) / 4.0
        @test f[edge_idx] ≈ avg_X * dx(s)

        # --- Test 2: Boundary Edge (on a face) ---
        edge_idx = coord_to_edge(s, 2, 3, 2, Z_ALIGN) # y=3 is boundary for edge
        b_indices, b_valid = edge_boids(s, 2, 3, 2, Z_ALIGN)
        @test count(b_valid) == 2
        
        avg_Z = (Z[b_indices[1]] + Z[b_indices[2]]) / 2.0
        @test f[edge_idx] ≈ avg_Z * dz(s)
        
        # --- Test 3: Boundary Edge (on a corner) ---
        edge_idx = coord_to_edge(s, 1, 1, 1, Y_ALIGN)
        b_indices, b_valid = edge_boids(s, 1, 1, 1, Y_ALIGN)
        @test count(b_valid) == 1
        
        avg_Y = Y[b_indices[4]] / 1.0
        @test f[edge_idx] ≈ avg_Y * dy(s)
    end

end

