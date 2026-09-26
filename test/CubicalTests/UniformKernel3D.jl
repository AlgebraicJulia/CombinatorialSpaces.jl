module TestUniformKernel3D

using Test
using KernelAbstractions
using LinearAlgebra
using Random
using CombinatorialSpaces

Random.seed!(1234)

@testset "Exterior Derivative Kernels" begin
    s = UniformCubicalComplex3D(2, 2, 2, 1.0, 1.0, 1.0)
    FT = Float64

    @testset "d0" begin
        f = fill(FT(5.0), nv(s))
        @test all(exterior_derivative(Val(0), s, f) .== 0.0)

        f = zeros(FT, nv(s))
        for z in 1:nz(s), y in 1:ny(s), x in 1:nx(s)
            f[coord_to_vert(s, x, y, z)] = FT(x + y + z)
        end

        result = exterior_derivative(Val(0), s, f)

        # nxe(s) = 1, nye(s) = 1, nze(s) = 1 on a 2×2×2 mesh
        for z in 1:nz(s), y in 1:ny(s), x in 1:nxe(s)
            @test result[coord_to_edge(s, x, y, z, X_ALIGN)] ≈ FT(1.0)
        end
        for z in 1:nz(s), y in 1:nye(s), x in 1:nx(s)
            @test result[coord_to_edge(s, x, y, z, Y_ALIGN)] ≈ FT(1.0)
        end
        for z in 1:nze(s), y in 1:ny(s), x in 1:nx(s)
            @test result[coord_to_edge(s, x, y, z, Z_ALIGN)] ≈ FT(1.0)
        end
    end

    @testset "d1" begin
        f = fill(FT(5.0), ne(s))
        @test all(exterior_derivative(Val(1), s, f) .== 0.0)

        # f[e] = edge index; verify d1 = signed sum on every quad
        f = zeros(FT, ne(s))
        f[1] = 1; f[2] = 5; f[5] = 5; f[6] = 6;
        f[3] = 5; f[9] = 5; f[10] = 6;
        f[7] = 1; f[11] = 2;
        result = exterior_derivative(Val(1), s, f)

        @test result[coord_to_quad(s, 1, 1, 1, Z_ALIGN)] ≈ FT(-3.0)
        @test result[coord_to_quad(s, 1, 1, 1, Y_ALIGN)] ≈ FT(3.0)
        @test result[coord_to_quad(s, 1, 1, 1, X_ALIGN)] ≈ FT(1.0)
    end

    @testset "d2" begin
        f = fill(FT(5.0), nquads(s))
        @test all(exterior_derivative(Val(2), s, f) .== 0.0)

        f = FT.(1:nquads(s))
        result = exterior_derivative(Val(2), s, f)
        b = coord_to_boid(s, 1, 1, 1)
        @test result[b] ≈ FT(3.0)
    end

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
        f0 = FT[1, 2, 3, 4, 5, 6, 7, 8]
        star_f0 = hodge_star(Val(0), s, f0)
        expected_star_f0 = f0 .* 3.0
        @test star_f0 ≈ expected_star_f0

        s_333 = UniformCubicalComplex3D(3, 3, 3, 2.0, 3.0, 4.0)
        f0_center = zeros(FT, nv(s_333)); f0_center[14] = 1.0; # Center vertex
        star_f0_center = hodge_star(Val(0), s_333, f0_center)

        @test star_f0_center[14] ≈ 1.0 * (1.0*1.5*2.0) # interior dual boid volume

        f1 = ones(FT, ne(s))
        star_f1 = hodge_star(Val(1), s, f1)
        @test star_f1[1:4] ≈ ones(4) .* 1.5
        @test star_f1[5:8] ≈ ones(4) .* (2/3)
        @test star_f1[9:12] ≈ ones(4) .* 0.375

        f2 = ones(FT, nquads(s))
        star_f2 = hodge_star(Val(2), s, f2)
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
        f0_dual = FT[i for i in boids(s)]
        f1_dual = dual_derivative(Val(0), s, f0_dual)

        # Interior only
        @test all(f1_dual[5:8] .== 4)

        @test all(f1_dual[15:16] .== 2)
        @test all(f1_dual[21:22] .== 2)

        @test all(f1_dual[[26, 29, 32, 35]] .== 1)

        # XY-quads (Z_ALIGN)
        f1_dual_xy = zeros(nquads(s))
        f1_dual_xy[5] = 1.0; f1_dual_xy[[6,7]] .= 2.0; f1_dual_xy[8] = 3.0
        f2_dual_xy = dual_derivative(Val(1), s, f1_dual_xy)
        @test f2_dual_xy[9] == 1.0
        @test f2_dual_xy[10] == 1.0

        @test f2_dual_xy[26] == -1.0
        @test f2_dual_xy[29] == -1.0

        # XZ-quads (Y_ALIGN)
        f1_dual_xz = zeros(FT, nquads(s))
        f1_dual_xz[15] = 1.0; f1_dual_xz[[16, 21]] .= 2.0; f1_dual_xz[22] = 3.0
        f2_dual_xz = dual_derivative(Val(1), s, f1_dual_xz)

        @test f2_dual_xz[9] == -1.0
        @test f2_dual_xz[10] == -1.0

        @test f2_dual_xz[41] == 1.0
        @test f2_dual_xz[50] == 1.0

        # YZ-quads (X_ALIGN)
        f1_dual_yz = zeros(FT, nquads(s))
        f1_dual_yz[26] = 1.0; f1_dual_yz[[29, 32]] .= 2.0; f1_dual_yz[35] = 3.0
        f2_dual_yz = dual_derivative(Val(1), s, f1_dual_yz)

        @test f2_dual_yz[26] == 1.0
        @test f2_dual_yz[29] == 1.0

        @test f2_dual_yz[41] == -1.0
        @test f2_dual_yz[50] == -1.0

        f = zeros(FT, ne(s))

        e_low  = coord_to_edge(s, 1, 2, 2, X_ALIGN)
        e_high = coord_to_edge(s, 2, 2, 2, X_ALIGN)
        f[e_low]  = FT(3.0)
        f[e_high] = FT(5.0)

        result = dual_derivative(Val(2), s, f)
        v_int  = coord_to_vert(s, 2, 2, 2)

        @test result[v_int] ≈ -2.0

        f = zeros(FT, ne(s))

        e_low  = coord_to_edge(s, 2, 1, 2, Y_ALIGN)
        e_high = coord_to_edge(s, 2, 2, 2, Y_ALIGN)
        f[e_low]  = FT(2.0)
        f[e_high] = FT(7.0)

        result = dual_derivative(Val(2), s, f)
        v_int  = coord_to_vert(s, 2, 2, 2)

        @test result[v_int] ≈ -5.0

        f = zeros(FT, ne(s))

        e_low  = coord_to_edge(s, 2, 2, 1, Z_ALIGN)
        e_high = coord_to_edge(s, 2, 2, 2, Z_ALIGN)
        f[e_low]  = FT(4.0)
        f[e_high] = FT(9.0)

        result = dual_derivative(Val(2), s, f)
        v_int  = coord_to_vert(s, 2, 2, 2)

        @test result[v_int] ≈ -5.0

        f = zeros(FT, ne(s))

        f[coord_to_edge(s, 1, 2, 2, X_ALIGN)] = FT(1.0)
        f[coord_to_edge(s, 2, 2, 2, X_ALIGN)] = FT(2.0)
        f[coord_to_edge(s, 2, 1, 2, Y_ALIGN)] = FT(3.0)
        f[coord_to_edge(s, 2, 2, 2, Y_ALIGN)] = FT(4.0)
        f[coord_to_edge(s, 2, 2, 1, Z_ALIGN)] = FT(5.0)
        f[coord_to_edge(s, 2, 2, 2, Z_ALIGN)] = FT(6.0)

        result = dual_derivative(Val(2), s, f)
        v_int  = coord_to_vert(s, 2, 2, 2)

        @test result[v_int] ≈ -3.0
    end

    @testset "dd == 0" begin
        f0_dual = rand(FT, nboids(s))
        f1_dual = dual_derivative(Val(0), s, f0_dual)
        f2_dual = dual_derivative(Val(1), s, f1_dual)
        @test all(isapprox.(f2_dual, 0, atol=1e-12))

        f1_dual = rand(FT, nquads(s))
        f2_dual = dual_derivative(Val(1), s, f1_dual)
        f3_dual = dual_derivative(Val(2), s, f2_dual)
        @test all(isapprox.(f3_dual, 0, atol=1e-12))

        # Exactness on asymmetric and halo-padded meshes, to catch axis-specific
        # ordering bugs (e.g. in edge_quads) that a cubic no-halo mesh could hide.
        for s2 in (UniformCubicalComplex3D(4, 5, 6, 1.0, 2.0, 3.0),
                   UniformCubicalComplex3D(3, 4, 5, 1.0, 1.0, 1.0; halo_x = 1, halo_y = 1, halo_z = 1))
            g0 = rand(FT, nboids(s2))
            g1 = dual_derivative(Val(0), s2, g0)
            g2 = dual_derivative(Val(1), s2, g1)
            @test all(isapprox.(g2, 0, atol=1e-12))

            h1 = rand(FT, nquads(s2))
            h2 = dual_derivative(Val(1), s2, h1)
            h3 = dual_derivative(Val(2), s2, h2)
            @test all(isapprox.(h3, 0, atol=1e-12))
        end
    end
end

@testset "Wedge Product Kernels" begin
    s = UniformCubicalComplex3D(3, 3, 3, 1.0, 1.0, 1.0)
    FT = Float64

    @testset "Wedge 1-1" begin
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
        @test all(isapprox.(xzquads(s, w11_xz), -8.0, atol=1e-12))
        @test all(isapprox.(yzquads(s, w11_xz), 0.0, atol=1e-12))

        w11_xz = wedge_product(Val(1), Val(1), s, g1, f1)
        @test all(isapprox.(xyquads(s, w11_xz), 0.0, atol=1e-12))
        @test all(isapprox.(xzquads(s, w11_xz), 8.0, atol=1e-12))
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

    @testset "Wedge 1-2" begin
        # Each edge component pairs with the quad family holding its complement:
        # dx∧dydz, dy∧dzdx, dz∧dxdy all give +dxdydz, so every term adds.
        # 2 * 3 * 3 = 18
        f1 = ones(FT, ne(s)) .* 2.0
        g2 = ones(FT, nquads(s)) .* 3.0
        w12 = wedge_product(Val(1), Val(2), s, f1, g2)
        @test all(isapprox.(w12, 18.0, atol=1e-12))

        # 2 * 3 + 4 * 5 + 6 * 7 = 6 + 20 + 42 = 68
        f1 = zeros(FT, ne(s))
        xedges(s, f1) .= 2.0
        yedges(s, f1) .= 4.0
        zedges(s, f1) .= 6.0

        g2 = zeros(FT, nquads(s))
        xyquads(s, g2) .= 7.0   # Z_ALIGN, holds the dxdy component
        xzquads(s, g2) .= 5.0   # Y_ALIGN, holds the dzdx component
        yzquads(s, g2) .= 3.0   # X_ALIGN, holds the dydz component

        w12 = wedge_product(Val(1), Val(2), s, f1, g2)
        @test all(isapprox.(w12, 68.0, atol=1e-12))
    end

    @testset "Wedge 1-1 and 1-2 share the dxdy/dzdx/dydz basis" begin
        # Wedge 1-1 writes the XZ-quad (Y_ALIGN) component as dz∧dx, so wedge
        # 1-2 must read it back as dz∧dx too. Chaining them gives the triple
        # product: a ∧ (b ∧ c) = det[a b c]. A dxdz/dzdx mismatch in either
        # kernel flips the sign of the middle column's cofactor and breaks this.
        oneform(v) = (f = zeros(FT, ne(s));
                      xedges(s, f) .= v[1];
                      yedges(s, f) .= v[2];
                      zedges(s, f) .= v[3];
                      f)
        triple(a, b, c) = wedge_product(Val(1), Val(2), s, oneform(a),
                                        wedge_product(Val(1), Val(1), s, oneform(b), oneform(c)))

        # Right-handed basis: x ∧ (y ∧ z) = +1
        @test all(isapprox.(triple([1,0,0], [0,1,0], [0,0,1]), 1.0, atol=1e-12))
        # Each cyclic rotation is also +1, the swap is -1
        @test all(isapprox.(triple([0,1,0], [0,0,1], [1,0,0]), 1.0, atol=1e-12))
        @test all(isapprox.(triple([0,0,1], [1,0,0], [0,1,0]), 1.0, atol=1e-12))
        @test all(isapprox.(triple([0,1,0], [1,0,0], [0,0,1]), -1.0, atol=1e-12))

        # Generic vectors, against the determinant
        rng = MersenneTwister(0xDEC3D)
        for _ in 1:5
            a, b, c = randn(rng, 3), randn(rng, 3), randn(rng, 3)
            @test all(isapprox.(triple(a, b, c), det(hcat(a, b, c)), atol=1e-12))
        end
    end
end

@testset "Dual 0-form advection" begin
    # Advecting a dual 0-form by the Lie derivative routes through wedge 1-2:
    #
    #   L_v ρ = ⋆₃ ( v ∧ ⋆₂⁻¹ d̃₀ ρ )
    #
    # For constant v and linear ρ this is exactly v · ∇ρ, with no discretization
    # error, so a sign flip on any component shows up as an exact negation.
    s = UniformCubicalComplex3D(6, 6, 6, 1.0, 1.0, 1.0)
    FT = Float64

    lie_dual0(v, ρ) =
        hodge_star(Val(3), s,
            wedge_product(Val(1), Val(2), s, v,
                inv_hodge_star(Val(2), s,
                    dual_derivative(Val(0), s, ρ))))

    # Constant vector field as a primal 1-form: its integral along each edge.
    function const_velocity(vx, vy, vz)
        v = zeros(FT, ne(s))
        for e in 1:ne(s)
            _, _, _, align = edge_to_coord(s, e)
            c = align == X_ALIGN ? vx : align == Y_ALIGN ? vy : vz
            v[e] = c * edge_len(s, align)
        end
        v
    end

    # Boids whose stencil stays clear of the mesh boundary, where the dual
    # derivative loses a neighbour and the dual edge length is halved.
    interior = [coord_to_boid(s, x, y, z)
                for x in 2:(nxb(s) - 1), y in 2:(nyb(s) - 1), z in 2:(nzb(s) - 1)][:]

    grad = [0.7, -1.3, 0.45]
    ρ = [sum(grad .* dual_point(s, boid_to_coord(s, b)...)) for b in 1:nboids(s)]

    for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0.4, -2.0, 1.5])
        @test all(isapprox.(lie_dual0(const_velocity(v...), ρ)[interior],
                            sum(v .* grad), atol=1e-10))
    end

    # A constant field has nothing to advect.
    @test all(isapprox.(lie_dual0(const_velocity(1.0, 1.0, 1.0), ones(FT, nboids(s)))[interior],
                        0.0, atol=1e-12))
end

@testset "Dual Wedge Product Kernels" begin
    s = UniformCubicalComplex3D(3, 3, 3, 1.0, 1.0, 1.0)
    FT = Float64

    @testset "Wedge DD 0-1" begin
        f = ones(FT, nboids(s)) .* 2.0
        a = ones(FT, nquads(s)) .* 3.0
        w01 = wedge_product_dd(Val(0), Val(1), s, f, a)

        # boundary (1 valid boid) or interior (avg of identical values): 2.0 * 3.0 = 6.0
        @test all(w01 .≈ 6.0)

        # boundary case: Z-aligned quad on z=1
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

@testset "Wedge Bilinearity and Antisymmetry" begin
    # On random data, where the axis-aligned checks above cannot see a
    # component-specific slip.
    Random.seed!(1234)
    s = UniformCubicalComplex3D(6, 5, 4, 1.0, 2.0, 3.0;
        halo_west=1, halo_east=2, halo_south=1, halo_north=0, halo_down=2, halo_up=1)

    for FT in (Float32, Float64)
        tol = eps(FT) * 64
        α, β = FT(0.37), FT(-1.21)
        a, b, c = rand(FT, ne(s)), rand(FT, ne(s)), rand(FT, ne(s))
        q = rand(FT, nquads(s))
        f, g = rand(FT, nboids(s)), rand(FT, nboids(s))

        @test isapprox(wedge_product(Val(1), Val(1), s, b, a), -wedge_product(Val(1), Val(1), s, a, b), atol = tol, rtol = tol)
        @test isapprox(wedge_product(Val(1), Val(2), s, α .* a .+ β .* c, q),
                       α .* wedge_product(Val(1), Val(2), s, a, q) .+ β .* wedge_product(Val(1), Val(2), s, c, q), atol = tol, rtol = tol)
        @test isapprox(wedge_product_dd(Val(0), Val(1), s, α .* f .+ β .* g, q),
                       α .* wedge_product_dd(Val(0), Val(1), s, f, q) .+ β .* wedge_product_dd(Val(0), Val(1), s, g, q), atol = tol, rtol = tol)
    end
end

@testset "Sharp and Flat Operators" begin
    @testset "Sharp DD" begin
        s = UniformCubicalComplex3D(4, 4, 4, 1.0, 2.0, 4.0)
        FT = Float64

        f = zeros(FT, nquads(s))

        # Interior boid (2,2,2)
        boid_idx = coord_to_boid(s, 2, 2, 2)
        q_z1, q_z2, q_y1, q_y2, q_x1, q_x2 = boid_quads(s, 2, 2, 2)

        f[q_x1] = 1.0; f[q_x2] = 3.0
        f[q_y1] = 5.0; f[q_y2] = 5.0
        f[q_z1] = 8.0; f[q_z2] = 6.0

        X, Y, Z = sharp_dd(s, f)

        @test X[boid_idx] ≈ 2.0 / dx(s)
        @test Y[boid_idx] ≈ 5.0 / dy(s)
        @test Z[boid_idx] ≈ 7.0 / dz(s)

        # Boundary corner boid (1,1,1)
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

        # Boid on edge (1, 2, nzb(s))
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

        # Interior edge
        edge_idx_int = coord_to_edge(s, 2, 2, 2, X_ALIGN)
        @test f_const[edge_idx_int] ≈ C_x * dx(s)

        # Boundary edge on a face
        edge_idx_face = coord_to_edge(s, 2, 3, 2, Z_ALIGN)
        @test f_const[edge_idx_face] ≈ C_z * dz(s)

        # Boundary edge on a corner
        edge_idx_corner = coord_to_edge(s, 1, 1, 1, Y_ALIGN)
        @test f_const[edge_idx_corner] ≈ C_y * dy(s)

        # Varying vector field
        boid_indices = FT.(1:nboids(s))
        X = boid_indices
        Y = 2 .* boid_indices
        Z = 3 .* boid_indices

        f = flat_dp(s, X, Y, Z)

        # Interior edge
        edge_idx = coord_to_edge(s, 2, 2, 2, X_ALIGN)
        b_indices, b_valid = edge_boids(s, 2, 2, 2, X_ALIGN)
        @test all(b_valid)

        avg_X = (X[b_indices[1]] + X[b_indices[2]] + X[b_indices[3]] + X[b_indices[4]]) / 4.0
        @test f[edge_idx] ≈ avg_X * dx(s)

        # Boundary edge (on a face)
        edge_idx = coord_to_edge(s, 2, 3, 2, Z_ALIGN) # y=3 is boundary for edge
        b_indices, b_valid = edge_boids(s, 2, 3, 2, Z_ALIGN)
        @test count(b_valid) == 2

        avg_Z = (Z[b_indices[1]] + Z[b_indices[2]]) / 2.0
        @test f[edge_idx] ≈ avg_Z * dz(s)

        # Boundary edge (on a corner)
        edge_idx = coord_to_edge(s, 1, 1, 1, Y_ALIGN)
        b_indices, b_valid = edge_boids(s, 1, 1, 1, Y_ALIGN)
        @test count(b_valid) == 1

        avg_Y = Y[b_indices[4]] / 1.0
        @test f[edge_idx] ≈ avg_Y * dy(s)
    end

end

@testset "Dual Laplacian on 0-forms" begin
    FT = Float64

    s = UniformCubicalComplex3D(4, 4, 4, 1.0, 1.0, 1.0)
    dlap_0 = x -> hodge_star(Val(3), s,
                    exterior_derivative(Val(2), s,
                        inv_hodge_star(Val(2), s,
                            dual_derivative(Val(0), s, x))))

    @testset "Constant Field" begin
        f = fill(FT(7.0), nboids(s))
        result = dlap_0(f)
        @test result[coord_to_boid(s, 2, 2, 2)] == 0
    end

    @testset "Linear Field" begin
        f = FT[boid_to_coord(s, b)[1] for b in boids(s)]
        result = dlap_0(f)
        @test result[coord_to_boid(s, 2, 2, 2)] == 0
    end
end

end
