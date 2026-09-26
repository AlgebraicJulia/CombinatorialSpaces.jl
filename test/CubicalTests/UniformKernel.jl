module TestUniformKernel

using Test
using SparseArrays
using Random
using CombinatorialSpaces
using LinearAlgebra

Random.seed!(1234)

@testset "Matrix DEC Operators" begin

  s = UniformCubicalComplex2D(5, 5, 1.0, 1.0)

  d0 = exterior_derivative(Val(0), s)
  d1 = exterior_derivative(Val(1), s)

  @test size(d0) == (ne(s), nv(s))
  @test size(d1) == (nquads(s), ne(s))

  # exactness: d1 * d0 == 0
  @test all(d1 * d0 .== 0)

  # derivatives of constant fields are zero
  @test all(d0 * ones(nv(s)) .== 0)
  @test all(d1 * ones(ne(s)) .== 0)

  # dual derivatives are transposes (with sign where implemented)
  @test dual_derivative(Val(0), s) == transpose(d1)
  @test dual_derivative(Val(1), s) == -transpose(d0)

  # Hodge stars
  hs0 = hodge_star(Val(0), s)
  @test diag(hs0) == map(dq -> dual_quad_area(s, dq), vertices(s))

  hs1 = hodge_star(Val(1), s)
  e_lens = map(e -> edge_len(s, e), edges(s))
  de_lens = map(de -> dual_edge_len(s, de), edges(s))
  @test diag(hs1) == de_lens ./ e_lens

  hs2 = hodge_star(Val(2), s)
  @test diag(hs2) == fill(1 / quad_area(s), nquads(s))

  # inv_hodge_star should invert the diagonal Hodge
  ihs0 = inv_hodge_star(Val(0), s)
  @test all(abs.(diag(ihs0 * hs0) .- 1) .< 1e-12)

  ihs1 = inv_hodge_star(Val(1), s)
  @test all(abs.(-diag(ihs1 * hs1) .- 1) .< 1e-12)

  ihs2 = inv_hodge_star(Val(2), s)
  @test all(abs.(diag(ihs2 * hs2) .- 1) .< 1e-12)

  # Codifferentials: check sizes and definitions
  cd1 = codifferential(Val(1), s)
  @test size(cd1) == (nv(s), ne(s))

  cd2 = codifferential(Val(2), s)
  @test size(cd2) == (ne(s), nquads(s))

  d_cd1 = dual_codifferential(Val(1), s)
  @test size(d_cd1) == (nquads(s), ne(s))

  d_cd2 = dual_codifferential(Val(2), s)
  @test size(d_cd2) == (ne(s), nv(s))

  # Laplacians: shapes and basic properties
  L0 = laplacian(Val(0), s)
  L1 = laplacian(Val(1), s)
  L2 = laplacian(Val(2), s)

  @test size(L0) == (nv(s), nv(s))
  @test size(L1) == (ne(s), ne(s))
  @test size(L2) == (nquads(s), nquads(s))

  # Laplacian of constant should be (near) zero for interior-preserving grid
  @test all(abs.(L0 * ones(nv(s))) .< 1e-12)

  out = wedge_product(Val(1), Val(1), s, ones(ne(s)), ones(ne(s)))
  @test all(out .== 0.0)

  # wedge of dx and dy is a constant 2-form
  V = ones(ne(s))
  V[1:nxedges(s)] .= 0 # No horizontal motion on the vertical edges

  W = ones(ne(s))
  W[end-nxedges(s)+1:end] .= 0 # No vertical motion on the horizontal edges

  # W is dx and V is dy, so their wedge should be 1 everywhere
  out = wedge_product(Val(1), Val(1), s, W, V)
  @test all(out .== 1.0)

  out = wedge_product(Val(1), Val(1), s, V, W)
  @test all(out .== -1.0)

  u = ones(ne(s))
  X, Y = sharp_dd(s, u)

  @test X[coord_to_quad(s, 2, 2)] == -4.0
  @test X[coord_to_quad(s, 3, 2)] == -4.0

  @test Y[coord_to_quad(s, 2, 2)] == 4.0
  @test Y[coord_to_quad(s, 2, 3)] == 4.0

  X = 4 * ones(nquads(s))
  Y = 4 * ones(nquads(s))
  u = flat_dp(s, X, Y)
  @test all(u .== 1.0)

  dd0 = dual_derivative(Val(0), s)
  dd1 = dual_derivative(Val(1), s)

  boundary_idxs = findall(x -> x != 0, dd0 * ones(nquads(s)))

  u = ones(ne(s))
  u[boundary_idxs] .= 0.5
  X, Y = sharp_dd(s, u)
  v = flat_dp(s, X, Y)

  @test all(v[1:nxedges(s)] .== -1.0)
  @test all(v[nxedges(s)+1:end] .== 1.0)

  d_beta = 0.5 * abs.(dd1) * spdiagm(dd0 * ones(nquads(s)));
  u = zeros(ne(s))
  v = zeros(ne(s))
  u[1] = 1.0; u[nxedges(s)+1] = 1.0
  v[1] = 2.0; v[nxedges(s)+1] = -2.0
  @test (dd1 * u + d_beta * v)[1] == 4.0

  u[nxedges(s)] = -1.0; u[end] = -1.0
  v[nxedges(s)] = -2.0; v[end] = 2.0
  @test (dd1 * u + d_beta * v)[end] == 4.0

  u = ones(Float64, ne(s))
  f = ones(Float64, nquads(s))
  res = zeros(Float64, ne(s))

  res = wedge_product_dd(Val(0), Val(1), s, f, u)
  @test all(res .== 1.0)

  res = wedge_product_dd(Val(0), Val(1), s, f, 5 * u)
  @test all(res .== 5.0)

  res = wedge_product_dd(Val(0), Val(1), s, 2 * f, 5 * u)
  @test all(res .== 10.0)

  f = [Float64(y) for y in 1:nyq(s) for x in 1:nxq(s)]
  res = wedge_product_dd(Val(0), Val(1), s, f, u)
  @test xedges(s, res)[1] == 1.0
  @test xedges(s, res)[end] == 4.0

  @test yedges(s, res)[1] == 1.0
  @test yedges(s, res)[end] == 4.0
end

@testset "Matrix DEC Operators with Halo" begin
  s = UniformCubicalComplex2D(5, 5, 1.0, 1.0; halo_x = 1, halo_y = 1)

  d0 = exterior_derivative(Val(0), s)
  d1 = exterior_derivative(Val(1), s)

  @test size(d0) == (ne(s), nv(s))
  @test size(d1) == (nquads(s), ne(s))

  # exactness: d1 * d0 == 0
  @test all(d1 * d0 .== 0)

  # derivatives of constant fields are zero
  @test all(d0 * ones(nv(s)) .== 0)
  @test all(d1 * ones(ne(s)) .== 0)

  # dual derivatives are transposes (with sign where implemented)
  @test dual_derivative(Val(0), s) == transpose(d1)
  @test dual_derivative(Val(1), s) == -transpose(d0)

  # Hodge stars
  hs0 = hodge_star(Val(0), s)
  @test diag(hs0) == map(dq -> dual_quad_area(s, dq), vertices(s))

  hs1 = hodge_star(Val(1), s)
  e_lens = map(e -> edge_len(s, e), edges(s))
  de_lens = map(de -> dual_edge_len(s, de), edges(s))
  @test diag(hs1) == de_lens ./ e_lens

  hs2 = hodge_star(Val(2), s)
  @test diag(hs2) == fill(1 / quad_area(s), nquads(s))

  # inv_hodge_star should invert the diagonal Hodge
  ihs0 = inv_hodge_star(Val(0), s)
  @test all(abs.(diag(ihs0 * hs0) .- 1) .< 1e-12)

  ihs1 = inv_hodge_star(Val(1), s)
  @test all(abs.(diag(-ihs1 * hs1) .- 1) .< 1e-12)

  ihs2 = inv_hodge_star(Val(2), s)
  @test all(abs.(diag(ihs2 * hs2) .- 1) .< 1e-12)

  # Codifferentials: check sizes and definitions
  cd1 = codifferential(Val(1), s)
  @test size(cd1) == (nv(s), ne(s))

  cd2 = codifferential(Val(2), s)
  @test size(cd2) == (ne(s), nquads(s))

  # Laplacians: shapes and basic properties
  L0 = laplacian(Val(0), s)
  L1 = laplacian(Val(1), s)
  L2 = laplacian(Val(2), s)

  @test size(L0) == (nv(s), nv(s))
  @test size(L1) == (ne(s), ne(s))
  @test size(L2) == (nquads(s), nquads(s))

  # Laplacian of constant should be (near) zero for interior-preserving grid
  @test all(abs.(L0 * ones(nv(s))) .< 1e-12)
end

@testset "Kernel DEC Operators" begin
  s = UniformCubicalComplex2D(5, 5, 1.0, 1.0)

  dx_form = vcat(ones(nxedges(s)), zeros(nyedges(s)))  # 1 on x-edges, 0 on y-edges
  dy_form = vcat(zeros(nxedges(s)), ones(nyedges(s)))  # 0 on x-edges, 1 on y-edges

  d0_mat = exterior_derivative(Val(0), s)
  d1_mat = exterior_derivative(Val(1), s)

  res_ne = zeros(ne(s))
  res_nq = zeros(nquads(s))

  # d0

  # derivative of constant is exactly zero
  exterior_derivative!(res_ne, Val(0), s, ones(nv(s)))
  @test all(res_ne .== 0)

  # matches matrix on a random field
  f0 = rand(nv(s))
  exterior_derivative!(res_ne, Val(0), s, f0)
  @test res_ne ≈ d0_mat * f0

  # d1

  # derivative of constant is exactly zero
  exterior_derivative!(res_nq, Val(1), s, ones(ne(s)))
  @test all(res_nq .== 0)

  # matches matrix on a random field
  f1 = rand(ne(s))
  exterior_derivative!(res_nq, Val(1), s, f1)
  @test res_nq ≈ d1_mat * f1

  # exactness: d1(d0(f)) = 0
  exterior_derivative!(res_ne, Val(0), s, f0)
  exterior_derivative!(res_nq, Val(1), s, res_ne)
  @test all(res_nq .== 0)

  # wedge_product 1∧1

  # self-wedge is exactly zero (same expression subtracted from itself)
  @test all(wedge_product(Val(1), Val(1), s, f1, f1) .== 0)

  # antisymmetry
  a = rand(ne(s)); b = rand(ne(s))
  @test wedge_product(Val(1), Val(1), s, a, b) ≈ -wedge_product(Val(1), Val(1), s, b, a)

  # dx ∧ dy = 1 everywhere; dy ∧ dx = -1; same-form wedge = 0
  @test all(wedge_product(Val(1), Val(1), s, dx_form, dy_form) .== 1.0)
  @test all(wedge_product(Val(1), Val(1), s, dy_form, dx_form) .== -1.0)
  @test all(wedge_product(Val(1), Val(1), s, ones(ne(s)), ones(ne(s))) .== 0.0)

  # wedge_product 0∧1

  # constant 0-form c: (c + c)/2 * a = c * a exactly
  @test wedge_product(Val(0), Val(1), s, 3.0 * ones(nv(s)), f1) ≈ 3.0 * f1

  # 0 ∧ 1 and 1 ∧ 0 dispatches are equal (0-forms commute through wedge)
  @test wedge_product(Val(0), Val(1), s, f0, f1) ≈ wedge_product(Val(1), Val(0), s, f1, f0)

  # zero 0-form gives zero result
  @test all(wedge_product(Val(0), Val(1), s, zeros(nv(s)), f1) .== 0)

  # wedge_product_dd 0∧1

  u  = ones(Float64, ne(s))
  f2 = ones(Float64, nquads(s))

  # constant unit fields → result is all ones
  @test all(wedge_product_dd(Val(0), Val(1), s, f2, u) .== 1.0)

  # linearity in both arguments
  @test all(wedge_product_dd(Val(0), Val(1), s, 2 * f2, 5 * u) .== 10.0)

  # linearly varying dual 0-form: boundary edges pick one neighbour, interior average two
  f_vary = [Float64(y) for y in 1:nyq(s) for x in 1:nxq(s)]
  res_dd = wedge_product_dd(Val(0), Val(1), s, f_vary, u)
  @test xedges(s, res_dd)[1]   == 1.0
  @test xedges(s, res_dd)[end] == 4.0
  @test yedges(s, res_dd)[1]   == 1.0
  @test yedges(s, res_dd)[end] == 4.0

  # sharp_dd

  # known values from uniform field
  u_ones = ones(ne(s))
  X, Y = sharp_dd(s, u_ones)
  @test X[coord_to_quad(s, 2, 2)] == -4.0
  @test X[coord_to_quad(s, 3, 2)] == -4.0
  @test Y[coord_to_quad(s, 2, 2)] ==  4.0
  @test Y[coord_to_quad(s, 2, 3)] ==  4.0

  # sharp of zero field is zero
  X0, Y0 = sharp_dd(s, zeros(ne(s)))
  @test all(X0 .== 0) && all(Y0 .== 0)

  # linearity: sharp(c * u) = c * sharp(u)
  X2, Y2 = sharp_dd(s, 2.0 * u_ones)
  @test X2 ≈ 2.0 * X && Y2 ≈ 2.0 * Y

  # flat_dp

  # uniform dual vector field (4, 4): flat_dp should recover all-ones 1-form
  Xc = 4.0 * ones(nquads(s)); Yc = 4.0 * ones(nquads(s))
  @test all(flat_dp(s, Xc, Yc) .== 1.0)

  # flat_dp of zero is zero
  @test all(flat_dp(s, zeros(nquads(s)), zeros(nquads(s))) .== 0)

  # linearity
  Xr = rand(nquads(s)); Yr = rand(nquads(s))
  @test flat_dp(s, 3.0 * Xr, 3.0 * Yr) ≈ 3.0 * flat_dp(s, Xr, Yr)

  # flat_dd

  # flat_dd of zero is zero
  @test all(flat_dd(s, zeros(nquads(s)), zeros(nquads(s))) .== 0)

  # linearity
  @test flat_dd(s, 3.0 * Xr, 3.0 * Yr) ≈ 3.0 * flat_dd(s, Xr, Yr)

  # X-aligned edges use Y component; Y-aligned edges use -X component.
  # Interior x-aligned edge at (1, 2): dual_edge_len = dy(s); result = dy(s).
  Y_ones = ones(nquads(s)); X_zeros = zeros(nquads(s))
  u_dd = flat_dd(s, X_zeros, Y_ones)
  @test u_dd[coord_to_edge(s, 1, 2, X_ALIGN)] ≈ dy(s)   # interior x-edge: Y component
  @test u_dd[coord_to_edge(s, 2, 1, Y_ALIGN)] ≈ 0.0     # interior y-edge: uses -X = 0

  X_ones = ones(nquads(s)); Y_zeros = zeros(nquads(s))
  u_dd2 = flat_dd(s, X_ones, Y_zeros)
  @test u_dd2[coord_to_edge(s, 2, 1, Y_ALIGN)] ≈ -dx(s) # interior y-edge: -X component
  @test u_dd2[coord_to_edge(s, 1, 2, X_ALIGN)] ≈ 0.0    # interior x-edge: uses Y = 0
end

@testset "Dual-Dual Wedge Product 1-1" begin
    FT = Float64
    s = UniformCubicalComplex2D(7, 6, FT(3), FT(2))

    dx̃ = zeros(FT, ne(s))
    dỹ = zeros(FT, ne(s))

    for e in edges(s)
        _, _, align = edge_to_coord(s, e)
        dual = dual_edge_len(s, e)

        if align == X_ALIGN
            dỹ[e] = dual
        else
            dx̃[e] = -dual
        end
    end

    expected_area = FT[
        dual_quad_area(s, v) for v in vertices(s)
    ]

    wedge_u(a, b) =
        wedge_product_dd(Val(1), Val(1), s, a, b)

    atol = 100eps(FT)
    rtol = 100eps(FT)

    @testset "Basis Forms and Dual-Cell Area" begin
        @test wedge_u(dx̃, dỹ) ≈ expected_area atol=atol rtol=rtol
        @test wedge_u(dỹ, dx̃) ≈ -expected_area atol=atol rtol=rtol
    end

    Random.seed!(1234)
    a = randn(FT, ne(s))
    b = randn(FT, ne(s))
    c = randn(FT, ne(s))
    α = FT(1.7)
    β = FT(-0.4)

    @testset "Antisymmetry" begin
        @test wedge_u(a, b) ≈ -wedge_u(b, a) atol=atol rtol=rtol

        @test wedge_u(a, a) ≈ zeros(FT, nv(s)) atol=atol
    end

    @testset "Bilinearity" begin
        @test wedge_u(α .* a .+ β .* b, c) ≈
              α .* wedge_u(a, c) .+ β .* wedge_u(b, c) atol=atol rtol=rtol

        @test wedge_u(c, α .* a .+ β .* b) ≈
              α .* wedge_u(c, a) .+ β .* wedge_u(c, b) atol=atol rtol=rtol
    end
end

@testset "Primal-Dual Wedge Product 1-1" begin
    # single-quad mesh: all edges are boundary, weight 1
    s1 = UniformCubicalComplex(2, 2, 1.0, 1.0)
    ne1 = ne(s1)

    # Basis contributions on single-quad mesh (all boundary, weight 1):
    #   e1 (bottom-x): +1,  e3 (top-x): +1
    #   e2 (right-y):  -1,  e4 (left-y): -1
    x1, y1 = quad_to_coord(s1, 1)
    e1, e2, e3, e4 = quad_edges(s1, x1, y1)

    a = zeros(ne1); b = zeros(ne1)
    a[1] = 1.0; b[1] = 1.0
    a[2] = 1.0; b[2] = -1.0
    a[3] = 1.0; b[3] = 1.0
    a[4] = 1.0; b[4] = -1.0
    r = wedge_product_pd(Val(1), Val(1), s1, a, b)
    @test r[1] ≈ 0.0

    for (e, expected) in [(e1, +0.25), (e2, -0.25), (e3, +0.25), (e4, -0.25)]
        a = zeros(ne1); b = zeros(ne1)
        a[e] = 1.0; b[e] = 1.0
        r = wedge_product_pd(Val(1), Val(1), s1, a, b)
        @test r[1] ≈ expected
    end

    # multi-quad mesh: interior edges weight 0.5, boundary 1
    s2 = UniformCubicalComplex(3, 3, 1.0, 1.0)
    ne2 = ne(s2)

    # Interior X-aligned edge: contributes +0.5 to quad above and quad below
    interior_x = coord_to_edge(s2, 1, 2, X_ALIGN)
    a2 = zeros(ne2); b2 = zeros(ne2)
    a2[interior_x] = 1.0; b2[interior_x] = 1.0
    r2 = wedge_product_pd(Val(1), Val(1), s2, a2, b2)
    q_above = coord_to_quad(s2, 1, 2)
    q_below = coord_to_quad(s2, 1, 1)
    @test r2[q_above] ≈ 0.5
    @test r2[q_below] ≈ 0.5

    # Interior Y-aligned edge: contributes -0.5 to quad left and quad right
    interior_y = coord_to_edge(s2, 2, 1, Y_ALIGN)
    a3 = zeros(ne2); b3 = zeros(ne2)
    a3[interior_y] = 1.0; b3[interior_y] = 1.0
    r3 = wedge_product_pd(Val(1), Val(1), s2, a3, b3)
    q_left  = coord_to_quad(s2, 1, 1)
    q_right = coord_to_quad(s2, 2, 1)
    @test r3[q_left]  ≈ -0.5
    @test r3[q_right] ≈ -0.5

    # Boundary X-aligned edge: contributes +1 to its single adjacent quad
    boundary_x = coord_to_edge(s2, 1, 1, X_ALIGN)
    a4 = zeros(ne2); b4 = zeros(ne2)
    a4[boundary_x] = 1.0; b4[boundary_x] = 1.0
    r4 = wedge_product_pd(Val(1), Val(1), s2, a4, b4)
    q_adj = coord_to_quad(s2, 1, 1)
    @test r4[q_adj] ≈ 0.25

    # Boundary Y-aligned edge: contributes -1 to its single adjacent quad
    boundary_y = coord_to_edge(s2, 1, 1, Y_ALIGN)
    a5 = zeros(ne2); b5 = zeros(ne2)
    a5[boundary_y] = 1.0; b5[boundary_y] = 1.0
    r5 = wedge_product_pd(Val(1), Val(1), s2, a5, b5)
    q_adj5 = coord_to_quad(s2, 1, 1)
    @test r5[q_adj5] ≈ -0.25

    # Forms on different edges contribute zero
    a6 = zeros(ne2); b6 = zeros(ne2)
    a6[interior_x] = 1.0; b6[interior_y] = 1.0
    r6 = wedge_product_pd(Val(1), Val(1), s2, a6, b6)
    @test all(iszero, r6)

    # Bilinearity
    a7 = rand(ne2); b7 = rand(ne2); a8 = rand(ne2)
    r_sum = wedge_product_pd(Val(1), Val(1), s2, a7 .+ a8, b7)
    r_a7  = wedge_product_pd(Val(1), Val(1), s2, a7, b7)
    r_a8  = wedge_product_pd(Val(1), Val(1), s2, a8, b7)
    @test r_sum ≈ r_a7 .+ r_a8

    # Float32 type stability
    s2_f32 = UniformCubicalComplex2D(3, 3, 1.0f0, 1.0f0)
    a_f32 = rand(Float32, ne(s2_f32))
    b_f32 = rand(Float32, ne(s2_f32))
    r_f32 = wedge_product_pd(Val(1), Val(1), s2_f32, a_f32, b_f32)
    @test eltype(r_f32) == Float32

    # Output length equals nquads
    @test length(r2) == nquads(s2)
end

end
