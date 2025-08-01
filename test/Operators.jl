module TestOperators

using Test
using SparseArrays
using LinearAlgebra
using Catlab
using CombinatorialSpaces
using CombinatorialSpaces.Meshes: tri_345, tri_345_false, grid_345, right_scalene_unit_hypot
using CombinatorialSpaces.SimplicialSets: boundary_inds
using CombinatorialSpaces.DiscreteExteriorCalculus: eval_constant_primal_form
using GeometryBasics: Point, QuadFace, MetaMesh
using Random
using StaticArrays: SVector
using Statistics: mean, var, std

import CombinatorialSpaces.Meshes: single_tetrahedron

Random.seed!(0)

function generate_dual_mesh(s::HasDeltaSet1D)
    orient!(s)
    sd = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2d}(s)
    subdivide_duals!(sd, Barycenter())
    sd
end

function generate_dual_mesh(s::HasDeltaSet2D)
    orient!(s)
    sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}(s)
    subdivide_duals!(sd, Barycenter())
    sd
end

function generate_dual_mesh(s::HasDeltaSet3D)
    orient!(s)
    sd = EmbeddedDeltaDualComplex3D{Bool,Float64,Point3d}(s)
    subdivide_duals!(sd, Barycenter())
    sd
end

primal_line = EmbeddedDeltaSet1D{Bool,Point2d}()
add_vertices!(primal_line, 3, point=[Point2d(1,0), Point2d(0,0), Point2d(0,2)])
add_edges!(primal_line, [1,2], [2,3])
line = generate_dual_mesh(primal_line)

primal_cycle = EmbeddedDeltaSet1D{Bool,Point2d}()
add_vertices!(primal_cycle, 3, point=[Point2d(1,0), Point2d(0,0), Point2d(0,1)])
add_edges!(primal_cycle, [1,2,3], [2,3,1])
cycle = generate_dual_mesh(primal_cycle)

primal_plus = EmbeddedDeltaSet1D{Bool,Point2d}()
add_vertices!(primal_plus, 5, point=[Point2d(0,0), Point2d(1,0), Point2d(-1,0), Point2d(0,1), Point2d(0, -1)])
add_edges!(primal_plus, [1,1,3,5], [2,4,1,1])
primal_plus[:edge_orientation] = true
plus = generate_dual_mesh(primal_plus)


dual_meshes_1D = [line, cycle, plus]

dual_meshes_2D = [(generate_dual_mesh ∘ loadmesh ∘ Icosphere).(1:2)...,
               (generate_dual_mesh ∘ loadmesh)(Rectangle_30x10()),
               (generate_dual_mesh).([triangulated_grid(10,10,8,8,Point3d), makeSphere(5, 175, 5, 0, 360, 5, 6371+90)[1]])...,
               (loadmesh)(Torus_30x10())];

dual_meshes_3D = [last(single_tetrahedron()), generate_dual_mesh(parallelepiped())]

tg′ = triangulated_grid(100,100,10,10,Point2d);
tg = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2d}(tg′);
subdivide_duals!(tg, Barycenter());

rect′ = loadmesh(Rectangle_30x10());
rect = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}(rect′);
subdivide_duals!(rect, Barycenter());

flat_meshes = [tri_345()[2], tri_345_false()[2], right_scalene_unit_hypot()[2], grid_345()[2], tg, rect];

tet_msh = tetgen_readme_mesh()
tet_msh_sd = EmbeddedDeltaDualComplex3D{Bool, Float64, Point3d}(tet_msh)
subdivide_duals!(tet_msh_sd, Circumcenter())

@testset "Exterior Derivative" begin
    for i in 0:0
        for sd in dual_meshes_1D
            @test all(dec_differential(i, sd) .== d(i, sd))
        end
    end

    for i in 0:1
        for sd in dual_meshes_2D
            @test all(dec_differential(i, sd) .== d(i, sd))
        end
    end

    for i in 0:2
        for sd in dual_meshes_3D
            @test all(dec_differential(i, sd) .== d(i, sd))
        end
    end
end

@testset "Boundary" begin
    for i in 1:1
        for sd in dual_meshes_1D
            @test all(dec_boundary(i, sd) .== ∂(i, sd))
        end
    end

    for i in 1:2
        for sd in dual_meshes_2D
            @test all(dec_boundary(i, sd) .== ∂(i, sd))
        end
    end

    for i in 1:3
        for sd in dual_meshes_3D
            @test all(dec_boundary(i, sd) .== ∂(i, sd))
        end
    end
end

@testset "Dual Derivative" begin
    for i in 0:0
        for sd in dual_meshes_1D
            @test all(dec_dual_derivative(i, sd) .== dual_derivative(i, sd))
        end
    end

    for i in 0:1
        for sd in dual_meshes_2D
            @test all(dec_dual_derivative(i, sd) .== dual_derivative(i, sd))
        end
    end

    for i in 0:2
        for sd in dual_meshes_3D
            @test all(dec_dual_derivative(i, sd) .== dual_derivative(i, sd))
        end
    end
end

#TODO: For hodge star 1, the values seems to be extremely close yet not quite equal
@testset "Diagonal Hodge" begin
    for i in 0:1
        for sd in dual_meshes_1D
            @test all(isapprox.(dec_hodge_star(Val{i}, sd, DiagonalHodge()), hodge_star(i, sd, DiagonalHodge()); rtol = 1e-12))
        end
    end

    for i in 0:2
        for sd in dual_meshes_2D
            @test all(isapprox.(dec_hodge_star(Val{i}, sd, DiagonalHodge()), hodge_star(i, sd, DiagonalHodge()); rtol = 1e-12))
        end
    end

    for i in 0:3
        for sd in dual_meshes_3D
            @test all(isapprox.(dec_hodge_star(Val{i}, sd, DiagonalHodge()), hodge_star(i, sd, DiagonalHodge()); rtol = 1e-12))
        end
    end
end

#TODO: For inv hodge star 1, the values seems to be extremely close yet not quite equal
@testset "Inverse Diagonal Hodge" begin
    for i in 0:1
        for sd in dual_meshes_1D
            @test all(isapprox.(dec_inv_hodge_star(Val{i}, sd, DiagonalHodge()), inv_hodge_star(i, sd, DiagonalHodge()); rtol = 1e-12))
        end
    end

    for i in 0:2
        for sd in dual_meshes_2D
            @test all(isapprox.(dec_inv_hodge_star(Val{i}, sd, DiagonalHodge()), inv_hodge_star(i, sd, DiagonalHodge()); rtol = 1e-12))
        end
    end

    for i in 0:3
        for sd in dual_meshes_3D
            @test all(isapprox.(dec_inv_hodge_star(Val{i}, sd, DiagonalHodge()), inv_hodge_star(i, sd, DiagonalHodge()); rtol = 1e-12))
        end
    end
end

@testset "Geometric Hodge" begin
    for i in 0:1
        for sd in dual_meshes_1D
            @test all(isapprox.(dec_hodge_star(Val{i}, sd, GeometricHodge()), hodge_star(i, sd, GeometricHodge()); rtol = 1e-12))
        end
    end

    for i in [0, 2]
        for sd in dual_meshes_2D
            @test all(isapprox.(dec_hodge_star(Val{i}, sd, GeometricHodge()), hodge_star(i, sd, GeometricHodge()); rtol = 1e-12))
        end
    end

    # TODO: Why does this test require atol, not rtol, to reasonably pass?
    for i in [1]
        for sd in dual_meshes_2D
            @test all(isapprox.(dec_hodge_star(Val{i}, sd, GeometricHodge()), hodge_star(i, sd, GeometricHodge()); atol = 1e-12))
        end
    end

end

@testset "Inverse Geometric Hodge" begin
    for i in 0:1
        for sd in dual_meshes_1D
            @test all(isapprox.(dec_inv_hodge_star(Val{i}, sd, GeometricHodge()), inv_hodge_star(i, sd, GeometricHodge()); rtol = 1e-12))
        end
    end

    for i in [0, 2]
        for sd in dual_meshes_2D
            @test all(isapprox.(dec_inv_hodge_star(Val{i}, sd, GeometricHodge()), inv_hodge_star(i, sd, GeometricHodge()); rtol = 1e-12))
        end
    end

    for i in 1:1
        for sd in dual_meshes_2D[1:end-1]
            V_1 = rand(ne(sd))
            @test all(isapprox.(dec_inv_hodge_star(Val{i}, sd, GeometricHodge())(V_1), inv_hodge_star(i, sd, GeometricHodge()) * V_1; rtol = 1e-12))
        end
    end

end

@testset "Wedge Product" begin
    for sd in dual_meshes_1D
        V_1, V_2 = rand(nv(sd)), rand(nv(sd))
        E_1 = rand(ne(sd))
        @test all(dec_wedge_product(Tuple{0, 0}, sd)(V_1, V_2) .== ∧(Tuple{0, 0}, sd, V_1, V_2))
        @test all(isapprox.(dec_wedge_product(Tuple{0, 1}, sd)(V_1, E_1), ∧(Tuple{0, 1}, sd, V_1, E_1); atol=1e-15))
    end

    for sd in dual_meshes_2D
        V_1, V_2 = rand(nv(sd)), rand(nv(sd))
        E_1, E_2 = rand(ne(sd)), rand(ne(sd))
        T_2 = rand(ntriangles(sd))
        V_ones = ones(nv(sd))
        E_ones = ones(ne(sd))
        @test all(dec_wedge_product(Tuple{0, 0}, sd)(V_1, V_2) .== ∧(Tuple{0, 0}, sd, V_1, V_2))

        wdg01 = dec_wedge_product(Tuple{0, 1}, sd)
        @test all(isapprox.(wdg01(V_1, E_2), ∧(Tuple{0, 1}, sd, V_1, E_2); rtol = 1e-14))
        @test all(wdg01(V_ones, E_ones) .== E_ones)

        @test all(dec_wedge_product(Tuple{0, 2}, sd)(V_1, T_2) .== ∧(Tuple{0, 2}, sd, V_1, T_2))

        @test all(dec_wedge_product(Tuple{1, 1}, sd)(E_1, E_2) .≈ ∧(Tuple{1, 1}, sd, E_1, E_2))
    end

    for sd in dual_meshes_3D
        V_1, V_2 = rand(nv(sd)), rand(nv(sd))
        E_1, E_2 = rand(ne(sd)), rand(ne(sd))
        T_2 = rand(ntriangles(sd))
        Tet_2 = rand(ntetrahedra(sd))
        V_ones = ones(nv(sd))
        E_ones = ones(ne(sd))
        T_ones = ones(ntriangles(sd))
        Tet_ones = ones(ntetrahedra(sd))

        wdg00 = dec_wedge_product(Tuple{0, 0}, sd)
        wdg01 = dec_wedge_product(Tuple{0, 1}, sd)
        wdg02 = dec_wedge_product(Tuple{0, 2}, sd)
        wdg03 = dec_wedge_product(Tuple{0, 3}, sd)

        wdg11 = dec_wedge_product(Tuple{1, 1}, sd)
        wdg12 = dec_wedge_product(Tuple{1, 2}, sd)

        @test all(wdg00(V_ones, V_ones) .== V_ones)
        @test all(wdg01(V_ones, E_ones) .== E_ones)
        @test all(wdg02(V_ones, T_ones) .≈ T_ones)
        @test all(wdg03(V_ones, Tet_ones) .≈ Tet_ones)
        @test all(wdg11(E_ones, E_ones) .== zeros(ntriangles(sd)))
        @test all(wdg12(E_ones, T_ones) .≈ ∧(Tuple{1,2}, sd, E_ones, T_ones))

        @test all(wdg00(V_1, V_2) .≈ ∧(Tuple{0, 0}, sd, V_1, V_2))
        @test all(wdg01(V_1, E_2) .≈ ∧(Tuple{0, 1}, sd, V_1, E_2))
        @test all(wdg02(V_1, T_2) .≈ ∧(Tuple{0, 2}, sd, V_1, T_2))
        @test all(wdg03(V_1, Tet_2) .≈ ∧(Tuple{0, 3}, sd, V_1, Tet_2))
        @test all(wdg11(E_1, E_2) .≈ ∧(Tuple{1, 1}, sd, E_1, E_2))
        @test all(wdg12(E_1, T_2) .≈ ∧(Tuple{1, 2}, sd, E_1, T_2))
    end

    # Test flipped edge orientation preserves value
    sd = first(dual_meshes_2D)
    E_1, E_2 = rand(ne(sd)), rand(ne(sd))
    for i in 1:ne(sd)
        sd[i, :edge_orientation] = !sd[i, :edge_orientation]
        wdg_11 = dec_wedge_product(Tuple{1, 1}, sd)
        E_1[i] = -E_1[i]; E_2[i] = -E_2[i];
        @test all(wdg_11(E_1, E_2) .≈ ∧(Tuple{1, 1}, sd, E_1, E_2))
    end

end

@testset "Dual Laplacian" begin
  # Test basic calculus properties on the interior of the mesh:
  # The second derivative of a linear function is 0.

  # 1D
  primal_line = EmbeddedDeltaSet1D{Bool,Point2d}()
  add_vertices!(primal_line, 400, point=map(p -> Point2D(p,0), range(0,10;length=400)))
  add_edges!(primal_line, 1:399, 2:400)
  sd = generate_dual_mesh(primal_line)
  twoX = map(p -> 2*p[1], sd[sd[:edge_center], :dual_point])
  nil = Δᵈ(Val{0}, sd)(twoX)
  @test all(abs.(nil[begin+1:end-1]) .< 2e-11)

  # 2D
  # TODO: This result should return near zero on the interior
  # TODO: The issue might arise from a numerically singular Geometric Hodge
  for sd in [tg, rect]
    twoX = map(p -> 2*p[1], sd[sd[:tri_center], :dual_point])
    nil = Δᵈ(Val{0}, sd)(twoX)
    interior_tris = setdiff(triangles(sd), boundary_inds(Val{2}, sd))
    @test_broken abs(mean(nil[interior_tris])) < 1e-13
    @test_broken std(nil[interior_tris]) < 1e-13
  end

  # 3D
  sd = tet_msh_sd
  twoX = map(p -> 2*p[1], sd[sd[:tet_center], :dual_point])
  nil = Δᵈ(Val{0}, sd)(twoX)
  interior_tets = setdiff(tetrahedra(sd), boundary_inds(Val{3}, sd))
  @test abs(mean(nil[interior_tets])) < 0.03
  @test std(nil[interior_tets]) < 6.7
end

@testset "Averaging Operator" begin
    for sd in dual_meshes_1D
        # Test that the averaging matrix can compute a wedge product.
        V_1 = rand(nv(sd))
        E_1 = rand(ne(sd))
        expected_wedge = dec_wedge_product(Tuple{0, 1}, sd)(V_1, E_1)
        avg_mat = avg₀₁_mat(sd)
        @test all(expected_wedge .== (avg_mat * V_1 .* E_1))
        @test all(expected_wedge .== (avg₀₁(sd, VForm(V_1)) .* E_1))
    end

    for sd in dual_meshes_2D
        # Test that the averaging matrix can compute a wedge product.
        V_1 = rand(nv(sd))
        E_1 = rand(ne(sd))
        expected_wedge = dec_wedge_product(Tuple{0, 1}, sd)(V_1, E_1)
        avg_mat = avg₀₁_mat(sd)
        @test all(expected_wedge .== (avg_mat * V_1 .* E_1))
        @test all(expected_wedge .== (avg₀₁(sd, VForm(V_1)) .* E_1))
    end
end

@testset "Primal-Dual Wedge Product 0-1" begin
    for sd in flat_meshes
      # Allocate the cached wedge operator.
      Λ10 = dec_wedge_product_dp(Tuple{1,0}, sd)
      Λ01 = dec_wedge_product_pd(Tuple{0,1}, sd)
      ♯_m = ♯_mat(sd, LLSDDSharp())

      # Define test data
      X♯ = SVector{3,Float64}(1/√2,1/√2,0)
      f = hodge_star(1,sd) * eval_constant_primal_form(sd, X♯)
      g = fill(5.0, nv(sd))

      # f := 1/√2dx + 1/√2dy:
      # g := 5
      # ⋆f = -1/√2dx + 1/√2dy
      # ⋆f∧g = 5(-1/√2dx + 1/√2dy) = -5/√2dx + 5/√2dy
      @test all(Λ10(f,g) .≈ hodge_star(1,sd) * eval_constant_primal_form(sd, SVector{3,Float64}(5/√2,5/√2,0)))

      # Test symmetry across Λ10 and Λ01.
      @test all(Λ10(f,g) .== Λ01(g,f))
    end
    for sd in flat_meshes
      # Here, We test only for matching values on interior edges, because the
      # numerical solution assumes values past the boundary are 0, whereas the
      # analytic solution has no such artifacting.  i.e. The numerical solution
      # does not hold on boundary edges.
      interior_edges = setdiff(edges(sd), boundary_inds(Val{1}, sd))
      length(interior_edges) == 0 && continue

      # Allocate the cached wedge operator.
      Λ10 = dec_wedge_product_dp(Tuple{1,0}, sd)
      ♯_m = ♯_mat(sd, LLSDDSharp())
      # Define test data and the analytic solution.
      a = map(point(sd)) do p
        p[1] + 4*p[2]
      end
      f = hodge_star(1,sd) * d(0,sd) * a
      g = map(point(sd)) do p
        4*p[1] + 16*p[2]
      end
      h = map(point(sd)) do p
        -p[1] - 4*p[2]
      end
      dx = eval_constant_primal_form(sd, SVector{3,Float64}(1,0,0))
      dy = eval_constant_primal_form(sd, SVector{3,Float64}(0,1,0))
      fΛa_analytic = hodge_star(1,sd) * (-dec_wedge_product(Tuple{0,1}, sd)(h, dx) .+ dec_wedge_product(Tuple{0,1}, sd)(g, dy))

      # a := x + 4y
      # f := ⋆da
      # f = ⋆(∂a/∂x dx + ∂a/∂y dy)
      #   = ⋆(dx + 4dy)
      # f = 4dx - dy
      # f∧a = (4dx - dy) ∧ (x + 4y)
      #     = 4(x + 4y)dx -(x + 4y)dy
      #     = (4x + 16y)dx + (-x - 4y)dy
      @test all(isapprox.(Λ10(f,a)[interior_edges], fΛa_analytic[interior_edges], atol=1e-10))
    end
end

@testset "Dual-Dual Wedge Product 0-1" begin
    for sd in flat_meshes
      # Allocate the cached wedge operator.
      Λ10 = dec_wedge_product_dd(Tuple{1,0}, sd)
      Λ01 = dec_wedge_product_dd(Tuple{0,1}, sd)

      # Define test data
      X♯ = SVector{3,Float64}(1/√2,1/√2,0)
      f = hodge_star(1,sd) * eval_constant_primal_form(sd, X♯)
      g = fill(5.0, ntriangles(sd))

      # f := 1/√2dx + 1/√2dy:
      # g := 5
      # ⋆f = -1/√2dx + 1/√2dy
      # ⋆f∧g = 5(-1/√2dx + 1/√2dy) = -5/√2dx + 5/√2dy
      @test all(Λ10(f,g) .≈ hodge_star(1,sd) * eval_constant_primal_form(sd, SVector{3,Float64}(5/√2,5/√2,0)))

      # Test symmetry across Λ10 and Λ01.
      @test all(Λ10(f,g) .== Λ01(g,f))
    end
end

@testset "Interior Product Dual-Dual 1-1" begin
    for sd in flat_meshes
      interior_edges = setdiff(edges(sd), boundary_inds(Val{1}, sd))
      isempty(interior_edges) && continue
      # Allocate the cached operators.
      d0 = dec_dual_derivative(0, sd)
      ι1 = interior_product_dd(Tuple{1,1}, sd)

      # Define test data
      X♯ = SVector{3,Float64}(1/√2,1/√2,0)
      u = hodge_star(1,sd) * eval_constant_primal_form(sd, X♯)

      # u := ⋆(1/√2dx + 1/√2dy)
      # iᵤu = -⋆(⋆u∧u)
      #     = -⋆(-1 dx∧dy)
      #     = 1
      # d1 = 0
      diᵤu = d0 * ι1(u,u);

      @test all(isapprox.(diᵤu[interior_edges], 0.0, atol=1e-13))
    end
end

function plot_dual0form(sd, f0)
  ps  = (stack(sd[sd[:tri_center], :dual_point])[[1,2],:])'
  f = Figure(); ax = CairoMakie.Axis(f[1,1]);
  sct = scatter!(ax, ps,
      color=f0);
  Colorbar(f[1,2], sct)
  f
end

function euler_equation_test(X♯, sd)
  interior_tris = setdiff(triangles(sd), boundary_inds(Val{2}, sd))

  # Allocate the cached operators.
  d0 = dec_dual_derivative(0, sd)
  d1 = dec_differential(1, sd);
  s1 = dec_hodge_star(1, sd);
  s2 = dec_hodge_star(2, sd);
  ι1 = interior_product_dd(Tuple{1,1}, sd)
  ι2 = interior_product_dd(Tuple{1,2}, sd)
  ℒ1 = ℒ_dd(Tuple{1,1}, sd)

  # This is a uniform, constant flow.
  u = s1 * eval_constant_primal_form(sd, X♯)

  # Recall Euler's Equation:
  # ∂ₜu = -ℒᵤu + 0.5dιᵤu  - 1/ρdp + b.
  # We expect for a uniform flow then that ∂ₜu = 0.
  # We will not explicitly set boundary conditions for this test.

  # The square root of the inner product is a suitable notion of magnitude.
  mag(x) = (sqrt ∘ abs).(ι1(x,x))

  # Test that the advection term -ℒᵤu + 0.5dιᵤu is 0.
  selfadv = ℒ1(u,u) - 0.5*d0*ι1(u,u)
  mag_selfadv = mag(selfadv)[interior_tris]

  # Solve for pressure using the Poisson equation
  div(x) = s2 * d1 * (s1 \ x);
  solveΔ(x) = float.(d0) \ (s1 * (float.(d1) \ (s2 \ x)))

  p = (solveΔ ∘ div)(selfadv)
  dp = d0*p
  mag_dp = mag(dp)[interior_tris]

  # Observe the derivative of u w.r.t time.
  ∂ₜu = -selfadv - dp;
  mag_∂ₜu = mag(∂ₜu)[interior_tris]

  mag_selfadv, mag_dp, mag_∂ₜu
end

@testset "Dual-Dual Interior Product and Lie Derivative" begin
  X♯ = SVector{3,Float64}(1/√2,1/√2,0)
  mag_selfadv, mag_dp, mag_∂ₜu = euler_equation_test(X♯, rect)
  # Note that "error" accumulates in the first two layers around ∂Ω.
  # That is not really error, but rather the effect of boundary conditions.
  #mag(x) = (sqrt ∘ abs).(ι1(x,x))
  #plot_dual0form(sd, mag(selfadv))
  #plot_dual0form(sd, mag(dp))
  #plot_dual0form(sd, mag(∂ₜu))
  @test .75 < (count(mag_selfadv .< 1e-8) / length(mag_selfadv))
  @test .80 < (count(mag_dp .< 1e-2) / length(mag_dp))
  @test .75 < (count(mag_∂ₜu .< 1e-2) / length(mag_∂ₜu))

  # This smaller mesh is proportionally more affected by boundary conditions.
  X♯ = SVector{3,Float64}(1/√2,1/√2,0)
  mag_selfadv, mag_dp, mag_∂ₜu  = euler_equation_test(X♯, tg)
  @test .64 < (count(mag_selfadv .< 1e-2) / length(mag_selfadv))
  @test .64 < (count(mag_dp .< 1e-2) / length(mag_dp))
  @test .60 < (count(mag_∂ₜu .< 1e-2) / length(mag_∂ₜu))

  X♯ = SVector{3,Float64}(3,3,0)
  mag_selfadv, mag_dp, mag_∂ₜu  = euler_equation_test(X♯, tg)
  @test 97/162 <= (count(mag_selfadv .< 1e-1) / length(mag_selfadv))
  @test 97/162 <= (count(mag_dp .< 1e-1) / length(mag_dp))
  @test 97/162 <= (count(mag_∂ₜu .< 1e-1) / length(mag_∂ₜu))

  # u := ⋆xdx
  # ιᵤu = x²
  sd = rect;
  f = map(point(sd)) do p
    p[1]
  end
  dx = eval_constant_primal_form(sd, SVector{3,Float64}(1,0,0))
  u = hodge_star(1,sd) * dec_wedge_product(Tuple{0,1}, sd)(f, dx)
  ι1 = interior_product_dd(Tuple{1,1}, sd)
  interior_tris = setdiff(triangles(sd), boundary_inds(Val{2}, sd))
  @test all(<(8e-3), (ι1(u,u) .- map(sd[sd[:tri_center], :dual_point]) do (x,_,_)
    x*x
  end)[interior_tris])
end

end
