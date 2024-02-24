module TestOperators

using Test
using SparseArrays
using LinearAlgebra
using CombinatorialSpaces
using CombinatorialSpaces.Meshes: tri_345, fri_345, grid_345, right_unit_hypot
using CombinatorialSpaces.DiscreteExteriorCalculus: eval_constant_primal_form
using Random
using GeometryBasics: Point2, Point3
using StaticArrays: SVector

Point2D = Point2{Float64}
Point3D = Point3{Float64}

Random.seed!(0)

function generate_dual_mesh(s::HasDeltaSet1D)
    orient!(s)
    sd = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(s)
    subdivide_duals!(sd, Barycenter())
    sd
end

function generate_dual_mesh(s::HasDeltaSet2D)
    orient!(s)
    sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(s)
    subdivide_duals!(sd, Barycenter())
    sd
end

primal_line = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(primal_line, 3, point=[Point2D(1,0), Point2D(0,0), Point2D(0,2)])
add_edges!(primal_line, [1,2], [2,3])
line = generate_dual_mesh(primal_line)

primal_cycle = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(primal_cycle, 3, point=[Point2D(1,0), Point2D(0,0), Point2D(0,1)])
add_edges!(primal_cycle, [1,2,3], [2,3,1])
cycle = generate_dual_mesh(primal_cycle)

primal_plus = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(primal_plus, 5, point=[Point2D(0,0), Point2D(1,0), Point2D(-1,0), Point2D(0,1), Point2D(0, -1)])
add_edges!(primal_plus, [1,1,3,5], [2,4,1,1])
primal_plus[:edge_orientation] = true
plus = generate_dual_mesh(primal_plus)


dual_meshes_1D = [line, cycle, plus]

dual_meshes_2D = [(generate_dual_mesh ∘ loadmesh ∘ Icosphere).(1:2)...,
               (generate_dual_mesh ∘ loadmesh)(Rectangle_30x10()),
               (generate_dual_mesh).([triangulated_grid(10,10,8,8,Point3D), makeSphere(5, 175, 5, 0, 360, 5, 6371+90)[1]])...,
               (loadmesh)(Torus_30x10())];

tg′ = triangulated_grid(100,100,10,10,Point2D);
tg = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(tg′);
subdivide_duals!(tg, Barycenter());

rect′ = loadmesh(Rectangle_30x10());
rect = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(rect′);
subdivide_duals!(rect, Barycenter());

flat_meshes = [tri_345()[2], fri_345()[2], right_unit_hypot()[2], grid_345()[2], tg, rect];

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
        @test all(dec_wedge_product(Tuple{0, 1}, sd)(V_1, E_1) .== ∧(Tuple{0, 1}, sd, V_1, E_1))
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

        @test all(dec_wedge_product(Tuple{1, 1}, sd)(E_1, E_2) .== ∧(Tuple{1, 1}, sd, E_1, E_2))
    end
end

# TODO: Move all boundary helper functions into CombinatorialSpaces.
function boundary_inds(::Type{Val{1}}, s)
  collect(findall(x -> x != 0, boundary(Val{2},s) * fill(1,ntriangles(s))))
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
      @test all(isapprox.(Λ10(f,a)[interior_edges], fΛa_analytic[interior_edges], atol=1e-8))
    end
end

end
