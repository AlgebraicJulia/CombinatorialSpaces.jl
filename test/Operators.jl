module TestOperators

using Test
using SparseArrays
using LinearAlgebra
using CombinatorialSpaces
using Random
using GeometryBasics: Point2, Point3

Point2D = Point2{Float64}
Point3D = Point3{Float64}

Random.seed!(0)

function generate_dual_mesh(s::HasDeltaSet1D)
    orient!(s)
    sd = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(s, FastMesh())
    subdivide_duals!(sd, FastMesh(), Barycenter())
    sd
end

function generate_dual_mesh(s::HasDeltaSet2D)
    orient!(s)
    sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(s, FastMesh())
    subdivide_duals!(sd, FastMesh(), Barycenter())
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

        @test all(dec_wedge_product(Tuple{0, 2}, sd)(V_1, T_2) / 2 .== ∧(Tuple{0, 2}, sd, V_1, T_2))

        @test all(dec_wedge_product(Tuple{1, 1}, sd)(E_1, E_2) .== ∧(Tuple{1, 1}, sd, E_1, E_2))
    end
end

end