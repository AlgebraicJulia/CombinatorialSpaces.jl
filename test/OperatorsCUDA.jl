module TestOperatorsCUDA

using Test
using SparseArrays
using LinearAlgebra
using CUDA
using Catlab
using CombinatorialSpaces
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

@testset "Exterior Derivative" begin
    for i in 0:0 
        for sd in dual_meshes_1D
            @test all(dec_differential(i, sd) .== sparse(dec_differential(i, sd, Val{:CUDA})))
        end
    end

    for i in 0:1 
        for sd in dual_meshes_2D
            @test all(dec_differential(i, sd) .== sparse(dec_differential(i, sd, Val{:CUDA})))
        end
    end
end

@testset "Boundary" begin
    for i in 1:1 
        for sd in dual_meshes_1D
            @test all(dec_boundary(i, sd) .== sparse(dec_boundary(i, sd, Val{:CUDA})))
        end
    end

    for i in 1:2
        for sd in dual_meshes_2D
            @test all(dec_boundary(i, sd) .== sparse(dec_boundary(i, sd, Val{:CUDA})))
        end
    end
end

@testset "Dual Derivative" begin
    for i in 0:0 
        for sd in dual_meshes_1D
            @test all(dec_dual_derivative(i, sd) .== sparse(dual_derivative(i, sd, Val{:CUDA})))
        end
    end

    for i in 0:1
        for sd in dual_meshes_2D
            @test all(dec_dual_derivative(i, sd) .== sparse(dual_derivative(i, sd, Val{:CUDA})))
        end
    end
end

#TODO: For hodge star 1, the values seems to be extremely close yet not quite equal
@testset "Diagonal Hodge" begin
    for i in 0:1
        for sd in dual_meshes_1D
            @test all(dec_hodge_star(Val{i}, sd, DiagonalHodge()) .== Array(dec_hodge_star(i, sd, DiagonalHodge(), Val{:CUDA})))
        end
    end

    for i in 0:2
        for sd in dual_meshes_2D
            @test all(dec_hodge_star(Val{i}, sd, DiagonalHodge()) .== Array(dec_hodge_star(i, sd, DiagonalHodge(), Val{:CUDA})))
        end
    end
end

#TODO: For inv hodge star 1, the values seems to be extremely close yet not quite equal
@testset "Inverse Diagonal Hodge" begin
    for i in 0:1
        for sd in dual_meshes_1D
            @test all(dec_inv_hodge_star(Val{i}, sd, DiagonalHodge()) .== Array(dec_inv_hodge_star(i, sd, DiagonalHodge(), Val{:CUDA})))
        end
    end

    for i in 0:2
        for sd in dual_meshes_2D
            @test all(dec_inv_hodge_star(Val{i}, sd, DiagonalHodge()) .== Array(dec_inv_hodge_star(i, sd, DiagonalHodge(), Val{:CUDA})))
        end
    end
end

@testset "Geometric Hodge" begin
    for i in 0:1
        for sd in dual_meshes_1D
            @test all(dec_hodge_star(Val{i}, sd, GeometricHodge()) .== Array(hodge_star(i, sd, GeometricHodge(), Val{:CUDA})))
        end
    end

    for i in [0, 2]
        for sd in dual_meshes_2D
            @test all(dec_hodge_star(Val{i}, sd, GeometricHodge()) .== Array(hodge_star(i, sd, GeometricHodge(), Val{:CUDA})))
        end
    end

    # TODO: Why does this test require atol, not rtol, to reasonably pass?
    for i in [1]
        for sd in dual_meshes_2D
            @test all(dec_hodge_star(Val{i}, sd, GeometricHodge()) .== sparse(dec_hodge_star(i, sd, GeometricHodge(), Val{:CUDA})))
        end
    end

end
            
@testset "Inverse Geometric Hodge" begin
    for i in 1:1
        for sd in dual_meshes_2D[1:end-1]
            V_1 = rand(ne(sd))
            @test all(isapprox.(dec_inv_hodge_star(Val{i}, sd, GeometricHodge())(V_1), Array(dec_inv_hodge_star(i, sd, GeometricHodge(), Val{:CUDA})(CuArray(V_1)); rtol = 1e-12)))
        end
    end
end

@testset "Wedge Product" begin
    for sd in dual_meshes_1D
        V_1, V_2 = rand(nv(sd)), rand(nv(sd))
        E_1 = rand(ne(sd))
        @test all(Array(dec_cu_wedge_product(Tuple{0, 0}, sd)(CuArray(V_1), CuArray(V_2))) .== ∧(Tuple{0, 0}, sd, V_1, V_2))
        @test all(isapprox.(Array(dec_cu_wedge_product(Tuple{0, 1}, sd)(CuArray(V_1), CuArray(E_1))), ∧(Tuple{0, 1}, sd, V_1, E_1); rtol = 1e-14))
    end

    for sd in dual_meshes_2D
        V_1, V_2 = rand(nv(sd)), rand(nv(sd))
        E_1, E_2 = rand(ne(sd)), rand(ne(sd))
        T_2 = rand(ntriangles(sd))
        V_ones = ones(nv(sd))
        E_ones = ones(ne(sd))
        @test all(Array(dec_cu_wedge_product(Tuple{0, 0}, sd)(CuArray(V_1), CuArray(V_2))) .== ∧(Tuple{0, 0}, sd, V_1, V_2))

        wdg01 = dec_cu_wedge_product(Tuple{0, 1}, sd)
        @test all(isapprox.(Array(wdg01(CuArray(V_1), CuArray(E_2))), ∧(Tuple{0, 1}, sd, V_1, E_2); rtol = 1e-14))
        @test all(Array(wdg01(CuArray(V_ones), CuArray(E_ones))) .== E_ones)

        @test all(isapprox.(Array(dec_cu_wedge_product(Tuple{0, 2}, sd)(CuArray(V_1), CuArray(T_2))), ∧(Tuple{0, 2}, sd, V_1, T_2); rtol = 1e-14))

        @test all(isapprox.(Array(dec_cu_wedge_product(Tuple{1, 1}, sd)(CuArray(E_1), CuArray(E_2))), ∧(Tuple{1, 1}, sd, E_1, E_2); rtol = 1e-12))
    end
end

end