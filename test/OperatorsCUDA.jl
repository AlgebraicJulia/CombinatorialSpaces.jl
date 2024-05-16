module TestOperatorsCUDA

using CUDA
using CUDA.CUSPARSE
using Catlab
using CombinatorialSpaces
using GeometryBasics: Point2, Point3
using Krylov
using LinearAlgebra
using LinearOperators
using Random
using SparseArrays
using StaticArrays: SVector
using Statistics: mean
using Test

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

function generate_dual_mesh_2(s::HasDeltaSet2D)
  orient!(s)
  sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(s)
  subdivide_duals!(sd, Circumcenter())
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
  (generate_dual_mesh).([triangulated_grid(10,10,8,8,Point3D), makeSphere(5, 175, 5, 0, 360, 5, 6371+90)[1]])...];

dual_meshes_2D_2 = [(generate_dual_mesh_2 ∘ loadmesh)(Rectangle_30x10()), (generate_dual_mesh_2)(triangulated_grid(10,10,8,8,Point2D))];

function test_cpu_gpu_equality(meshes, range, operator, conversion)
  for i in range
    for sd in meshes
      @test all(operator(i, sd) .== conversion(operator(i, sd, Val{:CUDA})))
    end
  end
end

function test_cpu_gpu_equality(meshes, range, operator, conversion, hodge)
  for i in range
    for sd in meshes
      @test all(operator(i, sd, hodge) .== conversion(operator(i, sd, hodge, Val{:CUDA})))
    end
  end
end

@testset "Exterior Derivative" begin
  test_cpu_gpu_equality(dual_meshes_1D, [0], dec_differential, SparseMatrixCSC)
  test_cpu_gpu_equality(dual_meshes_2D, [0,1], dec_differential, SparseMatrixCSC)
  test_cpu_gpu_equality(dual_meshes_2D_2, [0,1], dec_differential, SparseMatrixCSC)
end

@testset "Boundary" begin
  test_cpu_gpu_equality(dual_meshes_1D, [1], dec_boundary, SparseMatrixCSC)
  test_cpu_gpu_equality(dual_meshes_2D, [1,2], dec_boundary, SparseMatrixCSC)
  test_cpu_gpu_equality(dual_meshes_2D_2, [1,2], dec_boundary, SparseMatrixCSC)
end

@testset "Dual Derivative" begin
  test_cpu_gpu_equality(dual_meshes_1D, [0], dec_dual_derivative, SparseMatrixCSC)
  test_cpu_gpu_equality(dual_meshes_2D, [0,1], dec_dual_derivative, SparseMatrixCSC)
  test_cpu_gpu_equality(dual_meshes_2D_2, [0,1], dec_dual_derivative, SparseMatrixCSC)
end

@testset "Diagonal Hodge" begin
  test_cpu_gpu_equality(dual_meshes_1D, [0,1], dec_hodge_star, Array, DiagonalHodge())
  test_cpu_gpu_equality(dual_meshes_2D_2, [0,1,2], dec_hodge_star, Array, DiagonalHodge())
end

@testset "Inverse Diagonal Hodge" begin
  test_cpu_gpu_equality(dual_meshes_1D, [0,1], dec_inv_hodge_star, Array, DiagonalHodge())
  test_cpu_gpu_equality(dual_meshes_2D_2, [0,1,2], dec_inv_hodge_star, Array, DiagonalHodge())
end

@testset "Geometric Hodge" begin
  test_cpu_gpu_equality(dual_meshes_1D, [0,1], dec_hodge_star, Array, GeometricHodge())
  test_cpu_gpu_equality(dual_meshes_2D, [0,2], dec_hodge_star, Array, GeometricHodge())
  test_cpu_gpu_equality(dual_meshes_2D, [1], dec_hodge_star, SparseMatrixCSC, GeometricHodge())
end
      
@testset "Inverse Geometric Hodge" begin
  for i in 1:1
    for sd in dual_meshes_2D[1:end-1]
      V_1 = Float64.(I[1:ne(sd), 1])
      @test all(isapprox.(dec_inv_hodge_star(Val{i}, sd, GeometricHodge())(V_1), Array(dec_inv_hodge_star(Val{i}, sd, GeometricHodge(), Val{:CUDA})(CuArray(V_1))); atol = 1e-10))
    end
  end
end

@testset "Wedge Product" begin
  for sd in dual_meshes_1D
    V_1, V_2 = rand(nv(sd)), rand(nv(sd))
    E_1 = rand(ne(sd))
    @test all(Array(dec_wedge_product(Tuple{0, 0}, sd, Val{:CUDA})(CuArray(V_1), CuArray(V_2))) .== ∧(Tuple{0, 0}, sd, V_1, V_2))
    @test all(isapprox.(Array(dec_wedge_product(Tuple{0, 1}, sd, Val{:CUDA})(CuArray(V_1), CuArray(E_1))), ∧(Tuple{0, 1}, sd, V_1, E_1); atol = 1e-14))
    @test all(isapprox.(Array(dec_wedge_product(Tuple{1, 0}, sd, Val{:CUDA})(CuArray(E_1), CuArray(V_1))), ∧(Tuple{1, 0}, sd, E_1, V_1); atol = 1e-14))
  end

  for sd in dual_meshes_2D
    V_1, V_2 = rand(nv(sd)), rand(nv(sd))
    E_1, E_2 = rand(ne(sd)), rand(ne(sd))
    T_2 = rand(ntriangles(sd))
    V_ones = ones(nv(sd))
    E_ones = ones(ne(sd))
    @test all(Array(dec_wedge_product(Tuple{0, 0}, sd, Val{:CUDA})(CuArray(V_1), CuArray(V_2))) .== ∧(Tuple{0, 0}, sd, V_1, V_2))

    wdg01 = dec_wedge_product(Tuple{0, 1}, sd, Val{:CUDA})
    @test all(isapprox.(Array(wdg01(CuArray(V_1), CuArray(E_2))), ∧(Tuple{0, 1}, sd, V_1, E_2); atol = 1e-14))
    @test all(Array(wdg01(CuArray(V_ones), CuArray(E_ones))) .== E_ones)
    @test all(isapprox.(Array(dec_wedge_product(Tuple{1, 0}, sd, Val{:CUDA})(CuArray(E_1), CuArray(V_2))), ∧(Tuple{1,0}, sd, E_1, V_2); atol = 1e-14))

    @test all(isapprox.(Array(dec_wedge_product(Tuple{0, 2}, sd, Val{:CUDA})(CuArray(V_1), CuArray(T_2))), ∧(Tuple{0, 2}, sd, V_1, T_2); atol = 1e-14))

    @test all(isapprox.(Array(dec_wedge_product(Tuple{1, 1}, sd, Val{:CUDA})(CuArray(E_1), CuArray(E_2))), ∧(Tuple{1, 1}, sd, E_1, E_2); atol = 1e-12))
  end
end

@testset "Krylov" begin
  for sd in dual_meshes_2D
    Δ0_cpu = Δ(0,sd)
    Δ0 = CuSparseMatrixCSC(Δ0_cpu)
    w = Δ0 * CuArray(map(x -> x[1]^3, point(sd)))

    v = CuArray(zeros(Float64, nv(sd)))
    y = CuArray(zeros(Float64, nv(sd)))
    z = CuArray(zeros(Float64, nv(sd)))

    # Factorizing on the GPU:
    Δ0p = ilu02(Δ0)
    op = LinearOperator(Float64, nv(sd), nv(sd), false, false,
      (y,x) -> begin
        ldiv!(z, LowerTriangular(Δ0p), x)
        ldiv!(y, UnitUpperTriangular(Δ0p), z)
        y
      end)

    x, stats = bicgstab(Δ0, w, M=op, atol=1e-13)
    resid = Δ0*x .- w
    RMS = sqrt(mean(resid.^2))
    @test stats.solved
    @test RMS < 1e-5

    x, stats = gmres(Δ0, w, M=op, atol=1e-13)
    RMS = sqrt(mean(resid.^2))
    @test stats.solved
    @test RMS < 1e-5

    # Factorizing on the CPU:
    luΔ0_cpu = lu(Δ0_cpu)
    L = CuSparseMatrixCSC(luΔ0_cpu.L)
    U = CuSparseMatrixCSC(luΔ0_cpu.U)
    p = CuArray(luΔ0_cpu.p)
    q⁻¹ = CuArray(invperm(luΔ0_cpu.q))
    Rs = CuArray(luΔ0_cpu.Rs)

    # Solving on the GPU:
    w_Rsp = (w .* Rs)[p]
    ldiv!(z, LowerTriangular(L), w_Rsp)
    ldiv!(v, UpperTriangular(U), z)
    y .= v[q⁻¹]

    resid = Δ0*y .- w
    RMS = sqrt(mean(resid.^2))
    @test RMS < 1e-8

    # Solving on the CPU:
    y_cpu = luΔ0_cpu \ Array(w)
    resid = Δ0_cpu*(y_cpu) .- Array(w)
    RMS = sqrt(mean(resid.^2))
    @test RMS < 1e-8

    # Comparisons of the solutions:
    resid_resid = Array(Δ0*y) .- Δ0_cpu*y_cpu
    RMS = sqrt(mean(resid_resid.^2))
  end
end

end
