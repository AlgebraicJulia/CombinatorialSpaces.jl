module TestOperatorsCUDA

using Test
using SparseArrays
using LinearAlgebra
using CombinatorialSpaces
using CombinatorialSpaces.DiscreteExteriorCalculus: DiscreteHodge
using Catlab
using Random
using GeometryBasics: Point2, Point3
using StaticArrays: SVector
using Krylov

Point2D = Point2{Float64}
Point3D = Point3{Float64}

Random.seed!(0)

# Test Meshes
#------------

function generate_dual(s::HasDeltaSet1D)
  orient!(s)
  sd = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(s)
  subdivide_duals!(sd, Barycenter())
  sd
end

function generate_dual(s::EmbeddedDeltaSet2D{B, P}, c::SimplexCenter) where {B,P}
  orient!(s)
  sd = EmbeddedDeltaDualComplex2D{B,Float64,P}(s)
  subdivide_duals!(sd, c)
  sd
end

line = path_graph(EmbeddedDeltaSet1D{Bool,Point2D}, 3)
line[:point] = [Point2D(1,0), Point2D(0,0), Point2D(0,2)]

cycle = cycle_graph(EmbeddedDeltaSet1D{Bool,Point2D}, 3)
cycle[:point] = [Point2D(1,0), Point2D(0,0), Point2D(0,1)]

# The star is not a valid 1-manifold, so set an explicit orientation.
star = star_graph(EmbeddedDeltaSet1D{Bool,Point2D}, 5);
star[:point] = [Point2D(0,-1), Point2D(1,0), Point2D(-1,0), Point2D(0,1), Point2D(0,0)]
star[:edge_orientation] = true;

dual_meshes_1D = map(generate_dual, [line, cycle, star])

dual_meshes_2D_bary = map(x -> generate_dual(x, Barycenter()), [
  loadmesh(Icosphere(1)),
  loadmesh(Icosphere(2)),
  loadmesh(Rectangle_30x10()),
  triangulated_grid(10, 10, 8, 8, Point3D),
  makeSphere(5, 175, 5, 0, 360, 5, 6371+90)[1]]);

dual_meshes_2D_circum = map(x -> generate_dual(x, Circumcenter()), [
  loadmesh(Rectangle_30x10()),
  triangulated_grid(10, 10, 8, 8, Point2D)]);

# Operator Test Definitions
#--------------------------

function test_unary_operators(backend)
  function test_cpu_gpu_equality(meshes, degrees, operator)
    for (n, sd) in Iterators.product(degrees, meshes)
      cpu_ans = operator(n, sd)
      alt_ans = operator(n, sd, backend)
      @test all(cpu_ans .== typeof(cpu_ans)(alt_ans))
    end
  end
  
  function test_cpu_gpu_equality(meshes, degrees, operator, hodge::DiscreteHodge)
    for (n, sd) in Iterators.product(degrees, meshes)
      cpu_ans = operator(n, sd, hodge)
      alt_ans = operator(n, sd, hodge, backend)
      @test all(cpu_ans .== typeof(cpu_ans)(alt_ans))
    end
  end

  @testset "Exterior Derivative" begin
    test_cpu_gpu_equality(dual_meshes_1D,        [0],   dec_differential)
    test_cpu_gpu_equality(dual_meshes_2D_bary,   [0,1], dec_differential)
    test_cpu_gpu_equality(dual_meshes_2D_circum, [0,1], dec_differential)
  end
  @testset "Boundary" begin
    test_cpu_gpu_equality(dual_meshes_1D,        [1],   dec_boundary)
    test_cpu_gpu_equality(dual_meshes_2D_bary,   [1,2], dec_boundary)
    test_cpu_gpu_equality(dual_meshes_2D_circum, [1,2], dec_boundary)
  end
  @testset "Dual Derivative" begin
    test_cpu_gpu_equality(dual_meshes_1D,        [0],   dec_dual_derivative)
    test_cpu_gpu_equality(dual_meshes_2D_bary,   [0,1], dec_dual_derivative)
    test_cpu_gpu_equality(dual_meshes_2D_circum, [0,1], dec_dual_derivative)
  end
  @testset "Diagonal Hodge" begin
    test_cpu_gpu_equality(dual_meshes_1D,        [0,1],   dec_hodge_star, DiagonalHodge())
    test_cpu_gpu_equality(dual_meshes_2D_circum, [0,1,2], dec_hodge_star, DiagonalHodge())
  end
  @testset "Inverse Diagonal Hodge" begin
    test_cpu_gpu_equality(dual_meshes_1D,        [0,1],   dec_inv_hodge_star, DiagonalHodge())
    test_cpu_gpu_equality(dual_meshes_2D_circum, [0,1,2], dec_inv_hodge_star, DiagonalHodge())
  end
  @testset "Geometric Hodge" begin
    test_cpu_gpu_equality(dual_meshes_1D,      [0,1], dec_hodge_star, GeometricHodge())
    test_cpu_gpu_equality(dual_meshes_2D_bary, [0,2], dec_hodge_star, GeometricHodge())
    test_cpu_gpu_equality(dual_meshes_2D_bary, [1],   dec_hodge_star, GeometricHodge())
  end
end

function test_hodge_solver()
  @testset "Inverse Geometric Hodge" begin
    for sd in dual_meshes_2D[1:end-1]
      V_1 = Float64.(I[1:ne(sd), 1])
      cuV_1 = CuArray(V_1)
      @test all(isapprox.(
        dec_inv_hodge_star(Val{1}, sd, GeometricHodge())(V_1),
        Array(dec_inv_hodge_star(Val{1}, sd, GeometricHodge(), Val{:CUDA})(cuV_1));
        atol = 1e-10))
    end
  end
end

function test_binary_operators(float_type, backend, arr_cons, tol)
  function all_are_approx(x,y)
    all(isapprox.(Array(x), y; atol=tol))
  end

  @testset "Wedge Product" begin
    function test_wedge_1D(sd, backend)
      V1, V2, E1 = rand.(float_type, [nv(sd), nv(sd), ne(sd)])
      altV1, altV2, altE1 = arr_cons.([V1, V2, E1])
  
      wdg00 = dec_wedge_product(Tuple{0,0}, sd, backend)
      wdg01 = dec_wedge_product(Tuple{0,1}, sd, backend)
      wdg10 = dec_wedge_product(Tuple{1,0}, sd, backend)
  
      @test all_are_approx(wdg00(altV1, altV2), ∧(Tuple{0,0}, sd, V1, V2))
      @test all_are_approx(wdg01(altV1, altE1), ∧(Tuple{0,1}, sd, V1, E1))
      @test all_are_approx(wdg10(altE1, altV1), ∧(Tuple{1,0}, sd, E1, V1))
    end

    function test_wedge_2D(sd, backend)
      V1, V2, E1, E2, T1 = rand.(float_type, [nv(sd), nv(sd), ne(sd), ne(sd), ntriangles(sd)])
      altV1, altV2, altE1, altE2, altT1 = arr_cons.([V1, V2, E1, E2, T1])
      V_ones, E_ones = ones(float_type, nv(sd)), ones(float_type, ne(sd))
      altV_ones, altE_ones = arr_cons.([V_ones, E_ones])
  
      wdg00 = dec_wedge_product(Tuple{0,0}, sd, backend)
      wdg01 = dec_wedge_product(Tuple{0,1}, sd, backend)
      wdg10 = dec_wedge_product(Tuple{1,0}, sd, backend)
      wdg11 = dec_wedge_product(Tuple{1,1}, sd, backend)
      wdg02 = dec_wedge_product(Tuple{0,2}, sd, backend)
  
      @test all_are_approx(wdg01(altV_ones, altE_ones), ∧(Tuple{0,1}, sd, V_ones, E_ones))
      @test all_are_approx(wdg00(altV1, altV2), ∧(Tuple{0,0}, sd, V1, V2))
      @test all_are_approx(wdg01(altV1, altE2), ∧(Tuple{0,1}, sd, V1, E2))
      @test all_are_approx(wdg10(altE1, altV2), ∧(Tuple{1,0}, sd, E1, V2))
      @test all_are_approx(wdg02(altV1, altT1), ∧(Tuple{0,2}, sd, V1, T1))
      @test all_are_approx(wdg11(altE1, altE2), ∧(Tuple{1,1}, sd, E1, E2))
    end

    for sd in dual_meshes_1D
      test_wedge_1D(sd, backend)
    end
    for sd in dual_meshes_2D_bary
      test_wedge_2D(sd, backend)
    end
  end
end

# Execute Tests
#--------------

# Test that Float32s pass through correctly.
@testset "Float32 Operators" begin
  test_binary_operators(Float32, Val{:CPU}, Array, 1e-7)
end

using CUDA
if CUDA.functional()
  @testset "CUDA" begin
    test_unary_operators(Val{:CUDA})
    test_hodge_solver()
    test_binary_operators(Float64, Val{:CUDA}, CuArray, 1e-12)
  end
else
  @info "CUDA tests were not run, since CUDA.functional() is false."
  @info CUDA.functional(true)
end

using Metal
if Metal.functional()
  @testset "Metal" begin
    test_binary_operators(Float32, Val{:Metal}, MtlArray, 0.5e-6)
    test_binary_operators(Float16, Val{:Metal}, MtlArray, 0.5e-3)
  end
else
  @info "Metal tests were not run, since Metal.functional() is false."
  @info Metal.functional(true)
end

end
