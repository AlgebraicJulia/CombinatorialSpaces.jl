module TestOperatorsCUDA

using Adapt
using Catlab
using CombinatorialSpaces
using CombinatorialSpaces.DiscreteExteriorCalculus: DiscreteHodge
using GeometryBasics: Point2, Point3
using KernelAbstractions
using Krylov
using LinearAlgebra
using Random
using SparseArrays
using StaticArrays: SVector
using Statistics: mean
using Test

Random.seed!(0)

# Test Meshes
#------------

function generate_dual(s::HasDeltaSet1D)
  orient!(s)
  sd = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2d}(s)
  subdivide_duals!(sd, Barycenter())
  sd
end

function generate_dual(s::EmbeddedDeltaSet2D{B, P}, c::SimplexCenter) where {B,P}
  orient!(s)
  sd = EmbeddedDeltaDualComplex2D{B,Float64,P}(s)
  subdivide_duals!(sd, c)
  sd
end

line = path_graph(EmbeddedDeltaSet1D{Bool,Point2d}, 3)
line[:point] = [Point2d(1,0), Point2d(0,0), Point2d(0,2)]

cycle = cycle_graph(EmbeddedDeltaSet1D{Bool,Point2d}, 3)
cycle[:point] = [Point2d(1,0), Point2d(0,0), Point2d(0,1)]

# The star is not a valid 1-manifold, so set an explicit orientation.
star = star_graph(EmbeddedDeltaSet1D{Bool,Point2d}, 5);
star[:point] = [Point2d(0,-1), Point2d(1,0), Point2d(-1,0), Point2d(0,1), Point2d(0,0)]
star[:edge_orientation] = true;

dual_meshes_1D = map(generate_dual, [line, cycle, star])

dual_meshes_2D_bary = map(x -> generate_dual(x, Barycenter()), [
  loadmesh(Icosphere(1)),
  loadmesh(Icosphere(2)),
  loadmesh(Rectangle_30x10()),
  triangulated_grid(10, 10, 8, 8, Point3d),
  makeSphere(5, 175, 5, 0, 360, 5, 6371+90)[1]]);

dual_meshes_2D_circum = map(x -> generate_dual(x, Circumcenter()), [
  loadmesh(Rectangle_30x10()),
  triangulated_grid(10, 10, 8, 8, Point2d)]);

# Operator Test Definitions
#--------------------------

function test_unary_operators(backend,
  meshes_1D=dual_meshes_1D,
  meshes_2D_bary=dual_meshes_2D_bary,
  meshes_2D_circum=dual_meshes_2D_circum)
  function all_are_equal(cpu_ans::SparseMatrixCSC, alt_ans)
    KernelAbstractions.synchronize(get_backend(alt_ans))
    all(cpu_ans .== SparseMatrixCSC(alt_ans))
  end

  function all_are_equal(cpu_ans::Diagonal, alt_ans)
    KernelAbstractions.synchronize(get_backend(alt_ans))
    all(cpu_ans .== Array(alt_ans))
  end

  function test_cpu_gpu_equality(meshes, degrees, operator)
    for (n, sd) in Iterators.product(degrees, meshes)
      @test all_are_equal(operator(n, sd), operator(n, sd, backend))
    end
  end

  function test_cpu_gpu_equality(meshes, degrees, operator, hodge::DiscreteHodge)
    for (n, sd) in Iterators.product(degrees, meshes)
      @test all_are_equal(operator(n, sd, hodge), operator(n, sd, hodge, backend))
    end
  end

  @testset "Exterior Derivative" begin
    test_cpu_gpu_equality(meshes_1D,        [0],   dec_differential)
    test_cpu_gpu_equality(meshes_2D_bary,   [0,1], dec_differential)
    test_cpu_gpu_equality(meshes_2D_circum, [0,1], dec_differential)
  end
  @testset "Boundary" begin
    test_cpu_gpu_equality(meshes_1D,        [1],   dec_boundary)
    test_cpu_gpu_equality(meshes_2D_bary,   [1,2], dec_boundary)
    test_cpu_gpu_equality(meshes_2D_circum, [1,2], dec_boundary)
  end
  @testset "Dual Derivative" begin
    test_cpu_gpu_equality(meshes_1D,        [0],   dec_dual_derivative)
    test_cpu_gpu_equality(meshes_2D_bary,   [0,1], dec_dual_derivative)
    test_cpu_gpu_equality(meshes_2D_circum, [0,1], dec_dual_derivative)
  end
  @testset "Diagonal Hodge" begin
    test_cpu_gpu_equality(meshes_1D,        [0,1],   dec_hodge_star, DiagonalHodge())
    test_cpu_gpu_equality(meshes_2D_circum, [0,1,2], dec_hodge_star, DiagonalHodge())
  end
  @testset "Inverse Diagonal Hodge" begin
    test_cpu_gpu_equality(meshes_1D,        [0,1],   dec_inv_hodge_star, DiagonalHodge())
    test_cpu_gpu_equality(meshes_2D_circum, [0,1,2], dec_inv_hodge_star, DiagonalHodge())
  end
  @testset "Geometric Hodge" begin
    test_cpu_gpu_equality(meshes_1D,      [0,1], dec_hodge_star, GeometricHodge())
    test_cpu_gpu_equality(meshes_2D_bary, [0,2], dec_hodge_star, GeometricHodge())
    test_cpu_gpu_equality(meshes_2D_bary, [1],   dec_hodge_star, GeometricHodge())
  end
end

function test_hodge_solver()
  @testset "Inverse Geometric Hodge" begin
    for sd in dual_meshes_2D_bary[1:end-1]
      V_1 = Float64.(I[1:ne(sd), 1])
      cuV_1 = CuArray(V_1)
      @test all(isapprox.(
        dec_inv_hodge_star(1, sd, GeometricHodge())(V_1),
        Array(dec_inv_hodge_star(1, sd, GeometricHodge(), Val(:CUDA))(cuV_1));
        atol = 1e-10))
    end
  end
end

function test_binary_operators(float_type, backend, arr_cons, tol)
  function mse(x,y)
    KernelAbstractions.synchronize(get_backend(y))
    mean(map(z -> z^2, Array(x) .- y)) < tol
  end

  @testset "Wedge Product" begin
    function test_wedge_1D(sd, backend)
      V1, V2, E1 = rand.(float_type, [nv(sd), nv(sd), ne(sd)])
      altV1, altV2, altE1 = arr_cons.([V1, V2, E1])

      wdg00 = dec_wedge_product(0, 0, sd, backend, arr_cons, float_type)
      wdg01 = dec_wedge_product(0, 1, sd, backend, arr_cons, float_type)
      wdg10 = dec_wedge_product(1, 0, sd, backend, arr_cons, float_type)

      @test mse(wdg00(altV1, altV2), ∧(0, 0, sd, V1, V2))
      @test mse(wdg01(altV1, altE1), ∧(0, 1, sd, V1, E1))
      @test mse(wdg10(altE1, altV1), ∧(1, 0, sd, E1, V1))
    end

    function test_wedge_2D(sd, backend)
      V1, V2, E1, E2, T1 = rand.(float_type, [nv(sd), nv(sd), ne(sd), ne(sd), ntriangles(sd)])
      altV1, altV2, altE1, altE2, altT1 = arr_cons.([V1, V2, E1, E2, T1])
      V_ones, E_ones = ones(float_type, nv(sd)), ones(float_type, ne(sd))
      altV_ones, altE_ones = arr_cons.([V_ones, E_ones])

      wdg00 = dec_wedge_product(0, 0, sd, backend, arr_cons, float_type)
      wdg01 = dec_wedge_product(0, 1, sd, backend, arr_cons, float_type)
      wdg10 = dec_wedge_product(1, 0, sd, backend, arr_cons, float_type)
      wdg11 = dec_wedge_product(1, 1, sd, backend, arr_cons, float_type)
      wdg02 = dec_wedge_product(0, 2, sd, backend, arr_cons, float_type)

      @test mse(wdg01(altV_ones, altE_ones), E_ones)
      @test mse(wdg00(altV1, altV2), ∧(0, 0, sd, V1, V2))
      @test mse(wdg01(altV1, altE2), ∧(0, 1, sd, V1, E2))
      @test mse(wdg10(altE1, altV2), ∧(1, 0, sd, E1, V2))
      @test mse(wdg02(altV1, altT1), ∧(0, 2, sd, V1, T1))
      @test mse(wdg11(altE1, altE2), ∧(1, 1, sd, E1, E2))
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
  test_binary_operators(Float32, Val(:CPU), Array, 1e-15)
end

using CUDA
if CUDA.functional()
  @testset "CUDA" begin
    test_unary_operators(Val(:CUDA))
    test_hodge_solver()
    test_binary_operators(Float64, Val(:CUDA), CuArray, 1e-15)
    test_binary_operators(Float32, Val(:CUDA), CuArray, 1e-15)
  end
else
  # Get the short error description instead of full stacktrace
  error_msg = if isdefined(CUDA, :_initialization_error) && CUDA._initialization_error !== nothing
    CUDA._initialization_error
  else
    "unknown reason"
  end
  @info "CUDA tests were not run, since CUDA.functional() is false." reason=error_msg
end

if Sys.isapple()
  using Metal
  using AppleAccelerate
  dev = Metal.device()
  if Metal.supports_family(dev, Metal.MTL.MTLGPUFamilyApple7) && Metal.supports_family(dev, Metal.MTL.MTLGPUFamilyMetal3)

    # Float32 test meshes for Metal (Metal does not support Float64)
    line_f32 = path_graph(EmbeddedDeltaSet1D{Bool,Point2{Float32}}, 3)
    line_f32[:point] = [Point2{Float32}(1,0), Point2{Float32}(0,0), Point2{Float32}(0,2)]

    cycle_f32 = cycle_graph(EmbeddedDeltaSet1D{Bool,Point2{Float32}}, 3)
    cycle_f32[:point] = [Point2{Float32}(1,0), Point2{Float32}(0,0), Point2{Float32}(0,1)]

    star_f32 = star_graph(EmbeddedDeltaSet1D{Bool,Point2{Float32}}, 5)
    star_f32[:point] = [Point2{Float32}(0,-1), Point2{Float32}(1,0), Point2{Float32}(-1,0), Point2{Float32}(0,1), Point2{Float32}(0,0)]
    star_f32[:edge_orientation] = true

    function generate_dual_f32(s::HasDeltaSet1D)
      orient!(s)
      sd = EmbeddedDeltaDualComplex1D{Bool,Float32,Point2{Float32}}(s)
      subdivide_duals!(sd, Barycenter())
      sd
    end

    function generate_dual_f32(s::EmbeddedDeltaSet2D{B,P}, c::SimplexCenter) where {B,P}
      orient!(s)
      sd = EmbeddedDeltaDualComplex2D{B,Float32,P}(s)
      subdivide_duals!(sd, c)
      sd
    end

    dual_meshes_1D_f32 = map(generate_dual_f32, [line_f32, cycle_f32, star_f32])

    dual_meshes_2D_bary_f32 = map(x -> generate_dual_f32(x, Barycenter()), [
      triangulated_grid(10, 10, 8, 8, Point3{Float32}),
      triangulated_grid(15, 15, 6, 6, Point3{Float32})])

    dual_meshes_2D_circum_f32 = map(x -> generate_dual_f32(x, Circumcenter()), [
      triangulated_grid(10, 10, 8, 8, Point2{Float32})])

    function test_hodge_solver_metal()
      @testset "Inverse Geometric Hodge" begin
        for sd in dual_meshes_2D_bary_f32
          V_1 = Float32.(I[1:ne(sd), 1])
          hdg = dec_hodge_star(1, sd, GeometricHodge(), Val(:Metal))
          inv_hdg = dec_inv_hodge_star(1, sd, GeometricHodge(), Val(:Metal))
          # Verify hdg * inv_hdg(V_1) ≈ -V_1 (sign convention: ⋆⁻¹ = -⋆ for 1-forms in 2D).
          # Uses AAFactorization (AppleAccelerate direct sparse QR).
          # atol=5f-6: tolerance appropriate for Float32 arithmetic (≈10× machine epsilon at this scale).
          @test all(isapprox.(
            hdg * inv_hdg(V_1),
            -V_1;
            atol = 5f-6))
        end
      end
    end

    # Cubical complex kernels on Metal. These are KernelAbstractions kernels, so
    # the point of these tests is that they compile for the GPU at all (a call
    # reaching a non-static error path fails with InvalidIRError) and that the
    # GPU result matches the CPU result.
    function test_cubical_metal()
      FT = Float32
      s2 = UniformCubicalComplex2D(6, 5, FT(1), FT(2))
      s3 = UniformCubicalComplex3D(3, 4, 5, FT(1), FT(2), FT(3))
      c2, c3 = UniformDECCache(s2), UniformDECCache3D(s3)
      gc2, gc3 = Adapt.adapt(MtlArray, c2), Adapt.adapt(MtlArray, c3)

      # Compare against the CPU result, elementwise for tuple-valued operators.
      agree(cpu::Tuple, gpu::Tuple) =
        all(isapprox(cpu[i], Array(gpu[i]); atol = 1f-3, rtol = 1f-3) for i in eachindex(cpu))
      agree(cpu, gpu) = isapprox(cpu, Array(gpu); atol = 1f-3, rtol = 1f-3)

      f0, f1, f2 = rand(FT, nv(s2)), rand(FT, ne(s2)), rand(FT, nquads(s2))
      g0, g1, g2, g3 = rand(FT, nv(s3)), rand(FT, ne(s3)), rand(FT, nquads(s3)), rand(FT, nboids(s3))
      m = MtlArray

      @testset "2D mesh operators" begin
        @test agree(exterior_derivative(Val(0), s2, f0), exterior_derivative(Val(0), s2, m(f0)))
        @test agree(exterior_derivative(Val(1), s2, f1), exterior_derivative(Val(1), s2, m(f1)))
        @test agree(wedge_product(Val(0), Val(1), s2, f0, f1), wedge_product(Val(0), Val(1), s2, m(f0), m(f1)))
        @test agree(wedge_product(Val(1), Val(1), s2, f1, f1), wedge_product(Val(1), Val(1), s2, m(f1), m(f1)))
        @test agree(wedge_product_dd(Val(0), Val(1), s2, f2, f1), wedge_product_dd(Val(0), Val(1), s2, m(f2), m(f1)))
        @test agree(wedge_product_dd(Val(1), Val(1), s2, f1, f1), wedge_product_dd(Val(1), Val(1), s2, m(f1), m(f1)))
        @test agree(wedge_product_pd(Val(1), Val(1), s2, f1, f1), wedge_product_pd(Val(1), Val(1), s2, m(f1), m(f1)))
        @test agree(sharp_dd(s2, f1), sharp_dd(s2, m(f1)))
        @test agree(flat_dp(s2, f2, f2), flat_dp(s2, m(f2), m(f2)))
        @test agree(flat_dd(s2, f2, f2), flat_dd(s2, m(f2), m(f2)))
      end

      # 2D hodge_star / dual_derivative on a mesh are sparse-matrix operators;
      # the kernel forms live on the cache.
      @testset "2D cached operators" begin
        @test agree(exterior_derivative(Val(0), c2, f0), exterior_derivative(Val(0), gc2, m(f0)))
        @test agree(exterior_derivative(Val(1), c2, f1), exterior_derivative(Val(1), gc2, m(f1)))
        @test agree(hodge_star(Val(0), c2, f0), hodge_star(Val(0), gc2, m(f0)))
        @test agree(hodge_star(Val(1), c2, f1), hodge_star(Val(1), gc2, m(f1)))
        @test agree(hodge_star(Val(2), c2, f2), hodge_star(Val(2), gc2, m(f2)))
        @test agree(inv_hodge_star(Val(1), c2, f1), inv_hodge_star(Val(1), gc2, m(f1)))
        @test agree(dual_derivative(Val(0), c2, f2), dual_derivative(Val(0), gc2, m(f2)))
        @test agree(dual_derivative(Val(1), c2, f1), dual_derivative(Val(1), gc2, m(f1)))
        @test agree(wedge_product(Val(1), Val(1), c2, f1, f1), wedge_product(Val(1), Val(1), gc2, m(f1), m(f1)))
        @test agree(sharp_dd(c2, f1), sharp_dd(gc2, m(f1)))
        @test agree(flat_dd(c2, f2, f2), flat_dd(gc2, m(f2), m(f2)))
        @test agree(interpolate_dp(Val(1), c2, f1), interpolate_dp(Val(1), gc2, m(f1)))
        @test agree(codifferential(Val(1), c2, f1), codifferential(Val(1), gc2, m(f1)))
        @test agree(laplacian(Val(0), c2, f0), laplacian(Val(0), gc2, m(f0)))
        @test agree(d_beta_mul(c2, f1), d_beta_mul(gc2, m(f1)))
        @test agree(no_flux_dual_derivative(Val(0), c2, f2), no_flux_dual_derivative(Val(0), gc2, m(f2)))
      end

      @testset "3D mesh operators" begin
        @test agree(exterior_derivative(Val(0), s3, g0), exterior_derivative(Val(0), s3, m(g0)))
        @test agree(exterior_derivative(Val(1), s3, g1), exterior_derivative(Val(1), s3, m(g1)))
        @test agree(exterior_derivative(Val(2), s3, g2), exterior_derivative(Val(2), s3, m(g2)))
        for (k, fk) in enumerate((g0, g1, g2, g3))
          @test agree(hodge_star(Val(k - 1), s3, fk), hodge_star(Val(k - 1), s3, m(fk)))
          @test agree(inv_hodge_star(Val(k - 1), s3, fk), inv_hodge_star(Val(k - 1), s3, m(fk)))
        end
        @test agree(dual_derivative(Val(0), s3, g3), dual_derivative(Val(0), s3, m(g3)))
        @test agree(dual_derivative(Val(1), s3, g2), dual_derivative(Val(1), s3, m(g2)))
        @test agree(dual_derivative(Val(2), s3, g1), dual_derivative(Val(2), s3, m(g1)))
        @test agree(wedge_product(Val(1), Val(1), s3, g1, g1), wedge_product(Val(1), Val(1), s3, m(g1), m(g1)))
        @test agree(wedge_product(Val(1), Val(2), s3, g1, g2), wedge_product(Val(1), Val(2), s3, m(g1), m(g2)))
        @test agree(wedge_product_dd(Val(0), Val(1), s3, g3, g2), wedge_product_dd(Val(0), Val(1), s3, m(g3), m(g2)))
        @test agree(sharp_dd(s3, g2), sharp_dd(s3, m(g2)))
        @test agree(flat_dp(s3, g3, g3, g3), flat_dp(s3, m(g3), m(g3), m(g3)))
      end

      @testset "3D cached operators" begin
        @test agree(exterior_derivative(Val(0), c3, g0), exterior_derivative(Val(0), gc3, m(g0)))
        @test agree(exterior_derivative(Val(1), c3, g1), exterior_derivative(Val(1), gc3, m(g1)))
        @test agree(hodge_star(Val(2), c3, g2), hodge_star(Val(2), gc3, m(g2)))
        @test agree(inv_hodge_star(Val(2), c3, g2), inv_hodge_star(Val(2), gc3, m(g2)))
        @test agree(dual_derivative(Val(1), c3, g2), dual_derivative(Val(1), gc3, m(g2)))
        @test agree(wedge_product(Val(1), Val(1), c3, g1, g1), wedge_product(Val(1), Val(1), gc3, m(g1), m(g1)))
      end
    end
    @testset "Metal" begin
      test_unary_operators(Val(:Metal), dual_meshes_1D_f32, dual_meshes_2D_bary_f32, dual_meshes_2D_circum_f32)
      test_hodge_solver_metal()
      test_binary_operators(Float32, Val(:Metal), MtlArray, 0.5e-6)
      test_binary_operators(Float16, Val(:Metal), MtlArray, 0.5e-3)
      test_cubical_metal()
    end
  else
    @info "Metal tests were not run, since the current device does not support Apple7 and Metal3."
  end
else
  @info "Metal tests were not run, since Sys.isapple() is false."
end

end
