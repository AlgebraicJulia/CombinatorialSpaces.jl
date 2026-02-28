module TestMultigrid
using Krylov, CombinatorialSpaces, LinearAlgebra, Test
using Random
using SparseArrays
using CombinatorialSpaces.CombMeshes: tri_345

using CombinatorialSpaces.Multigrid: UnarySubdivision, unary_subdivision, unary_subdivision_map, subdivision

Random.seed!(0)

# Subdivision schemes unit tests
#-------------------------------

s = triangulated_grid(1,1,1,1,Point3d,false)
bin_s = binary_subdivision_map(s)
@test bin_s.matrix[1:nv(s), 1:nv(s)] == I
for e in 1:ne(s)
  @test findnz(bin_s.matrix[1:nv(s), nv(s)+e]) == ([s[e, :∂v1], s[e, :∂v0]], [0.5, 0.5])
end

unary_nv_ne_ntriangles(s) =
  (nv(s), ne(s), ntriangles(s))

binary_nv_ne_ntriangles(s) =
  (nv(s) + ne(s), 2*ne(s) + 3*ntriangles(s), 4*ntriangles(s))

cubic_nv_ne_ntriangles(s) =
  (nv(s) + 2*ne(s) + ntriangles(s), 3*ne(s) + 9*ntriangles(s), 9*ntriangles(s))

function test_expected_parts(s, subdivider, nv_ne_ntriangles)
  for _ in 1:4
    t = subdivider(s)
    @test (nv(t), ne(t), ntriangles(t)) == nv_ne_ntriangles(s)
    @test orient!(t)
    s = t
  end
end

test_expected_parts(s, unary_subdivision, unary_nv_ne_ntriangles)
test_expected_parts(s, binary_subdivision, binary_nv_ne_ntriangles)
test_expected_parts(s, cubic_subdivision, cubic_nv_ne_ntriangles)

# Subdivision integration
#------------------------

function fast_laplace_beltrami(sd)
  -1 * dec_inv_hodge_star(0,sd) * dec_dual_derivative(1,sd) * dec_hodge_star(1,sd) * dec_differential(0,sd)
end

function fast_laplace_derham(sd)
  dec_inv_hodge_star(0,sd) * dec_dual_derivative(1,sd) * dec_hodge_star(1,sd) * dec_differential(0,sd)
end

function test_residuals(s::HasDeltaSet2D, scheme::AbstractSubdivisionScheme)
  series = PrimalGeometricMapSeries(s, scheme, 4);

  md = MGData(series, fast_laplace_beltrami, 3)
  sd = finest_mesh(series)
  L = first(md.operators)

  Random.seed!(0)
  b = L*rand(nv(sd))
  u0 = zeros(nv(sd))

  mgv_lapl = dec_Δ⁻¹(Val(0), series, scheme=scheme)
  u = mgv_lapl(b)
  @test norm(L*u-b)/norm(b) < 1e-6

  u = multigrid_vcycles(u0,b,md,5)
  @test norm(L*u-b)/norm(b) < 1e-7
  @debug "Relative residual for V: $(norm(L*u-b)/norm(b))"

  u = multigrid_wcycles(u0,b,md,5)
  @test norm(L*u-b)/norm(b) < 1e-7
  @debug "Relative residual for W: $(norm(L*u-b)/norm(b))"

  u = full_multigrid(b,md,5)
  @test norm(L*u-b)/norm(b) < 1e-6
  @debug "Relative residual for FMG_V: $(norm(L*u-b)/norm(b))"

  u = full_multigrid(b,md,5,cg,2)
  @test norm(L*u-b)/norm(b) < 1e-7
  @debug "Relative residual for FMG_W: $(norm(L*u-b)/norm(b))"
end

s = triangulated_grid(1,1,1/4,sqrt(3)/2*1/4,Point3d,false)

test_residuals(s, UnarySubdivision())
test_residuals(s, BinarySubdivision())
test_residuals(s, CubicSubdivision())

# Equivalence between constructors
#---------------------------------
s = triangulated_grid(1, 1, 1/4, sqrt(3)/2 * 1/4, Point3d, false)
series = PrimalGeometricMapSeries(s, BinarySubdivision(), 4);
md_via_series = MGData(series, fast_laplace_beltrami, 3)
md_directly = MultigridData(s, BinarySubdivision(), 4, fast_laplace_beltrami, 3)

@test md_directly.operators == md_via_series.operators
@test md_directly.prolongations == md_via_series.prolongations
@test md_directly.restrictions == md_via_series.restrictions
@test md_directly.steps == md_via_series.steps

md_via_series_allocs = @allocated begin
  series = PrimalGeometricMapSeries(s, BinarySubdivision(), 4);
  MGData(series, fast_laplace_beltrami, 3);
end;
md_directly_allocs = @allocated begin
  MultigridData(s, BinarySubdivision(), 4, fast_laplace_beltrami, 3);
end;
@test md_directly_allocs < md_via_series_allocs

# Galerkin optimization
#----------------------

function test_galerkin(s::HasDeltaSet2D, scheme::AbstractSubdivisionScheme)
  md = MultigridData(s, scheme, 4, fast_laplace_beltrami, 3; mode=GalerkinMode())
  for _ in 1:4
    s = subdivision(s, scheme)
  end
  sd = dualize(s)
  L = first(md.operators)

  Random.seed!(0)
  b = L*rand(nv(sd))
  u0 = zeros(nv(sd))

  u = multigrid_vcycles(u0,b,md,5)
  @test norm(L*u-b)/norm(b) < 2e-6
  @debug "Relative residual for V: $(norm(L*u-b)/norm(b))"

  u = multigrid_wcycles(u0,b,md,5)
  @test norm(L*u-b)/norm(b) < 7e-7
  @debug "Relative residual for W: $(norm(L*u-b)/norm(b))"

  u = full_multigrid(b,md,5)
  @test norm(L*u-b)/norm(b) < 1e-3
  @debug "Relative residual for FMG_V: $(norm(L*u-b)/norm(b))"

  u = full_multigrid(b,md,5,cg,2)
  @test norm(L*u-b)/norm(b) < 8e-7
  @debug "Relative residual for FMG_W: $(norm(L*u-b)/norm(b))"
end

s = triangulated_grid(1, 1, 1/4, sqrt(3)/2 * 1/4, Point3d, false)

test_galerkin(s, UnarySubdivision())
test_galerkin(s, BinarySubdivision())
test_galerkin(s, CubicSubdivision())

# Divergence from Default Krylov.jl Behavior. (No iterations). Issue #178
#------------------------------------------------------------------------

s = triangulated_grid(1,1,1/4,sqrt(3)/2*1/4,Point3d,false)
bin_series = PrimalGeometricMapSeries(s, BinarySubdivision(), 4);
md_zero_iterations = MGData(bin_series, fast_laplace_beltrami, 0)
md_one_iteration = MGData(bin_series, fast_laplace_beltrami, 1)
sd = finest_mesh(bin_series)
L = first(md_zero_iterations.operators)
Random.seed!(0)
b = L*rand(nv(sd))
u0 = zeros(nv(sd))
u_zero_iterations = multigrid_vcycles(u0,b,md_zero_iterations,5)
u_one_iteration = multigrid_vcycles(u0,b,md_one_iteration,5)
relative_residual_zero_iterations = norm(L*u_zero_iterations-b)/norm(b)
relative_residual_one_iteration = norm(L*u_one_iteration-b)/norm(b)

# Test that no iterations of `cg` are performed by checking that the residual
# is higher than when one iteration is performed.
@test relative_residual_one_iteration < relative_residual_zero_iterations

end
