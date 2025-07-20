module TestMultigrid
using Krylov, CombinatorialSpaces, LinearAlgebra, Test
using Random
using SparseArrays
using CombinatorialSpaces.Meshes: tri_345

using CombinatorialSpaces.Multigrid: UnarySubdivision, unary_subdivision, unary_subdivision_map

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

function expected_parts(s, subdivider, nv_ne_ntriangles)
  for _ in 1:4
    t = subdivider(s)
    @test (nv(t), ne(t), ntriangles(t)) == nv_ne_ntriangles(s)
    @test orient!(t)
    s = t
  end
end

expected_parts(s, unary_subdivision, unary_nv_ne_ntriangles)
expected_parts(s, binary_subdivision, binary_nv_ne_ntriangles)
expected_parts(s, cubic_subdivision, cubic_nv_ne_ntriangles)

# Subdivision integration
#------------------------

function test_residuals(s::HasDeltaSet2D, scheme::AbstractSubdivisionScheme)
  series = PrimalGeometricMapSeries(s, scheme, 4);

  md = MGData(series, sd -> ∇²(0, sd), 3)
  sd = finest_mesh(series)
  L = first(md.operators)

  Random.seed!(0)
  b = L*rand(nv(sd)) #put into range of the Laplacian for solvability
  u0 = zeros(nv(sd))

  mgv_lapl = dec_Δ⁻¹(Val{0}, series, scheme=scheme)
  u = mgv_lapl(b)
  @test norm(L*u-b)/norm(b) < 10^-6

  u = multigrid_vcycles(u0,b,md,5)
  @test norm(L*u-b)/norm(b) < 10^-7
  @debug "Relative residual for V: $(norm(L*u-b)/norm(b))"

  u = multigrid_wcycles(u0,b,md,5)
  @test norm(L*u-b)/norm(b) < 10^-7
  @debug "Relative residual for W: $(norm(L*u-b)/norm(b))"

  u = full_multigrid(b,md,5)
  @test norm(L*u-b)/norm(b) < 10^-6
  @debug "Relative residual for FMG_V: $(norm(L*u-b)/norm(b))"

  u = full_multigrid(b,md,5,cg,2)
  @test norm(L*u-b)/norm(b) < 10^-7
  @debug "Relative residual for FMG_W: $(norm(L*u-b)/norm(b))"
end

s = triangulated_grid(1,1,1/4,sqrt(3)/2*1/4,Point3d,false)

test_residuals(s, UnarySubdivision())
test_residuals(s, BinarySubdivision())
test_residuals(s, CubicSubdivision())

end
