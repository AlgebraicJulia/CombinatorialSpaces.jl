using Test
using SparseArrays

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../src/CubicalCode/UniformKernelDEC.jl")

@testset "UniformMatrixDEC" begin

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
  # TODO: Add test to ensure signs are correct
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

  # Compute the divergence of a constant vector field (should be zero)
  @test all(cd1 * ones(ne(s)) .== 0)

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

  # Test that wedge of dx and dy is constant 2-form
  V = ones(ne(s))
  V[1:nxedges(s)] .= 0 # No horizontal motion on the vertical edges

  W = ones(ne(s))
  W[end-nxedges(s)+1:end] .= 0 # No vertical motion on the horizontal edges

  # W is dx and V is dy, so their wedge should be 1 everywhere
  out = wedge_product(Val(1), Val(1), s, W, V)
  @test all(out .== 1.0)

  out = wedge_product(Val(1), Val(1), s, V, W)
  @test all(out .== -1.0)

  # Test that sharp works
  u = ones(ne(s))
  X, Y = sharp_dd(s, u)

  @test X[coord_to_quad(s, 2, 2)] == -4.0
  @test X[coord_to_quad(s, 3, 2)] == -4.0

  @test Y[coord_to_quad(s, 2, 2)] == 4.0
  @test Y[coord_to_quad(s, 2, 3)] == 4.0

  # Test that flat works
  X = 4 * ones(nquads(s))
  Y = 4 * ones(nquads(s))
  u = flat_dp(s, X, Y)
  @test all(u .== 1.0)

  dd0 = dual_derivative(Val(0), s)
  dd1 = dual_derivative(Val(1), s)

  boundary_idxs = findall(x -> x != 0, dd0 * ones(nquads(s)))

  u = ones(ne(s))
  u[boundary_idxs] .= 0.5
  v = zeros(ne(s))
  X, Y = sharp_dd(s, u)
  u = flat_dp(s, X, Y)

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

  u = ones(ne(s))
  f = ones(nquads(s))
  res = zeros(ne(s))

  res = wedge_product_dd(Val(0), Val(1), s, f, u)
  @test all(res .== 1.0)

  res = wedge_product_dd(Val(0), Val(1), s, f, 5 * u)
  @test all(res .== 5.0)

  res = wedge_product_dd(Val(0), Val(1), s, 2 * f, 5 * u)
  @test all(res .== 10.0)

  f = [y for y in 1:nyquads(s) for x in 1:nxquads(s)]
  res = wedge_product_dd(Val(0), Val(1), s, f, u)
  @test xedges(s, res)[1] == 1.0
  @test xedges(s, res)[end] == 4.0

  @test yedges(s, res)[1] == 1.0
  @test yedges(s, res)[end] == 4.0
end

@testset "UniformMatrixDEC with Halo" begin
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
  # TODO: Add test to ensure signs are correct
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
  @test all(abs.(diag(ihs1 * hs1) .- 1) .< 1e-12)

  ihs2 = inv_hodge_star(Val(2), s)
  @test all(abs.(diag(ihs2 * hs2) .- 1) .< 1e-12)

  # Codifferentials: check sizes and definitions
  cd1 = codifferential(Val(1), s)
  @test size(cd1) == (nv(s), ne(s))

  cd2 = codifferential(Val(2), s)
  @test size(cd2) == (ne(s), nquads(s))

  # Compute the divergence of a constant vector field (should be zero)
  @test all(interior(cd1 * ones(ne(s)), s) .== 0)

  # Laplacians: shapes and basic properties
  L0 = laplacian(Val(0), s)
  L1 = laplacian(Val(1), s)
  L2 = laplacian(Val(2), s)

  @test size(L0) == (nv(s), nv(s))
  @test size(L1) == (ne(s), ne(s))
  @test size(L2) == (nquads(s), nquads(s))

  # Laplacian of constant should be (near) zero for interior-preserving grid
  @test all(abs.(L0 * ones(nv(s))) .< 1e-12)

  # Test that the halo values are set correctly by set_halo! and set_periodic!
  f = zeros(nv(s))
  set_halo!(f, Val(0), s, 1.0, ALL)
  g = reshape(f, (nx(s), ny(s)))
  @test all(g[1, :] .== 1.0)
  @test all(g[end, :] .== 1.0)
  @test all(g[:, 1] .== 1.0)
  @test all(g[:, end] .== 1.0)

  # Tests for periodicity of 0-forms
  f = [0 0 0 0 0 0 0;
       0 2 1 1 1 1 0;
       0 2 1 1 1 1 0;
       0 2 1 1 1 1 0;
       0 2 1 1 1 1 0;
       0 2 1 1 1 1 0;
       0 0 0 0 0 0 0] |> vec
  set_periodic!(f, Val(0), s, NORTHSOUTH)
  g = reshape(f, (nx(s), ny(s)))
  @test all(g[2:end-1, 1:3] .== g[2:end-1, end-2:end])

  f = [0 0 0 0 0 0 0;
       0 2 2 2 2 2 0;
       0 1 1 1 1 1 0;
       0 1 1 1 1 1 0;
       0 1 1 1 1 1 0;
       0 1 1 1 1 1 0;
       0 0 0 0 0 0 0] |> vec
  set_periodic!(f, Val(0), s, EASTWEST)
  g = reshape(f, (nx(s), ny(s)))
  @test all(g[1:3, 2:end-1] .== g[end-2:end, 2:end-1])
  
  # Tests for periodicity of 1-forms 
  # TODO: Fix these tests to reflect the fact that we're also replacing a real edge
  f = vcat(collect(1:nxedges(s)), collect(1:nyedges(s)));
  set_periodic!(f, Val(1), s, NORTHSOUTH);
  g = reshape(f[1:nxedges(s)], (nxe(s), ny(s)))
  @test all(g[:, 1:2] .== g[:, end - 1:end])
  @test all(g[:, end] .== g[:, 2])

  g = reshape(f[nxedges(s)+1:end], (nx(s), nye(s)))
  @test all(g[:, 1] .== g[:, end - 1])
  @test all(g[:, end] .== g[:, 2])

  f = vcat(collect(1:nxedges(s)), collect(1:nyedges(s)));
  set_periodic!(f, Val(1), s, EASTWEST);
  g = reshape(f[1:nxedges(s)], (nxe(s), ny(s)))
  @test all(g[1, :] .== g[end - 1, :])
  @test all(g[end, :] .== g[2, :])

  g = reshape(f[nxedges(s)+1:end], (nx(s), nye(s)))
  @test all(g[1, :] .== g[end - 1, :])
  @test all(g[end, :] .== g[2, :])

  # Tests for periodicity of 2-forms
  f = repeat(collect(1:nxquads(s)), inner = nyquads(s))
  set_periodic!(f, Val(2), s, NORTHSOUTH)
  g = reshape(f, (nxquads(s), nyquads(s)))
  @test all(g[:, 1] .== g[:, end - 1])
  @test all(g[:, end] .== g[:, 2])

  f = repeat(collect(1:nxquads(s)), nyquads(s))
  set_periodic!(f, Val(2), s, EASTWEST)
  g = reshape(f, (nxquads(s), nyquads(s)))
  @test all(g[1, :] .== g[end - 1, :])
  @test all(g[end, :] .== g[2, :])
end

@testset "UniformMatrixDEC with Large Halo" begin
  s = UniformCubicalComplex2D(10, 10, 1.0, 1.0; halo_x = 2, halo_y = 2)
  f = repeat(collect(1:nxquads(s)), inner = nyquads(s))
  set_periodic!(f, Val(2), s, NORTHSOUTH)
  g = reshape(f, (nxquads(s), nyquads(s)))
  @test all(g[:, 1:2] .== g[:, end - 3:end - 2])
  @test all(g[:, end-1:end] .== g[:, 3:4])

  f = repeat(collect(1:nxquads(s)), nyquads(s))
  set_periodic!(f, Val(2), s, EASTWEST)
  g = reshape(f, (nxquads(s), nyquads(s)))
  @test all(g[1:2, :] .== g[end - 3:end - 2, :])
  @test all(g[end-1:end, :] .== g[3:4, :])

  f = collect(1:nquads(s));
  set_periodic!(f, Val(2), s, EASTWEST);
  g = reshape(f, (nxquads(s), nyquads(s)))
  @test all(g[1:2, :] .== g[end - 3:end - 2, :])
  @test all(g[end-1:end, :] .== g[3:4, :])
end
