module TestMultigrid
using Krylov, CombinatorialSpaces, LinearAlgebra, Test
using GeometryBasics: Point3, Point2
using Random
using SparseArrays
const Point3D = Point3{Float64}
const Point2D = Point2{Float64}
using CombinatorialSpaces.Meshes: tri_345

Random.seed!(0)

# Subdivision schemes unit tests
#-------------------------------

s = triangulated_grid(1,1,1,1,Point3D,false)
bin_s = binary_subdivision_map(s)
@test bin_s.matrix[1:nv(s), 1:nv(s)] == I
for e in 1:ne(s)
  @test findnz(bin_s.matrix[1:nv(s), nv(s)+e]) == ([s[e, :∂v1], s[e, :∂v0]], [0.5, 0.5])
end

binary_expected_parts(s) =
  (nv(s) + ne(s), 2*ne(s) + 3*ntriangles(s), 4*ntriangles(s))

cubic_expected_parts(s) =
  (nv(s) + 2*ne(s) + ntriangles(s), 3*ne(s) + 9*ntriangles(s), 9*ntriangles(s))

function expected_parts(s, subdivider, expected_parts)
  for _ in 1:4
    t = subdivider(s)
    @test (nv(t), ne(t), ntriangles(t)) == expected_parts(s)
    @test orient!(t)
    s = t
  end
end

expected_parts(s, binary_subdivision, binary_expected_parts)
expected_parts(s, cubic_subdivision, cubic_expected_parts)

# Binary subdivision integration
#-------------------------------

s = triangulated_grid(1,1,1/4,sqrt(3)/2*1/4,Point3D,false)
series = PrimalGeometricMapSeries(s, binary_subdivision_map, 4);

md = MultigridData(series, sd -> ∇²(0, sd), 3, 4.0) #3, (and 5 below) chosen empirically, presumably there's deep lore and chaos here
sd = finest_mesh(series)
L = first(md.operators)

Random.seed!(0)
b = L*rand(nv(sd)) #put into range of the Laplacian for solvability
u0 = zeros(nv(sd))

mgv_lapl = dec_Δ⁻¹(Val{0}, series, denominator = 4.0)
u = mgv_lapl(b)
@test norm(L*u-b)/norm(b) < 10^-6

u = multigrid_vcycles(u0,b,md,5)
@test norm(L*u-b)/norm(b) < 10^-5
@debug "Relative error for V: $(norm(L*u-b)/norm(b))"

u = multigrid_wcycles(u0,b,md,5)
@test norm(L*u-b)/norm(b) < 10^-6
@debug "Relative error for W: $(norm(L*u-b)/norm(b))"

u = full_multigrid(b,md,5)
@test norm(L*u-b)/norm(b) < 10^-4
@debug "Relative error for FMG_V: $(norm(L*u-b)/norm(b))"

u = full_multigrid(b,md,5,cg,2)
@test norm(L*u-b)/norm(b) < 10^-6
@debug "Relative error for FMG_W: $(norm(L*u-b)/norm(b))"

# Cubic subdivision integration
#------------------------------

s = triangulated_grid(1,1,1/4,sqrt(3)/2*1/4,Point3D,false)
series = PrimalGeometricMapSeries(s, cubic_subdivision_map, 4);

md = MultigridData(series, sd -> ∇²(0, sd), 3, 9.0) #3, (and 5 below) chosen empirically, presumably there's deep lore and chaos here
sd = finest_mesh(series)
L = first(md.operators)

Random.seed!(0)
u0 = zeros(nv(sd))
b = L*rand(nv(sd)) #put into range of the Laplacian for solvability

mgv_lapl = dec_Δ⁻¹(Val{0}, series, denominator = 9.0)
u = mgv_lapl(b)
@test norm(L*u-b)/norm(b) < 10^-6

u = multigrid_vcycles(u0,b,md,5)
@debug "Relative error for V: $(norm(L*u-b)/norm(b))"
@test norm(L*u-b)/norm(b) < 10^-6

u = multigrid_wcycles(u0,b,md,5)
@debug "Relative error for W: $(norm(L*u-b)/norm(b))"
@test norm(L*u-b)/norm(b) < 10^-6

u = full_multigrid(b,md,5)
@test norm(L*u-b)/norm(b) < 10^-6
@debug "Relative error for FMG_V: $(norm(L*u-b)/norm(b))"

u = full_multigrid(b,md,5,cg,2)
@test norm(L*u-b)/norm(b) < 10^-6
@debug "Relative error for FMG_W: $(norm(L*u-b)/norm(b))"

#=
function plot_residuals(s, levels=1:50)
  function residuals(subdivision_map, denominator)
    series = PrimalGeometricMapSeries(s, subdivision_map, 4);
    md = MultigridData(series, sd -> ∇²(0, sd), 3, denominator)
    sd = finest_mesh(series)
    L = first(md.operators)
    b = L*rand(nv(sd)) #put into range of the Laplacian for solvability
    u0 = zeros(nv(sd))
    ress = map(levels) do lev
      u = multigrid_vcycles(u0,b,md,lev)
      norm(L*u-b)/norm(b)
    end
  end

  bin_ress = residuals(binary_subdivision_map, 4.0)
  cub_ress = residuals(cubic_subdivision_map, 9.0)

  f = Figure()
  ax = GLMakie.Axis(f[1,1];
    title="Multigrid V-cycles",
    yscale=log10,
    ylabel="log₁₀(relative error)",
    xlabel="# max levels")
  lines!(ax, bin_ress, label="binary")
  lines!(ax, cub_ress, label="cubic")
  f[1,2] = Legend(f,ax,"Scheme")
  f
end
plot_residuals(s)
=#

end
