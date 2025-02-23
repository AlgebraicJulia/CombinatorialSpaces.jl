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

expected_parts(s, binary_subdivision, binary_nv_ne_ntriangles)
expected_parts(s, cubic_subdivision, cubic_nv_ne_ntriangles)

# Subdivision integration
#------------------------

function test_residuals(s::HasDeltaSet2D, scheme::AbstractSubdivisionScheme)
  series = PrimalGeometricMapSeries(s, scheme, 4);
  
  md = MGData(series, sd -> ∇²(0, sd), 3, scheme)
  sd = finest_mesh(series)
  L = first(md.operators)
  
  Random.seed!(0)
  b = L*rand(nv(sd)) #put into range of the Laplacian for solvability
  u0 = zeros(nv(sd))
  
  mgv_lapl = dec_Δ⁻¹(Val{0}, series, scheme=scheme)
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
end

s = triangulated_grid(1,1,1/4,sqrt(3)/2*1/4,Point3D,false)

test_residuals(s, BinarySubdivision())
test_residuals(s, CubicSubdivision())

#=
function plot_residuals(s; cycles=1:50)
  function residuals(scheme::AbstractSubdivisionScheme)
    series = PrimalGeometricMapSeries(s, scheme, 4);
    md = MGData(series, sd -> ∇²(0, sd), 3, scheme)
    sd = finest_mesh(series)
    L = first(md.operators)
    b = L*rand(nv(sd)) #put into range of the Laplacian for solvability
    u0 = zeros(nv(sd))
    ress = map(cycles) do cyc
      u = multigrid_vcycles(u0,b,md,cyc)
      norm(L*u-b)/norm(b)
    end
  end

  bin_ress = residuals(BinarySubdivision())
  cub_ress = residuals(CubicSubdivision())

  f = Figure()
  ax = GLMakie.Axis(f[1,1];
    title = "Multigrid V-cycles",
    yscale = log10,
    ylabel = "log₁₀(relative error)",
    xlabel = "# cycles")
  lines!(ax, bin_ress, label="binary")
  lines!(ax, cub_ress, label="cubic")
  f[1,2] = Legend(f,ax,"Scheme")
  f
end
plot_residuals(s)

function heatmap_residuals(s; levels=2:4, cycles=1:50)
  function residuals(scheme::AbstractSubdivisionScheme, level, mesh_sizes)
    series = PrimalGeometricMapSeries(s, scheme, level);
    md = MGData(series, sd -> ∇²(0, sd), level-1, scheme)
    sd = finest_mesh(series)
    push!(mesh_sizes, nv(sd))
    L = first(md.operators)
    b = L*rand(nv(sd)) #put into range of the Laplacian for solvability
    u0 = zeros(nv(sd))
    ress = map(cycles) do cyc
      u = multigrid_wcycles(u0,b,md,cyc)
      norm(L*u-b)/norm(b)
    end
  end

  bin_sizes = []
  bin_ress = mapreduce(x -> residuals(BinarySubdivision(), x, bin_sizes), hcat, levels)
  cub_sizes = []
  cub_ress = mapreduce(x -> residuals(CubicSubdivision(), x, cub_sizes), hcat, levels)

  f = Figure()
  ax = GLMakie.Axis(f[1,1];
    title = "log10 of Multigrid W-cycles Residuals, Binary Subdivision",
    ylabel = "# vertices",
    yticks = (1:3, string.(bin_sizes)),
    xlabel = "# cycles")
  bin_hmp = heatmap!(ax, log10.(bin_ress), colormap = :jet)
  Colorbar(f[1,2], bin_hmp)

  ax = GLMakie.Axis(f[2,1];
    title = "log10 of Multigrid W-cycles Residuals, Cubic Subdivision",
    ylabel = "# vertices",
    yticks = (1:3, string.(cub_sizes)),
    xlabel = "# cycles")
  bin_hmp = heatmap!(ax, log10.(cub_ress), colormap = :jet)
  Colorbar(f[2,2], bin_hmp)
  f
end
heatmap_residuals(s)
=#

end
