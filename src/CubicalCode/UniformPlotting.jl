using GeometryBasics
import GeometryBasics.Mesh

using Makie
import Makie: convert_arguments

function GeometryBasics.Mesh(s::UniformCubicalComplex2D)
  ps = interior(Val(0), points(s), s)

  qs = QuadFace{Int}[]
  for y in 1:nyquads(s)
    for x in 1:nxquads(s)
      is_halo_quad(s, x, y) && continue
      push!(qs, map(v -> vert_to_real_vert(s, v), quad_vertices(s, x, y)))
    end
  end

  GeometryBasics.Mesh(ps, qs)
end

convert_arguments(P::Union{Type{<:Makie.Wireframe}, Type{<:Makie.Mesh}, Type{<:Makie.Scatter}}, s::UniformCubicalComplex2D) = convert_arguments(P, GeometryBasics.Mesh(s))

plottype(::UniformCubicalComplex2D) = GeometryBasics.Mesh

function plot_wireframe(s::UniformCubicalComplex2D; kwargs...)
  fig = Figure();
  ax = CairoMakie.Axis(fig[1, 1])
  CairoMakie.wireframe!(ax, s; kwargs...)
  fig
end

# TODO: Add in generic kwargs
function plot_zeroform(s::UniformCubicalComplex2D, f)
  fig = Figure();
  ax = CairoMakie.Axis(fig[1, 1])
  msh = CairoMakie.mesh!(ax, s, color=interior(Val(0), f, s), colormap=:jet)
  Colorbar(fig[1, 2], msh)
  fig
end

# Plot only the interior of a 1-form, since the halo values are not meaningful for visualization
function plot_oneform(s::UniformCubicalComplex2D, alpha; lengthscale = 1, normalize = true)
  dps = dual_points(s)
  interdps = interior(Val(2), dps, s)
  x = map(a -> a[1], interdps)
  y = map(a -> a[2], interdps)

  X = zeros(nquads(s)); Y = zeros(nquads(s))

  sharp_dd!(X, Y, s, alpha)

  X = interior(Val(2), X, s)
  Y = interior(Val(2), Y, s)

  color = sqrt.(X.^2 + Y.^2)

  fig = Figure();
  ax = CairoMakie.Axis(fig[1,1])
  wireframe!(ax, s; alpha=0.5)
  arrows2d!(ax, x, y, X, Y, color = color, colormap=:jet, lengthscale = lengthscale, normalize = normalize)
  fig
end

function plot_xy_oneform(s::UniformCubicalComplex2D, alpha)
  dps = dual_points(s)
  interdps = interior(Val(2), dps, s)
  x = map(a -> a[1], interdps)
  y = map(a -> a[2], interdps)

  X = zeros(nquads(s)); Y = zeros(nquads(s))

  sharp_dd!(X, Y, s, alpha)

  X = interior(Val(2), X, s)
  Y = interior(Val(2), Y, s)

  fig = Figure();
  axX = CairoMakie.Axis(fig[1,1]; title = "X Component", xlabel = "x", ylabel = "y")
  msh = CairoMakie.scatter!(axX, interdps, color=X, colormap=:jet)
  Colorbar(fig[1, 2], msh)

  axY = CairoMakie.Axis(fig[2,1]; title = "Y Component", xlabel = "x", ylabel = "y")
  msh = CairoMakie.scatter!(axY, interdps, color=Y, colormap=:jet)
  Colorbar(fig[2, 2], msh)

  fig
end

function plot_twoform(s::UniformCubicalComplex2D, f)
  fig = Figure();
  ax = CairoMakie.Axis(fig[1, 1])

  dps = dual_points(s)
  interdps = interior(Val(2), dps, s)
  x = map(a -> a[1], interdps)
  y = map(a -> a[2], interdps)

  msh = CairoMakie.heatmap!(ax, x, y, interior(Val(2), f, s), colormap=:jet)
  Colorbar(fig[1, 2], msh)
  fig
end

plot_dual_zeroform(s::UniformCubicalComplex2D, f) = plot_twoform(s, f)

function create_gif(solution, file_name; frames = length(solution), framerate = 15)
  t_e = solution.t[end]

  fig = Figure()
  ax = CairoMakie.Axis(fig[1,1])
  msh = CairoMakie.mesh!(ax, s, color=interior(Val(0), first(solution), s), colormap=:jet, colorrange=extrema(first(solution)))
  Colorbar(fig[1,2], msh)
  CairoMakie.record(fig, file_name, 1:frames; framerate = framerate) do t
    msh.color = interior(Val(0), solution(t * t_e / frames), s)
  end
end
