using GeometryBasics
import GeometryBasics.Mesh

using Makie
import Makie: convert_arguments

function GeometryBasics.Mesh(s::UniformCubicalComplex2D)
  ps = interior(Val(0), points(s), s)

  qs = QuadFace{Int}[]
  for y in 1:nyq(s)
    for x in 1:nxq(s)
      is_halo_quad(s, x, y) && continue
      push!(qs, map(v -> vert_to_real_vert(s, v), quad_vertices(s, x, y)))
    end
  end

  GeometryBasics.Mesh(ps, qs)
end

convert_arguments(P::Union{Type{<:Makie.Wireframe}, Type{<:Makie.Mesh}, Type{<:Makie.Scatter}}, s::UniformCubicalComplex2D) = convert_arguments(P, GeometryBasics.Mesh(s))

plottype(::UniformCubicalComplex2D) = GeometryBasics.Mesh

function plot_wireframe(
  s::UniformCubicalComplex2D;
  figure_kwargs = (;),
  axis_kwargs = (;),
  wireframe_kwargs = (;),
  kwargs..., # Backward-compatible passthrough to wireframe!
)
  fig = Figure(; figure_kwargs...)
  ax = CairoMakie.Axis(fig[1, 1]; axis_kwargs...)
  wf_kwargs = merge(wireframe_kwargs, (; kwargs...))
  CairoMakie.wireframe!(ax, s; wf_kwargs...)
  fig
end

function plot_zeroform(
  s::UniformCubicalComplex2D,
  f;
  figure_kwargs = (;),
  axis_kwargs = (;),
  mesh_kwargs = (;),
  colorbar_kwargs = (;),
)
  fig = Figure(; figure_kwargs...)
  ax = CairoMakie.Axis(fig[1, 1]; axis_kwargs...)
  mesh_defaults = (
    color = interior(Val(0), f, s),
    colormap = :jet,
  )
  msh = CairoMakie.mesh!(ax, s; merge(mesh_defaults, mesh_kwargs)...)
  Colorbar(fig[1, 2], msh; colorbar_kwargs...)
  fig
end

# Plot only the interior of a 1-form, since the halo values are not meaningful for visualization
function plot_oneform(
  s::UniformCubicalComplex2D,
  alpha;
  lengthscale = 1,
  normalize = true,
  figure_kwargs = (;),
  axis_kwargs = (;),
  wireframe_kwargs = (;),
  arrows_kwargs = (;),
)
  dps = dual_points(s)
  interdps = interior(Val(2), dps, s)
  x = map(a -> a[1], interdps)
  y = map(a -> a[2], interdps)

  X, Y = sharp_dd(s, alpha)

  X = interior(Val(2), X, s)
  Y = interior(Val(2), Y, s)

  color = sqrt.(X.^2 + Y.^2)

  fig = Figure(; figure_kwargs...)
  ax = CairoMakie.Axis(fig[1, 1]; axis_kwargs...)
  wireframe_defaults = (
    alpha = 0.5,
  )
  arrows_defaults = (
    color = color,
    colormap = :jet,
    lengthscale = lengthscale,
    normalize = normalize,
  )
  wireframe!(ax, s; merge(wireframe_defaults, wireframe_kwargs)...)
  arrows2d!(ax, x, y, X, Y; merge(arrows_defaults, arrows_kwargs)...)
  fig
end

function plot_xy_oneform(
  s::UniformCubicalComplex2D,
  alpha;
  figure_kwargs = (;),
  axis_x_kwargs = (;),
  axis_y_kwargs = (;),
  heatmap_x_kwargs = (;),
  heatmap_y_kwargs = (;),
  colorbar_x_kwargs = (;),
  colorbar_y_kwargs = (;),
)
  dps = dual_points(s)
  interdps = interior(Val(2), dps, s)
  x = map(a -> a[1], interdps)
  y = map(a -> a[2], interdps)

  X, Y = sharp_dd(s, alpha)

  X = interior(Val(2), X, s)
  Y = interior(Val(2), Y, s)

  fig = Figure(; figure_kwargs...)
  axis_x_defaults = (
    title = "X Component",
    xlabel = "x",
    ylabel = "y",
  )
  axis_y_defaults = (
    title = "Y Component",
    xlabel = "x",
    ylabel = "y",
  )
  heatmap_defaults = (
    colormap = :jet,
  )

  axX = CairoMakie.Axis(fig[1, 1]; merge(axis_x_defaults, axis_x_kwargs)...)
  mshX = CairoMakie.heatmap!(axX, x, y, X; merge(heatmap_defaults, heatmap_x_kwargs)...)
  Colorbar(fig[1, 2], mshX; colorbar_x_kwargs...)

  axY = CairoMakie.Axis(fig[2, 1]; merge(axis_y_defaults, axis_y_kwargs)...)
  mshY = CairoMakie.heatmap!(axY, x, y, Y; merge(heatmap_defaults, heatmap_y_kwargs)...)
  Colorbar(fig[2, 2], mshY; colorbar_y_kwargs...)

  fig
end

function plot_twoform(
  s::UniformCubicalComplex2D,
  f;
  figure_kwargs = (;),
  axis_kwargs = (;),
  heatmap_kwargs = (;),
  colorbar_kwargs = (;),
)
  fig = Figure(; figure_kwargs...)
  ax = CairoMakie.Axis(fig[1, 1]; axis_kwargs...)

  dps = dual_points(s)
  interdps = interior(Val(2), dps, s)
  x = map(a -> a[1], interdps)
  y = map(a -> a[2], interdps)

  heatmap_defaults = (
    colormap = :jet,
  )
  msh = CairoMakie.heatmap!(ax, x, y, interior(Val(2), f, s); merge(heatmap_defaults, heatmap_kwargs)...)
  Colorbar(fig[1, 2], msh; colorbar_kwargs...)
  fig
end

function plot_dual_zeroform(s::UniformCubicalComplex2D, f; kwargs...)
  return plot_twoform(s, f; kwargs...)
end

function create_gif(
  solution,
  file_name;
  frames = length(solution),
  framerate = 15,
  figure_kwargs = (;),
  axis_kwargs = (;),
  mesh_kwargs = (;),
  colorbar_kwargs = (;),
  record_kwargs = (;),
)
  t_e = solution.t[end]

  fig = Figure(; figure_kwargs...)
  ax = CairoMakie.Axis(fig[1, 1]; axis_kwargs...)
  mesh_defaults = (
    color = interior(Val(0), first(solution), s),
    colormap = :jet,
    colorrange = extrema(first(solution)),
  )
  msh = CairoMakie.mesh!(ax, s; merge(mesh_defaults, mesh_kwargs)...)
  Colorbar(fig[1, 2], msh; colorbar_kwargs...)
  CairoMakie.record(fig, file_name, 1:frames; framerate = framerate, record_kwargs...) do t
    msh.color = interior(Val(0), solution(t * t_e / frames), s)
  end
end
