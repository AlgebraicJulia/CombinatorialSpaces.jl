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

using CairoMakie
using KernelAbstractions

# Make sure all dependencies are included
# include("UniformMesh.jl")
# include("UniformMesh3D.jl")

"""
    plot_dual_zeroform_slice(s::UniformCubicalComplex3D, f, align::Align, slice_idx::Int; kwargs...)

Plots a 2D heatmap of a slice of a 3D dual zero-form (data defined on boid centers).

# Arguments
- `s::UniformCubicalComplex3D`: The 3D mesh.
- `f`: A 1D array containing the data for all boids in the mesh.
- `align::Align`: The axis normal to the slice (`X_ALIGN`, `Y_ALIGN`, or `Z_ALIGN`).
- `slice_idx::Int`: The integer index of the slice along the specified alignment axis.
- `figure_kwargs`, `axis_kwargs`, `heatmap_kwargs`, `colorbar_kwargs`: Makie keyword arguments.
"""
function plot_dual_zeroform_slice(
  s::UniformCubicalComplex3D,
  f,
  align::Align,
  slice_idx::Int;
  figure_kwargs = (;),
  axis_kwargs = (;),
  heatmap_kwargs = (;),
  colorbar_kwargs = (;),
)
    # 1. Determine axes ranges, labels, and physical coordinates directly
    if align == X_ALIGN
        x_range, y_range = 1:nyb(s), 1:nzb(s)
        x_ax_label, y_ax_label = "Y", "Z"
        title_text = "Slice at X = $slice_idx"
        
        # Physical coordinates: Y is horizontal axis, Z is vertical axis
        plot_x_axis = [dual_point(s, slice_idx, y, 1)[2] for y in x_range]
        plot_y_axis = [dual_point(s, slice_idx, 1, z)[3] for z in y_range]

    elseif align == Y_ALIGN
        x_range, y_range = 1:nxb(s), 1:nzb(s)
        x_ax_label, y_ax_label = "X", "Z"
        title_text = "Slice at Y = $slice_idx"
        
        # Physical coordinates: X is horizontal axis, Z is vertical axis
        plot_x_axis = [dual_point(s, x, slice_idx, 1)[1] for x in x_range]
        plot_y_axis = [dual_point(s, 1, slice_idx, z)[3] for z in y_range]

    else # align == Z_ALIGN
        x_range, y_range = 1:nxb(s), 1:nyb(s)
        x_ax_label, y_ax_label = "X", "Y"
        title_text = "Slice at Z = $slice_idx"
        
        # Physical coordinates: X is horizontal axis, Y is vertical axis
        plot_x_axis = [dual_point(s, x, 1, slice_idx)[1] for x in x_range]
        plot_y_axis = [dual_point(s, 1, y, slice_idx)[2] for y in y_range]
    end


    # Pre-allocate a 2D array for the sliced data
    data_slice = zeros(Float64, length(x_range), length(y_range))

    # Extract the data for the specified slice
    for i in x_range, j in y_range
        b_coord = if align == X_ALIGN
            (slice_idx, i, j)
        elseif align == Y_ALIGN
            (i, slice_idx, j)
        else # align == Z_ALIGN
            (i, j, slice_idx)
        end
        
        # Check if the boid coordinate is valid before accessing
        if valid_boid(s, b_coord...)
            boid_idx = coord_to_boid(s, b_coord...)
            data_slice[i, j] = f[boid_idx]
        else
            # If the slice index is out of bounds, data will be zero
            data_slice[i, j] = NaN # Use NaN for out-of-bounds to get blank spots
        end
    end

    # --- Plotting ---
    fig = Figure(; figure_kwargs...)
    
    axis_defaults = (
        title = title_text,
        xlabel = x_ax_label,
        ylabel = y_ax_label,
        # aspect = DataAspect(), # Ensure correct aspect ratio, TODO: Do we include this?
    )
    ax = CairoMakie.Axis(fig[1, 1]; merge(axis_defaults, axis_kwargs)...)

    heatmap_defaults = (
        colormap = :jet,
    )
    
    # The data matrix might need to be transposed depending on the alignment
    # Makie plots columns along x and rows along y.
    final_data = (align == Y_ALIGN) ? transpose(data_slice) : data_slice

    hm = CairoMakie.heatmap!(ax, plot_x_axis, plot_y_axis, final_data; merge(heatmap_defaults, heatmap_kwargs)...)
    
    Colorbar(fig[1, 2], hm; colorbar_kwargs...)
    
    return fig
end
