# TODO: Fix plotting to work with multiple dimensions

function GeometryBasics.Mesh(s::HasCubicalComplex)
  ps = points(s)
  qs = map(q -> QuadFace{Int}(quad_vertices(s, q)), quadrilaterals(s))
  GeometryBasics.Mesh(ps, qs)
end

function convert_arguments(P::Union{Type{<:Makie.Wireframe},
                                    Type{<:Makie.Mesh},
                                    Type{<:Makie.Scatter}},
                           s::HasCubicalComplex)
  convert_arguments(P, GeometryBasics.Mesh(s))
end

plottype(::HasCubicalComplex) = GeometryBasics.Mesh

# TODO: Add in generic kwargs
function plot_zeroform(s::HasCubicalComplex, f)
  fig = Figure();
  ax = CairoMakie.Axis(fig[1, 1])
  msh = CairoMakie.mesh!(ax, s, color=ω, colormap=:jet)
  Colorbar(fig[1, 2], msh)
  fig
end

function plot_oneform(s::HasCubicalComplex, alpha)
  dps = dual_points(s)
  x = map(a -> dps[a][1], quadrilaterals(s))
  y = map(a -> dps[a][2], quadrilaterals(s))

  X,Y = sharp_pd(s, alpha)

  color = sqrt.(X.^2 + Y.^2)

  fig = Figure();
  ax = CairoMakie.Axis(fig[1,1])
  arrows2d!(ax, x, y, X, Y, color = color)
  wireframe!(ax, s)
  fig
end

function create_gif(solution, file_name)
  frames = length(solution)
  fig = Figure()
  ax = CairoMakie.Axis(fig[1,1])
  msh = CairoMakie.mesh!(ax, s, color=first(solution), colormap=:jet, colorrange=extrema(first(solution)))
  Colorbar(fig[1,2], msh)
  CairoMakie.record(fig, file_name, 1:10:frames; framerate = 15) do t
    msh.color = solution[t]
  end
end
