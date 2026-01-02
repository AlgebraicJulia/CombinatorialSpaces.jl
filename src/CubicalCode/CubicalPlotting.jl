# TODO: Fix plotting to work with multiple dimensions


function GeometryBasics.Mesh(s::EmbeddedCubicalComplex2D)
  ps = points(s)

  qs = QuadFace{Int}[]
  for y in 1:nyquads(s)
    for x in 1:nxquads(s)
      push!(qs, map(v -> coord_to_vert(s, v), quad_vertices(CartesianIndex(x, y))))
    end
  end
  
  GeometryBasics.Mesh(ps, qs)
end

convert_arguments(P::Union{Type{<:Makie.Wireframe}, Type{<:Makie.Mesh}, Type{<:Makie.Scatter}}, s::HasCubicalComplex) = convert_arguments(P, GeometryBasics.Mesh(s))

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
  x = map(a -> a[1], dps)
  y = map(a -> a[2], dps)

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
