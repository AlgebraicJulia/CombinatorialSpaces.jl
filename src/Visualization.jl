module Visualization

import ..SimplicialSets: DeltaSet2D
using ..SimplicialSets, ..DualSimplicialSets
using Catlab.CategoricalAlgebra
using GeometryBasics
using AbstractPlotting
using FileIO, MeshIO
using StaticArrays

export DeltaSet2D, plot_deltaset

function dual_triangle_vertices(s::AbstractDeltaDualComplex2D,t...)
  SVector(s[s[t..., :D_∂e1], :D_∂v0], s[s[t..., :D_∂e2], :D_∂v1], s[s[t..., :D_∂e2], :D_∂v0])
end

function DeltaSet2D(m::GeometryBasics.Mesh)
  coords = coordinates(m)
  tri_inds = map(x->(getfield.(x, :i) .+ 1), faces(m))
  s = EmbeddedDeltaSet2D{Bool, eltype(coords)}()
  add_vertices!(s, length(coords), point=coords)
  for ind in tri_inds
    glue_sorted_triangle!(s, ind[1], ind[2], ind[3], tri_orientation = false)
  end
  s
end

function make_mesh(ds::EmbeddedDeltaSet2D)
  points = Point{3, Float64}[point(ds)...]
  tris = TriangleFace{Int}[zip(triangle_vertices(ds)...)...]
  GeometryBasics.Mesh(points, tris)
end
function make_mesh(ds::EmbeddedDeltaDualComplex2D)
  points = Point{3, Float64}[dual_point(ds)...]
  tris = TriangleFace{Int}[zip(dual_triangle_vertices(ds)...)...]
  GeometryBasics.Mesh(points, tris)
end
function plot_deltaset(ds; kw...)
   wireframe(make_mesh(ds); kw...)
end
function plot_deltaset!(ds; kw...)
   wireframe!(make_mesh(ds); kw...)
end

function DeltaSet2D(fn::String)
	DeltaSet2D(FileIO.load(fn))
end

end
