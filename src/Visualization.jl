""" Visualization for Delta Sets and their duals.

This module provides wrapper functions for plotting embedded delta sets
using the Makie functionality. This includes the wireframe, mesh, and scatter
plotting functions. We also include a tool for importing delta sets from mesh
files supported by MeshIO.

It is important to note that the end-user will need to import one of the Makie
backends (CairoMakie, GLMakie, or WGLMakie) in order for plotting to work
correctly.
"""


module Visualization
import ..SimplicialSets: EmbeddedDeltaSet2D
using ..SimplicialSets, ..DualSimplicialSets
using Catlab.CategoricalAlgebra
using GeometryBasics
import AbstractPlotting: wireframe, wireframe!, mesh, mesh!, scatter, scatter!
import Base: convert
using FileIO, MeshIO
using StaticArrays

export DeltaSet2D, wireframe, wireframe!, mesh, mesh!, scatter, scatter!

# Helper Functions (should these be exposed?)
##################

""" Construct a GeometryBasics.Mesh object from an embedded delta set
"""
function make_mesh(ds::EmbeddedDeltaSet2D)
  points = Point{3, Float64}[point(ds)...]
  tris = TriangleFace{Int}[zip(triangle_vertices(ds)...)...]
  GeometryBasics.Mesh(points, tris)
end

""" Construct a GeometryBasics.Mesh object from a dual embedded delta set
"""
function make_mesh(ds::EmbeddedDeltaDualComplex2D)
  points = Point{3, Float64}[dual_point(ds)...]
  tris = TriangleFace{Int}[zip(dual_triangle_vertices(ds)...)...]
  GeometryBasics.Mesh(points, tris)
end

# Plotting Wrapper Functions
############################
""" Wrapper function for AbstractPlotting.wireframe
"""
function wireframe(ds::AbstractACSet; kw...)
   wireframe(make_mesh(ds); kw...)
end
""" Wrapper function for AbstractPlotting.wireframe!
"""
function wireframe!(ds::AbstractACSet; kw...)
   wireframe!(make_mesh(ds); kw...)
end

""" Wrapper function for AbstractPlotting.mesh!
"""
function mesh(ds::AbstractACSet; kw...)
   mesh(make_mesh(ds); kw...)
end
""" Wrapper function for AbstractPlotting.mesh!
"""
function mesh!(ds::AbstractACSet; kw...)
   mesh!(make_mesh(ds); kw...)
end

""" Wrapper function for AbstractPlotting.scatter!
"""
function scatter(ds::AbstractACSet; kw...)
  scatter(make_mesh(ds); kw...)
end
""" Wrapper function for AbstractPlotting.scatter!
"""
function scatter!(ds::AbstractACSet; kw...)
   scatter!(make_mesh(ds); kw...)
end

# Import Tooling
################

convert(Int64, a::OffsetInteger) = GeometryBasics.value(a)

""" Constructor for EmbeddedDeltaSet2D from GeometryBasics.Mesh object

This operator should work for any triangular mesh object. Note that it will
not preserve any normal, texture, or other attributes from the Mesh object.
"""
function EmbeddedDeltaSet2D(m::GeometryBasics.Mesh)
  coords = coordinates(m)
  tris = faces(m)
  s = EmbeddedDeltaSet2D{Bool, eltype(coords)}()
  add_vertices!(s, length(coords), point=coords)
  for tri in tris
    tri = convert.(Int64, tri)
    glue_sorted_triangle!(s, tri[1], tri[2], tri[3], tri_orientation = false)
  end
  s
end

""" Constructor for EmbeddedDeltaSet2D from mesh file

This operator should work for any file support for import from MeshIO. Note
that it will not preserve any normal, texture, or other data beyond points and
triangles from the mesh file.
"""
function EmbeddedDeltaSet2D(fn::String)
	EmbeddedDeltaSet2D(FileIO.load(fn))
end

end
