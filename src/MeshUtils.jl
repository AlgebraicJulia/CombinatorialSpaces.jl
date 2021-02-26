""" Mesh utilities
This file includes tools for importing delta sets from mesh
files supported by MeshIO and for converting delta sets to
meshes (for the purposes of plotting.

Meshes here are stored in the GeometryBasics.Mesh object.
"""

module MeshUtils

using GeometryBasics
using FileIO, MeshIO
using ..SimplicialSets, ..DualSimplicialSets

import Base: convert
import ..SimplicialSets: EmbeddedDeltaSet2D

export make_mesh, EmbeddedDeltaSet2D

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
    glue_sorted_triangle!(s, tri[1], tri[2], tri[3])
  end
  # Assign orientation to 1, then spread to neighbors
  orient_component!(s, 1, true)

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
