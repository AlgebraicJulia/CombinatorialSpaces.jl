""" Mesh Tools
This file includes tools for importing delta sets from mesh
files supported by MeshIO and for converting delta sets to
meshes (for the purposes of plotting.

Meshes here are stored in the GeometryBasics.Mesh object.
"""
module MeshInterop
export EmbeddedDeltaSet2D, Mesh

using FileIO, MeshIO, GeometryBasics
import GeometryBasics: Mesh

using Catlab.CategoricalAlgebra: copy_parts!
using ..SimplicialSets, ..DualSimplicialSets
import ..SimplicialSets: EmbeddedDeltaSet2D

# Helper Functions (should these be exposed?)
##################

""" Construct a GeometryBasics.Mesh object from an embedded delta set
"""
function Mesh(ds::EmbeddedDeltaSet2D)
  points = Point{3, Float64}[point(ds)...]
  tris = TriangleFace{Int}[zip(triangle_vertices(ds)...)...]
  GeometryBasics.Mesh(points, tris)
end

""" Construct a GeometryBasics.Mesh object from a dual embedded delta set
"""
function Mesh(ds::EmbeddedDeltaDualComplex2D{O,R,P}; primal=false) where {O,R,P}
  if(primal)
    ds′ = EmbeddedDeltaSet2D{O,P}()
    copy_parts!(ds′, ds)
    Mesh(ds′)
  else
    points = Point{3, Float64}[dual_point(ds)...]
    tris = TriangleFace{Int}[zip(dual_triangle_vertices(ds)...)...]
    GeometryBasics.Mesh(points, tris)
  end
end

# Import Tooling
################

""" Constructor for EmbeddedDeltaSet2D from GeometryBasics.Mesh object

This operator should work for any triangular mesh object. Note that it will
not preserve any normal, texture, or other attributes from the Mesh object.
"""
function EmbeddedDeltaSet2D(m::GeometryBasics.Mesh; force_unique=false)
  coords = metafree.(coordinates(m))
  ind_map = collect(1:length(coords))
  if(force_unique) 
    indices = unique(i->coords[i], 1:length(coords))
    val2ind = Dict(coords[indices[i]]=>i for i in 1:length(indices))
    ind_map = map(c->val2ind[c], coords)
    coords = coords[indices]
  end
  tris = faces(m)
  s = EmbeddedDeltaSet2D{Bool, eltype(coords)}()
  add_vertices!(s, length(coords), point=coords)
  for tri in tris
    tri = ind_map[convert.(Int64, tri)]
    glue_sorted_triangle!(s, tri...)
  end
  # Properly orient the delta set
  orient!(s)
  s
end

""" Constructor for EmbeddedDeltaSet2D from mesh file

This operator should work for any file support for import from MeshIO. Note
that it will not preserve any normal, texture, or other data beyond points and
triangles from the mesh file.
"""
function EmbeddedDeltaSet2D(fn::String)
  if(fn[(end-2):end] == "stl")
    EmbeddedDeltaSet2D(FileIO.load(fn); force_unique=true)
  else
    EmbeddedDeltaSet2D(FileIO.load(fn))
  end
end
end
