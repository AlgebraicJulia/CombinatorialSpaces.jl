""" Interoperation with mesh files.

This module enables delta sets to be imported from mesh files supported by
MeshIO.jl and for delta sets to be converted to meshes, mainly for the purposes
of plotting. Meshes are represented by the `GeometryBasics.Mesh` type.
"""
module MeshInterop

using FileIO, MeshIO, GeometryBasics

using Catlab.CategoricalAlgebra: copy_parts!
using ..SimplicialSets, ..DiscreteExteriorCalculus
import ..SimplicialSets: EmbeddedDeltaSet2D

# Export meshes
###############

""" Construct a Mesh object from an embedded delta set.
"""
function GeometryBasics.Mesh(ds::EmbeddedDeltaSet2D)
  points = Point{3, Float64}[point(ds)...]
  tris = TriangleFace{Int}[zip(triangle_vertices(ds)...)...]
  Mesh(points, tris)
end

""" Construct a Mesh object from a dual embedded delta set.
"""
function GeometryBasics.Mesh(ds::EmbeddedDeltaDualComplex2D{O,R,P};
                             primal=false) where {O,R,P}
  if primal
    ds′ = EmbeddedDeltaSet2D{O,P}()
    copy_parts!(ds′, ds)
    Mesh(ds′)
  else
    points = Point{3, Float64}[dual_point(ds)...]
    tris = TriangleFace{Int}[zip(dual_triangle_vertices(ds)...)...]
    Mesh(points, tris)
  end
end

# Import meshes
###############

""" Construct EmbeddedDeltaSet2D from Mesh object

This operator should work for any triangular mesh object. Note that it will
not preserve any normal, texture, or other attributes from the Mesh object.

The `force_unique` flag merges all points which are at the same location. This
is necessary to process `.stl` files which references points by location.
"""
function EmbeddedDeltaSet2D(m::GeometryBasics.Mesh; force_unique=false)
  coords = metafree.(coordinates(m))
  ind_map = 1:length(coords)
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

""" Construct EmbeddedDeltaSet2D from mesh file

This operator should work for any file support for import from MeshIO. Note
that it will not preserve any normal, texture, or other data beyond points and
triangles from the mesh file.
"""
function EmbeddedDeltaSet2D(fn::String)
  if(splitext(fn)[2] == ".stl")
    # The .stl format references points by location so we apply the
    # force_unique flag
    EmbeddedDeltaSet2D(FileIO.load(fn); force_unique=true)
  else
    EmbeddedDeltaSet2D(FileIO.load(fn))
  end
end

end
