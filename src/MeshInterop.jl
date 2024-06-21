""" Interoperation with mesh files.

This module enables delta sets to be imported from mesh files supported by
MeshIO.jl and for delta sets to be converted to meshes, mainly for the purposes
of plotting. Meshes are represented by the `GeometryBasics.Mesh` type.
"""
module MeshInterop

using FileIO, MeshIO, GeometryBasics

using Catlab.CategoricalAlgebra: copy_parts!
using ..SimplicialSets, ..DiscreteExteriorCalculus
import ..SimplicialSets: EmbeddedDeltaSet2D, EmbeddedDeltaSet3D

# Export meshes
###############

"""    function GeometryBasics.Mesh(sd::EmbeddedDeltaSet2D)

Construct a Mesh object from an EmbeddedDeltaSet2D.
"""
function GeometryBasics.Mesh(sd::EmbeddedDeltaSet2D)
  points = Point{3, Float64}[point(sd)...]
  tris = TriangleFace{Int}[zip(triangle_vertices(sd)...)...]
  Mesh(points, tris)
end

"""    function GeometryBasics.Mesh(ds::EmbeddedDeltaSet3D)

Construct a Mesh object from an EmbeddedDeltaSet3D.
"""
function GeometryBasics.Mesh(sd::EmbeddedDeltaSet3D)
  points = Point{3, Float64}[point(sd)...]
  tets = TetrahedronFace{Int}[zip(tetrahedron_vertices(sd)...)...]
  Mesh(points, tets)
end

"""    function GeometryBasics.Mesh(ds::EmbeddedDeltaDualComplex2D{O,R,P};
                             primal=false) where {O,R,P}

Construct a Mesh object from an EmbeddedDeltaDualComplex2D.

If `primal = true`, then only save primal simplices.
"""
function GeometryBasics.Mesh(sd::EmbeddedDeltaDualComplex2D{O,R,P};
                             primal=false) where {O,R,P}
  if primal
    sd′ = EmbeddedDeltaSet2D{O,P}()
    copy_parts!(sd′, sd)
    Mesh(sd′)
  else
    points = Point{3, Float64}[dual_point(sd)...]
    tris = TriangleFace{Int}[zip(dual_triangle_vertices(sd)...)...]
    Mesh(points, tris)
  end
end

"""    function GeometryBasics.Mesh(ds::EmbeddedDeltaDualComplex3D{O,R,P};
                             primal=false) where {O,R,P}

Construct a Mesh object from an EmbeddedDeltaDualComplex3D.

If `primal = true`, then only save primal simplices.
"""
function GeometryBasics.Mesh(sd::EmbeddedDeltaDualComplex3D{O,R,P};
                             primal=false) where {O,R,P}
  if primal
    sd′ = EmbeddedDeltaSet3D{O,P}()
    copy_parts!(sd′, sd)
    Mesh(sd′)
  else
    points = Point{3, Float64}[dual_point(sd)...]
    tets = TetrahedronFace{Int}[zip(dual_tetrahedron_vertices(sd)...)...]
    Mesh(points, tets)
  end
end

# Import meshes
###############

# Given a vector, return an endofunction that picks out the "unique" subset,
# and the indices of first occurences of the entries in that endofunction.
# The second return value is equivalent to:
# `unique(i -> ind_map[i], eachindex[ind_map])`.
function unique_index_map(coords::AbstractVector{T}, force_unique::Bool) where T
  uniques = 1:length(coords)
  ind_map = 1:length(coords)
  if force_unique
    s = Dict{T,Int}()
    i = 1
    uniques = Vector{Int}()
    ind_map = map(enumerate(coords)) do (j,c)
      if c in keys(s)
        s[c]
      else
        s[c] = i
        push!(uniques, j)
        i += 1
        s[c]
      end
    end
  end
  ind_map, uniques
end

"""    function EmbeddedDeltaSet2D(m::GeometryBasics.Mesh; force_unique=false)

Construct an EmbeddedDeltaSet2D from a Mesh object

This operator should work for any triangular mesh object. Note that it will
not preserve any normal, texture, or other attributes from the Mesh object.

The `force_unique` flag merges all points which are at the same location. This
is necessary to process `.stl` files which reference points by location.
"""
function EmbeddedDeltaSet2D(m::GeometryBasics.Mesh; force_unique=false)
  coords = metafree.(coordinates(m))
  ind_map, uniques = unique_index_map(coords, force_unique)
  tris = GeometryBasics.faces(m)
  s = EmbeddedDeltaSet2D{Bool, eltype(coords)}()
  add_vertices!(s, length(uniques), point=coords[uniques])
  for tri in tris
    tri = ind_map[convert.(Int64, tri)]
    glue_sorted_triangle!(s, tri...)
  end
  # Properly orient the delta set
  orient!(s)
  s
end

"""    function EmbeddedDeltaSet3D(m::GeometryBasics.Mesh; force_unique=false)

Construct an EmbeddedDeltaSet3D from a Mesh object

This operator should work for any tetrahedral mesh object. Note that it will
not preserve any normal, texture, or other attributes from the Mesh object.

The `force_unique` flag merges all points which are at the same location,
since some file formats reference points by location.
"""
function EmbeddedDeltaSet3D(m::GeometryBasics.Mesh; force_unique=false)
  coords = metafree.(coordinates(m))
  ind_map, uniques = unique_index_map(coords, force_unique)
  tets = GeometryBasics.faces(m)
  s = EmbeddedDeltaSet3D{Bool, eltype(coords)}()
  add_vertices!(s, length(uniques), point=coords[uniques])
  for tet in tets
    tet = ind_map[convert.(Int64, tet)]
    glue_sorted_tetrahedron!(s, tet...)
  end
  # Properly orient the delta set
  orient!(s)
  s
end

"""    function EmbeddedDeltaSet2D(fn::String)

Construct an EmbeddedDeltaSet2D from a mesh file.

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


#=
# XXX: No MeshIO file format supports tetrahedra.
# One thing to do is use TetGen.jl to load .ele files.
# Or load any mesh as usual, and use TetGen.jl to tesselate it.
# Note that TetGen.jl has examples of loading several file types and tesselating.
"""    function EmbeddedDeltaSet3D(fn::String)

Construct an EmbeddedDeltaSet3D from a mesh file.

This operator should work for any file support for import from MeshIO. Note
that it will not preserve any normal, texture, or other data beyond points and
tetrahedra from the mesh file.
"""
EmbeddedDeltaSet3D(fn::String) = EmbeddedDeltaSet3D(FileIO.load(fn))
=#

end
