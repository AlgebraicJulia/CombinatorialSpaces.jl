""" Visualization of delta sets and dual complexes.

This module providers wrapper functions to plotting embedded delta sets using
Makie.jl, including wireframe, mesh, and scatter plots. Note that one of the
Makie backends (CairoMakie, GLMakie, or WGLMakie) must be imported in order for
plotting to work.
"""
module CombinatorialSpacesMakieExt

# using Catlab.CategoricalAlgebra
# using CombinatorialSpaces.SimplicialSets
# using CombinatorialSpaces.MeshInterop

using Catlab
using CombinatorialSpaces

#import ...Makie: wireframe, wireframe!, mesh, mesh!, scatter, scatter!
#
#export wireframe, wireframe!, mesh, mesh!, scatter, scatter!

using GeometryBasics: Mesh
using Makie
import Makie: convert_arguments

""" This extends the "Mesh" plotting recipe for embedded deltasets by converting
an embedded deltaset into arguments that can be passed into Makie.mesh
"""

function convert_arguments(P::Union{Type{<:Makie.Wireframe},
                                    Type{<:Makie.Mesh},
                                    Type{<:Makie.Scatter}},
                           dset::HasDeltaSet)
  convert_arguments(P, Mesh(dset))
end

""" This extends the "LineSegments" plotting recipe for embedded deltasets by converting
an embedded deltaset into arguments that can be passed into Makie.linesegments
"""
function convert_arguments(P::Type{<:Makie.LineSegments}, dset::EmbeddedDeltaSet2D)
  edge_positions = zeros(ne(dset)*2,3)
  for e in edges(dset)
    edge_positions[2*e-1,:] = point(dset, src(dset, e))
    edge_positions[2*e,:] = point(dset, tgt(dset, e))
  end
  convert_arguments(P, edge_positions)
end

plottype(::EmbeddedDeltaSet2D) = Mesh

end
