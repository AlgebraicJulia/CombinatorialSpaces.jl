module Interop
export mesh_to_deltaset

using ..SimplicialSets
using Catlab.Graphs
using Requires

function __init__()
  @require GeometryBasics="5c1252a2-5f33-56bf-86c9-59e7332b4326" begin
    using .GeometryBasics

    """ Turn a Mesh from GeometryBasics into an embedded deltaset
    """
    function mesh_to_deltaset(mesh::Mesh{3,T}) where {T}
      n = length(coordinates(mesh))
      dset = EmbeddedDeltaSet2D{Tuple{},Point{3,T}}()
      add_vertices!(dset,n,point=coordinates(mesh))
      for tri in faces(mesh)
        vertices = map(e -> Int32(e.i)+1, [tri...])
        glue_sorted_triangle!(dset, vertices..., tri_orientation=())
      end
      dset
    end
  end

  @require AbstractPlotting="537997a7-5e4e-5d89-9595-2241ea00577e" begin
    import .AbstractPlotting
    import .AbstractPlotting: convert_arguments

    """ This extends the "Mesh" plotting recipe for embedded deltasets by converting
    an embedded deltaset into arguments that can be passed into AbstractPlotting.mesh
    """
    function convert_arguments(P::Type{<:AbstractPlotting.Mesh}, dset::EmbeddedDeltaSet2D)
      convert_arguments(P, point_array(dset), hcat(triangle_vertices(dset,:)...))
    end

    """ This extends the "LineSegments" plotting recipe for embedded deltasets by converting
    an embedded deltaset into arguments that can be passed into AbstractPlotting.linesegments
    """
    function convert_arguments(P::Type{<:AbstractPlotting.LineSegments}, dset::EmbeddedDeltaSet2D)
      edge_positions = zeros(ne(dset)*2,3)
      for e in edges(dset)
        edge_positions[2*e-1,:] = point(dset, dset[e,:src])
        edge_positions[2*e,:] = point(dset, dset[e,:tgt])
      end
      convert_arguments(P, edge_positions)
    end

    plottype(::EmbeddedDeltaSet2D) = Mesh
  end
end

end
