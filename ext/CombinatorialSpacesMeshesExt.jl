module CombinatorialSpacesMeshesExt
using CombinatorialSpaces
using Meshes

import CombinatorialSpaces: EmbeddedDeltaSet2D

function EmbeddedDeltaSet2D(mesh::SimpleMesh)
  # This code supports capturing both 2D and 3D input points
  # @warn "EmbeddedDeltaSet2D mesh will use 3D points, setting z-coordinate to 0"

  function numerical_coords(p)
    c = getproperty.(getfield(p.coords, :coords), :val)
    dim = length(c)
    float_type = eltype(c)
    if dim == 3
      return Point3{float_type}(c)
    elseif dim == 2
      return Point3{float_type}(c..., 0)
    else
      @error "The mesh provided has points of unsupported dimension $dim"
    end
  end
  cs = map(p -> numerical_coords(p), Meshes.vertices(mesh))

  s = EmbeddedDeltaSet2D{Bool, typeof(first(cs))}()
  add_vertices!(s, length(cs), point=cs)
  for tri in mesh.topology.connec
    glue_sorted_triangle!(s, tri.indices...)
  end
  s[:edge_orientation] = false
  orient!(s)
  s
end

end
