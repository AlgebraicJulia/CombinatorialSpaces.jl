module CombinatorialSpacesMeshesExt
using CombinatorialSpaces
using Meshes

import CombinatorialSpaces: EmbeddedDeltaSet2D

function EmbeddedDeltaSet2D(mesh::SimpleMesh)
  # This code supports capturing both 2D and 3D input points
  function numerical_coords(p)
    c = getproperty.(getfield(p.coords, :coords), :val)
    dim = length(c)
    if dim == 3
      return Point3d(x)
    elseif dim == 2
      @warn "EmbeddedDeltaSet2D mesh will use 3D points, setting z-coordinate to 0"
      return Point3d(x..., 0)
    else
      @error "The mesh provided has points of unsupported dimension $dim"
    end
  end
  cs = map(p -> numerical_coords(p), Meshes.vertices(mesh))

  s = EmbeddedDeltaSet2D{Bool, Point3{Float64}}()
  add_vertices!(s, length(cs), point=cs)
  for tri in mesh.topology.connec
    length(tri.indices) == 3 || @error "Mesh contains non-triangular elements"
    glue_sorted_triangle!(s, tri.indices...)
  end
  s[:edge_orientation] = false
  orient!(s)
  s
end

end
