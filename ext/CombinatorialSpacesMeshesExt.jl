module CombinatorialSpacesMeshesExt
using CombinatorialSpaces
using Meshes

import CombinatorialSpaces: EmbeddedDeltaSet2D
import Meshes: vertices

function EmbeddedDeltaSet2D(mesh::SimpleMesh)
  @warn "EmbeddedDeltaSet2D mesh will use 3D points, setting z-coordinate to 0"
  cs = map(p -> Point3{Float64}(p.coords.x.val, p.coords.y.val, 0.0), Meshes.vertices(mesh))

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
