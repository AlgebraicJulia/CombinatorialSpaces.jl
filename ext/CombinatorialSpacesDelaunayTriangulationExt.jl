module CombinatorialSpacesDelaunayTriangulationExt
using CombinatorialSpaces
using DelaunayTriangulation

import CombinatorialSpaces: EmbeddedDeltaSet2D
import DelaunayTriangulation: number_type

function EmbeddedDeltaSet2D(triangulation::T) where {T<:DelaunayTriangulation.Triangulation}
  @warn "EmbeddedDeltaSet2D mesh will use 3D points, setting z-coordinate to 0"
  float_type = number_type(triangulation.points)
  coords = map(p -> Point3{float_type}(p[1], p[2], 0), triangulation.points)

  s = EmbeddedDeltaSet2D{Bool, Point3{float_type}}()
  add_vertices!(s, length(coords), point=coords)
  for tri in each_solid_triangle(triangulation)
    glue_sorted_triangle!(s, tri...)
  end
  s[:edge_orientation] = false
  orient!(s)
  s
end

end
