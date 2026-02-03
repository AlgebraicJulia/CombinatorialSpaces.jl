module CombinatorialSpacesDelaunayTriangulationExt
using CombinatorialSpaces
using DelaunayTriangulation

import CombinatorialSpaces: EmbeddedDeltaSet2D
import DelaunayTriangulation: number_type

function EmbeddedDeltaSet2D(triangulation::T) where {T<:DelaunayTriangulation.Triangulation}
  z_coords = map(p -> p[3], triangulation.points)
  if any(z_coords .!= 0)
    error("This EmbeddedDeltaSet2D is designed for triangulations on the XY plane, but some z-coordinates of T are nonzero.")
  end
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
