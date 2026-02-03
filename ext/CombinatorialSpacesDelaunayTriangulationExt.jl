module CombinatorialSpacesDelaunayTriangulationExt
using CombinatorialSpaces
using DelaunayTriangulation

import CombinatorialSpaces: EmbeddedDeltaSet2D
import DelaunayTriangulation: number_type

function EmbeddedDeltaSet2D(triangulation::T) where {T<:DelaunayTriangulation.Triangulation}
  # Check if points have z-coordinates and if they're non-zero
  # DelaunayTriangulation typically uses 2D points, so we need to check the dimension
  if !isempty(triangulation.points)
    first_point = first(triangulation.points)
    if length(first_point) >= 3
      # If points have 3+ dimensions, check if all z-coordinates are zero
      # Assuming all points in triangulation have uniform dimensionality
      for p in triangulation.points
        if p[3] != 0
          error("This EmbeddedDeltaSet2D is designed for triangulations on the XY plane, but some z-coordinates of T are nonzero.")
        end
      end
    end
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
