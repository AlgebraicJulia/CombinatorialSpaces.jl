module Meshes

using Artifacts
using Catlab, Catlab.CategoricalAlgebra
using CombinatorialSpaces, CombinatorialSpaces.SimplicialSets
using FileIO
using JSON
using GeometryBasics: Point2, Point3
using LinearAlgebra: diagm

export loadmesh, Icosphere, Rectangle_30x10, Torus_30x10, Point_Map, triangulated_grid, makeSphere

Point2D = Point2{Float64}
Point3D = Point3{Float64}

abstract type AbstractMeshKey end

struct Icosphere{N, R} <: AbstractMeshKey
  n::N
  r::R
end
struct Rectangle_30x10 <: AbstractMeshKey end
struct Torus_30x10 <: AbstractMeshKey end
struct Point_Map <: AbstractMeshKey end

Icosphere(n) = Icosphere(n, 1.0)

"""    loadmesh(s::Icosphere)

Load in a icosphere mesh.
"""
function loadmesh(s::Icosphere)
  1 <= s.n <= 8 || error("The only icosphere divisions supported are 1-8")
  m = loadmesh_icosphere_helper("UnitIcosphere$(s.n).obj")
  m[:point] = m[:point]*s.r
  return m
end

"""    loadmesh(s::Rectangle_30x10)

Load in a rectangular mesh.
"""
function loadmesh(s::Rectangle_30x10)
  parse_json_acset(EmbeddedDeltaSet2D{Bool, Point3{Float64}},
    read(joinpath(artifact"Rectangle_30x10", "Rectangle_30x10.json"), String))
end

"""    loadmesh(s::Torus_30x10)

Load in a toroidal mesh.
"""
function loadmesh(s::Torus_30x10)
  parse_json_acset(EmbeddedDeltaDualComplex2D{Bool, Float64, Point3{Float64}},
    read(joinpath(artifact"Torus_30x10", "Torus_30x10.json"), String))
end

"""    loadmesh(s::Point_Map)

Load in a point map describing the connectivity of the toroidal mesh.
"""
function loadmesh(s::Point_Map)
  JSON.parse(read(joinpath(artifact"point_map", "point_map.json"), String))
end

#loadmesh(meshkey::AbstractMeshKey)::EmbeddedDeltaSet2D

loadmesh_icosphere_helper(obj_file_name) = EmbeddedDeltaSet2D(
  joinpath(artifact"icospheres1", obj_file_name))


# This function was once the gridmeshes.jl file from Decapodes.jl.
"""    function triangulated_grid(max_x, max_y, dx, dy, point_type, compress=false)

Triangulate the rectangle [0,max_x] x [0,max_y] by approximately equilateral
triangles of width `dx` and height `dy`.

If `compress` is true (default), then enforce that all rows of points are less than `max_x`,
otherwise, keep `dx` as is.
"""
function triangulated_grid(max_x, max_y, dx, dy, point_type, compress=true)
  s = EmbeddedDeltaSet2D{Bool, point_type}()

  # Place equally-spaced points in a max_x by max_y rectangle.
  coords = point_type == Point3D ? map(x -> point_type(x..., 0), Iterators.product(0:dx:max_x, 0:dy:max_y)) : map(x -> point_type(x...), Iterators.product(0:dx:max_x, 0:dy:max_y))
  # Perturb every other row right by half a dx.
  coords[:, 2:2:end] = map(coords[:, 2:2:end]) do row
    if point_type == Point3D
      row .+ [dx/2, 0, 0]
    else
      row .+ [dx/2, 0]
    end
  end

  if compress
    # The perturbation moved the right-most points past max_x, so compress along x.
    map!(coords, coords) do coord
      if point_type == Point3D
        diagm([max_x/(max_x+dx/2), 1, 1]) * coord
      else
        diagm([max_x/(max_x+dx/2), 1]) * coord
      end
    end
  end
 
  add_vertices!(s, length(coords), point = vec(coords))

  nx = length(0:dx:max_x)

  # Matrix that stores indices of points.
  idcs = reshape(eachindex(coords), size(coords))
  # Only grab vertices that will be the bottom-left corner of a subdivided square.
  idcs = idcs[begin:end-1, begin:end-1]
  
  # Subdivide every other row along the opposite diagonal.
  for i in idcs[:, begin+1:2:end]
    glue_sorted_triangle!(s, i, i+nx, i+nx+1)
    glue_sorted_triangle!(s, i, i+1, i+nx+1)
  end
  for i in idcs[:, begin:2:end]
    glue_sorted_triangle!(s, i, i+1, i+nx)
    glue_sorted_triangle!(s, i+1, i+nx, i+nx+1)
  end

  # Orient and return.
  s[:edge_orientation]=true
  orient!(s)
  s
end

# This function was once the sphericalmeshes.jl file from Decapodes.jl.
"""
    makeSphere(minLat, maxLat, dLat, minLong, maxLong, dLong, radius)

Construct a spherical mesh (inclusively) bounded by the given latitudes and
longitudes, discretized at dLat and dLong intervals, at the given radius from
Earth's center.

Note that this construction returns a UV-sphere. DEC simulations are more
accurate on meshes with (near) equilateral triangles, such as the icospheres
available through [`loadmesh`](@ref).

We say that:
- 90°N is 0
- 90°S is 180
- Prime Meridian is 0
- 10°W is 355

We say that:
- (x=0,y=0,z=0) is at the center of the sphere
- the x-axis points toward 0°,0°
- the y-axis points toward 90°E,0°
- the z-axis points toward the North Pole

# References:
[List of common coordinate transformations](https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations?oldformat=true#From_spherical_coordinates)

# Examples
```julia-repl
# Regular octahedron.
julia> s, npi, spi = makeSphere(0, 180, 90, 0, 360, 90, 1)
```
```julia-repl
# 72 points along the unit circle on the x-y plane.
julia> s, npi, spi = makeSphere(90, 90, 0, 0, 360, 5, 1)
```
```julia-repl
# 72 points along the equator at 0km from Earth's surface.
julia> s, npi, spi = makeSphere(90, 90, 1, 0, 360, 5, 6371)
```
```julia-repl
# TIE-GCM grid at 90km altitude (with no poles,   i.e. a bulbous cylinder).
julia> s, npi, spi = makeSphere(5, 175, 5, 0, 360, 5, 6371+90)
```
```julia-repl
# TIE-GCM grid at 90km altitude (with South pole, i.e. a bowl).
julia> s, npi, spi = makeSphere(5, 180, 5, 0, 360, 5, 6371+90)
```
```julia-repl
# TIE-GCM grid at 90km altitude (with poles,      i.e. a sphere).
julia> s, npi, spi = makeSphere(0, 180, 5, 0, 360, 5, 6371+90)
```
```julia-repl
# The Northern hemisphere of the TIE-GCM grid at 90km altitude.
julia> s, npi, spi = makeSphere(0, 180, 5, 0, 360, 5, 6371+90)
```
"""
function makeSphere(minLat, maxLat, dLat, minLong, maxLong, dLong, radius)
  if (   !(0 ≤ minLat ≤ 180)  || !(0 ≤ maxLat ≤ 180)
      || !(0 ≤ minLong ≤ 360) || !(0 ≤ minLong ≤ 360)
      ||  (maxLat < minLat)   ||  (maxLong < minLong))
    throw(ArgumentError("Mins must be less than Maxs, lats must be in [0,180],"*
                        " and longs must be in [0,360]."))
  end
  sph2car(ρ,ϕ,θ) = (ρ*sind(θ)*cosd(ϕ),
                    ρ*sind(θ)*sind(ϕ),
                    ρ*cosd(θ))
  if (minLat == maxLat && dLat == 0)
    dLat = 1 # User wants a just one parallel. a:0:a is not valid julia.
  end
  s = EmbeddedDeltaSet2D{Bool, Point3D}()
  # Neither pole counts as a Meridian.
  connect_north_pole = false
  connect_south_pole = false
  if (minLat == 0)
    # Don't create num_meridians points at the North pole.
    minLat += dLat
    connect_north_pole = true
  end
  if (maxLat == 180)
    # Don't create num_meridians points at the South pole.
    maxLat -= dLat
    connect_south_pole = true
  end
  connect_long = false
  if (maxLong == 360)
    maxLong -= dLong
    connect_long = true
  end
  # TODO: Should we warn the user if the stitching edges are shorter than the
  # rest?
  num_parallels = length(minLat:dLat:maxLat)
  num_meridians = length(minLong:dLong:maxLong)
  ρ = radius
  # Add points one parallel at a time.
  for θ in minLat:dLat:maxLat
    vertex_offset = nv(s)+1
    add_vertices!(s, num_meridians,
                  point=map(minLong:dLong:maxLong) do ϕ
                    Point3D(sph2car(ρ,ϕ,θ)...)
                  end)
    # Connect this parallel.
    if (connect_long)
      add_sorted_edge!(s, vertex_offset+num_meridians-1, vertex_offset)
    end
    # Don't make triangles with the previous parallel if there isn't one.
    if (vertex_offset == 1)
      add_sorted_edges!(s,
                        vertex_offset:vertex_offset+num_meridians-2,
                        vertex_offset+1:vertex_offset+num_meridians-1)
      continue
    end
    # Add the triangles.
    foreach(vertex_offset  :vertex_offset+num_meridians-2,
            vertex_offset+1:vertex_offset+num_meridians-1,
            vertex_offset-num_meridians:vertex_offset-2) do i,j,k
      glue_sorted_triangle!(s, i, j, k)
    end
    foreach(vertex_offset+1:vertex_offset+num_meridians-1,
            vertex_offset-num_meridians:vertex_offset-2,
            vertex_offset-num_meridians+1:vertex_offset-1) do i,j,k
      glue_sorted_triangle!(s, i, j, k)
    end
    # Connect with the previous parallel.
    if (connect_long)
      glue_sorted_triangle!(s, vertex_offset+num_meridians-1,
                            vertex_offset, vertex_offset-1)
      glue_sorted_triangle!(s, vertex_offset-num_meridians,
                            vertex_offset, vertex_offset-1)
    end
  end
  # Add the North and South poles.
  north_pole_idx = 0
  if (connect_north_pole)
    ϕ, θ = 0, 0
    add_vertex!(s, point=Point3D(sph2car(ρ,ϕ,θ)...))
    north_pole_idx = nv(s)
    foreach(1:num_meridians-1, 2:num_meridians) do i,j
      glue_sorted_triangle!(s, north_pole_idx, i, j)
    end
    if (connect_long)
      glue_sorted_triangle!(s, north_pole_idx, 1, num_meridians)
    end
  end
  south_pole_idx = 0
  if (connect_south_pole)
    south_parallel_start = num_meridians*(num_parallels-1)+1
    ϕ, θ = 0, 180
    add_vertex!(s, point=Point3D(sph2car(ρ,ϕ,θ)...))
    south_pole_idx = nv(s)
    foreach(south_parallel_start:south_parallel_start+num_meridians-2,
            south_parallel_start+1:south_parallel_start+num_meridians-1) do i,j
      glue_sorted_triangle!(s, south_pole_idx, i, j)
    end
    if (connect_long)
      glue_sorted_triangle!(s, south_pole_idx, south_parallel_start,
                            south_parallel_start+num_meridians-1)
    end
  end
  return s, north_pole_idx, south_pole_idx
end

"""    function tri_345()

Return the primal and dual mesh of a triangle with edge lengths 3,4,5 and with true orientation.
See also: [`tri_345_false`](@ref)
"""
function tri_345()
  primal_s = EmbeddedDeltaSet2D{Bool,Point3D}()
  add_vertices!(primal_s, 3, point=[Point3D(0,0,0), Point3D(3,0,0), Point3D(3,4,0)])
  glue_triangle!(primal_s, 1, 2, 3, tri_orientation=true)
  primal_s[:edge_orientation] = true
  s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(primal_s)
  subdivide_duals!(s, Barycenter())
  (primal_s, s)
end

"""    function tri_345_false()

Return the primal and dual mesh of a triangle with edge lengths 3,4,5 and with false orientation.
See also: [`tri_345`](@ref)
"""
function tri_345_false()
  primal_s = EmbeddedDeltaSet2D{Bool,Point3D}()
  add_vertices!(primal_s, 3, point=[Point3D(0,0,0), Point3D(3,0,0), Point3D(3,4,0)])
  glue_triangle!(primal_s, 1, 2, 3, tri_orientation=false)
  primal_s[:edge_orientation] = true
  s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(primal_s)
  subdivide_duals!(s, Barycenter())
  (primal_s, s)
end

"""    function grid_345()

Return the primal and dual mesh of a grid of 3-4-5 triangles.
See also: [`tri_345`](@ref)
"""
function grid_345()
  primal_s = EmbeddedDeltaSet2D{Bool,Point3D}()
  add_vertices!(primal_s, 9,
    point=[Point3D(0,+4,0), Point3D(3,+4,0), Point3D(6,+4,0),
          Point3D(0, 0,0), Point3D(3, 0,0), Point3D(6, 0,0),
          Point3D(0,-4,0), Point3D(3,-4,0), Point3D(6,-4,0)])
  glue_sorted_triangle!(primal_s, 1, 2, 4)
  glue_sorted_triangle!(primal_s, 5, 2, 4)
  glue_sorted_triangle!(primal_s, 5, 2, 3)
  glue_sorted_triangle!(primal_s, 5, 6, 3)
  glue_sorted_triangle!(primal_s, 5, 7, 4)
  glue_sorted_triangle!(primal_s, 5, 7, 8)
  glue_sorted_triangle!(primal_s, 5, 6, 8)
  glue_sorted_triangle!(primal_s, 9, 6, 8)
  primal_s[:edge_orientation] = true
  s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(primal_s)
  subdivide_duals!(s, Barycenter())
  (primal_s, s)
end

"""    function right_scalene_unit_hypot()

Return the primal and dual mesh of a right scalene triangle with unit hypotenuse.
"""
function right_scalene_unit_hypot()
  primal_s = EmbeddedDeltaSet2D{Bool,Point2D}()
  add_vertices!(primal_s, 3,
    point=[Point2D(0,0), Point2D(1/√2,0), Point2D(1/√2,1/√2)])
  glue_sorted_triangle!(primal_s, 1, 2, 3)
  primal_s[:edge_orientation] = true
  s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(primal_s)
  subdivide_duals!(s, Barycenter())
  (primal_s, s)
end

"""    function single_tetrahedron()

Return the primal and dual mesh of a single tetrahedron with points at the origin and Euclidean basis vectors.
"""
function single_tetrahedron()
  pnts = Point3D[
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
  primal_s = EmbeddedDeltaSet3D{Bool, Point3D}()
  add_vertices!(primal_s, 4, point=pnts)
  glue_sorted_tetrahedron!(primal_s, 1:4...)
  s = EmbeddedDeltaDualComplex3D{Bool, Float64, Point3D}(primal_s)
  subdivide_duals!(s, Barycenter())
  (primal_s, s)
end

end
