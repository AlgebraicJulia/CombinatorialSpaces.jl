# Points will be given implicitly using the user defined size of domain and number of points in each direction.
# Edges will be given a coordinate based on their source point and their alignment (x or y).
# Quads will be given a coordinate based on their lower left point.

# Default edge orientation will be from smaller to greater vertex index, so x-aligned edges will be oriented in the positive x direction and y-aligned edges will be oriented in the positive y direction.
# Default quad orientation will be counterclockwise starting from the lower left vertex.

using GeometryBasics

import Base: show

abstract type AbstractCubicalComplex end

abstract type AbstractCubicalComplex2D{FT <: AbstractFloat} <: AbstractCubicalComplex end

struct UniformCubicalComplex2D{FT} <: AbstractCubicalComplex2D{FT}
  nx::Int
  ny::Int

  dx::FT
  dy::FT

  halo_x::Int
  halo_y::Int

  base_x::FT
  base_y::FT
end

@enum Align X_ALIGN Y_ALIGN Z_ALIGN

@enum GridSide EASTWEST NORTHSOUTH UPDOWN ALL

# Base point of real mesh will be (halo_x + 1, halo_y + 1)
# End point of real mesh will be (halo_x + nxr, halo_y + nyr)

base_x(s::UniformCubicalComplex2D) = s.base_x
base_y(s::UniformCubicalComplex2D) = s.base_y

# This grabs the number of real points in the x and y directions, excluding halo points
nxr(s::AbstractCubicalComplex2D) = s.nx
nyr(s::AbstractCubicalComplex2D) = s.ny

hx(s::AbstractCubicalComplex2D) = s.halo_x
hy(s::AbstractCubicalComplex2D) = s.halo_y

nx(s::AbstractCubicalComplex2D) = nxr(s) + 2 * hx(s)
ny(s::AbstractCubicalComplex2D) = nyr(s) + 2 * hy(s)

dx(s::UniformCubicalComplex2D) = s.dx
dy(s::UniformCubicalComplex2D) = s.dy

nv(s::AbstractCubicalComplex2D) = nx(s) * ny(s)
nvr(s::AbstractCubicalComplex2D) = nxr(s) * nyr(s)

nxe(s::AbstractCubicalComplex2D) = nx(s) - 1
nye(s::AbstractCubicalComplex2D) = ny(s) - 1

nxe_r(s::AbstractCubicalComplex2D) = nxe(s) - 2 * hx(s)
nye_r(s::AbstractCubicalComplex2D) = nye(s) - 2 * hy(s)

nxedges(s::AbstractCubicalComplex2D) = nxe(s) * ny(s)
nyedges(s::AbstractCubicalComplex2D) = nx(s) * nye(s)

ne(s::AbstractCubicalComplex2D) = nxedges(s) + nyedges(s)

nxq(s::AbstractCubicalComplex2D) = nx(s) - 1
nyq(s::AbstractCubicalComplex2D) = ny(s) - 1
nquads(s::AbstractCubicalComplex2D) = nxq(s) * nyq(s)

hxq(s::AbstractCubicalComplex2D) = hx(s)
hyq(s::AbstractCubicalComplex2D) = hy(s)

nxqr(s::AbstractCubicalComplex2D) = nxq(s) - 2 * hxq(s)
nyqr(s::AbstractCubicalComplex2D) = nyq(s) - 2 * hyq(s)
nquadsr(s::AbstractCubicalComplex2D) = nxqr(s) * nyqr(s)

vertices(s::AbstractCubicalComplex2D) = 1:nv(s)
edges(s::AbstractCubicalComplex2D) = 1:ne(s)
quads(s::AbstractCubicalComplex2D) = 1:nquads(s)

top_edges(s::AbstractCubicalComplex2D) = coord_to_edge.(Ref(s), 1:nxe(s), Ref(ny(s)), Ref(X_ALIGN))
bottom_edges(s::AbstractCubicalComplex2D) = coord_to_edge.(Ref(s), 1:nxe(s), Ref(1), Ref(X_ALIGN))
left_edges(s::AbstractCubicalComplex2D) = coord_to_edge.(Ref(s), Ref(1), 1:nye(s), Ref(Y_ALIGN))
right_edges(s::AbstractCubicalComplex2D) = coord_to_edge.(Ref(s), Ref(nx(s)), 1:nye(s), Ref(Y_ALIGN))

boundary_edges(s::AbstractCubicalComplex2D) = vcat(bottom_edges(s), top_edges(s), left_edges(s), right_edges(s))

coord_to_vert(s::AbstractCubicalComplex2D, x::Int, y::Int) = x + (y - 1) * nx(s)
function coord_to_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align)
  if align == X_ALIGN
    return x + (y - 1) * nxe(s)
  elseif align == Y_ALIGN
    return x + (y - 1) * nx(s) + nxedges(s)
  else # align == Z_ALIGN
    return 0
  end
end
coord_to_quad(s::AbstractCubicalComplex2D, x::Int, y::Int) = (y - 1) * nxq(s) + x

is_X_aligned(e::Int, s::AbstractCubicalComplex2D) = e <= nxedges(s)
is_Y_aligned(e::Int, s::AbstractCubicalComplex2D) = e > nxedges(s)

real_coord_to_vert(s::AbstractCubicalComplex2D, x::Int, y::Int) = coord_to_vert(s, x + hx(s), y + hy(s))
# This function takes a coordinate on the interior and maps it to a vertex index
real_coord_to_real_vert(s::AbstractCubicalComplex2D, x::Int, y::Int) = x + (y - 1) * nxr(s)
coord_to_real_coord(s::AbstractCubicalComplex2D, x::Int, y::Int) = (x - hx(s), y - hy(s))
function vert_to_real_vert(s::AbstractCubicalComplex2D, v::Int)
  x, y = vert_to_coord(s, v)
  x, y = coord_to_real_coord(s, x, y)
  real_coord_to_real_vert(s, x, y)
end

is_halo_vert(s::AbstractCubicalComplex2D, x::Int, y::Int) = x <= hx(s) || x > nxr(s) + hx(s) || y <= hy(s) || y > nyr(s) + hy(s)
is_halo_quad(s::AbstractCubicalComplex2D, x::Int, y::Int) = x <= hx(s) || x > nxr(s) + hx(s) - 1 || y <= hy(s) || y > nyr(s) + hy(s) - 1

function vert_to_coord(s::AbstractCubicalComplex2D, v::Int)
  y = div(v - 1, nx(s)) + 1
  x = v - (y - 1) * nx(s)
  return x, y
end

function edge_to_coord(s::AbstractCubicalComplex2D, e::Int)
  if e <= nxe(s) * ny(s)
    # x-aligned edge
    y = div(e - 1, nxe(s)) + 1
    x = e - (y - 1) * nxe(s)
    return x, y, X_ALIGN
  else
    # y-aligned edge
    e_adj = e - nxe(s) * ny(s)
    y = div(e_adj - 1, nx(s)) + 1
    x = e_adj - (y - 1) * nx(s)
    return x, y, Y_ALIGN
  end
end

function dual_edge_to_coord(s::AbstractCubicalComplex2D, e::Int)
  x, y, align = edge_to_coord(s, e)
  if align == X_ALIGN
    return x, y, Y_ALIGN
  else # align == Y_ALIGN
    return x, y, X_ALIGN
  end
end

function quad_to_coord(s::AbstractCubicalComplex2D, q::Int)
  y = div(q - 1, nxq(s)) + 1
  x = q - (y - 1) * nxq(s)
  return x, y
end

function point(s::UniformCubicalComplex2D, x::Int, y::Int)
  px = base_x(s) + (x - 1) * dx(s) - hx(s) * dx(s)
  py = base_y(s) + (y - 1) * dy(s) - hy(s) * dy(s)
  return Point3(px, py, 0.0)
end
point(s::UniformCubicalComplex2D, v::Int) = point(s, vert_to_coord(s, v)...)

# Returns the point corresponding to the given real coordinate, where the real coordinate (1, 1) corresponds to the first real point (halo_x + 1, halo_y + 1)
real_point(s::AbstractCubicalComplex2D, x::Int, y::Int) = point(s, x + hx(s), y + hy(s))

# TODO: Make this a generator that yields points one at a time instead of creating an array of all points at once
function points(s::AbstractCubicalComplex2D)
  return map(v -> point(s, v), vertices(s))
end

# Function to get spacing given a length and number of points
spacing(len::FT, np::Int) where FT <: AbstractFloat = return len / (np - 1)

# The interval given (lx, ly) is the size of the real domain, excluding halo points. So the total size of the mesh will be (lx + 2 * halo_x * dx, ly + 2 * halo_y * dy)
function UniformCubicalComplex2D(nxr::Int, nyr::Int, lx::FT, ly::FT;
                                  halo_x::Int = 0, halo_y::Int = 0,
                                  base_x::FT  = zero(FT),
                                  base_y::FT  = zero(FT)) where FT <: AbstractFloat
    dx = spacing(lx, nxr)
    dy = spacing(ly, nyr)
    UniformCubicalComplex2D{FT}(nxr, nyr, dx, dy, halo_x, halo_y, base_x, base_y)
end
# Basic show method for uniform mesh
function Base.show(io::IO, s::UniformCubicalComplex2D)
  println(io, "UniformCubicalComplex2D with dimensions: $(nx(s)) x $(ny(s))")
  println(io, "Spacing: dx = $(dx(s)), dy = $(dy(s))")
  println(io, "Halo: halo_x = $(s.halo_x), halo_y = $(s.halo_y)")
  println(io, "Base point: ($(base_x(s)), $(base_y(s)))")
end

# Get the index of the source point of an edge
src(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align) = coord_to_vert(s, x, y)
src(s::AbstractCubicalComplex2D, e::Int) = src(s, edge_to_coord(s, e)...)

# Get the index of the target of an edge
function tgt(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align)
  if align == X_ALIGN
    return coord_to_vert(s, x + 1, y)
  elseif align == Y_ALIGN
    return coord_to_vert(s, x, y + 1)
  else # align == Z_ALIGN
    return 0
  end
end
tgt(s::AbstractCubicalComplex, e::Int) = tgt(s, edge_to_coord(s, e)...)

function edge_len(s::UniformCubicalComplex2D{FT}, align::Align) where FT <: AbstractFloat
  if align == X_ALIGN
    return dx(s)
  elseif align == Y_ALIGN
    return dy(s)
  else # align == Z_ALIGN
    return zero(FT)
  end
end

edge_len(s::UniformCubicalComplex2D, x::Int, y::Int, align::Align) = edge_len(s, align)

edge_len(s::AbstractCubicalComplex2D, e::Int) = edge_len(s, edge_to_coord(s, e)...)

xedges(s::AbstractCubicalComplex2D, arr::AbstractVector) = arr[1:nxedges(s)]
yedges(s::AbstractCubicalComplex2D, arr::AbstractVector) = arr[nxedges(s)+1:end]

# Get the index of the vertices of a quad
# The vertices are ordered counterclockwise starting from the lower left vertex
function quad_vertices(s::AbstractCubicalComplex2D, x::Int, y::Int)
  v1 = coord_to_vert(s, x, y)
  v2 = coord_to_vert(s, x + 1, y)
  v3 = coord_to_vert(s, x + 1, y + 1)
  v4 = coord_to_vert(s, x, y + 1)
  return (v1, v2, v3, v4)
end

# The edges of a quad are ordered counterclockwise starting from the bottom edge
function quad_edges(s::AbstractCubicalComplex2D, x::Int, y::Int)
  e1 = coord_to_edge(s, x, y, X_ALIGN)
  e2 = coord_to_edge(s, x + 1, y, Y_ALIGN)
  e3 = coord_to_edge(s, x, y + 1, X_ALIGN)
  e4 = coord_to_edge(s, x, y, Y_ALIGN)
  return (e1, e2, e3, e4)
end

# Given a quad, gives the edge offset by the given amount in the given direction
# An offset of zero will give either the left or bottom edge, depending on the direction
function quad_edge_offset(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align, offset::Int)
  if align == X_ALIGN
    return coord_to_edge(s, x, y + offset, X_ALIGN)
  elseif align == Y_ALIGN
    return coord_to_edge(s, x + offset, y, Y_ALIGN)
  else # align == Z_ALIGN
    return 0
  end
end

quad_area(s::AbstractCubicalComplex2D) = dx(s) * dy(s)

function dual_point(s::UniformCubicalComplex2D, x::Int, y::Int)
  px = base_x(s) + (x - 0.5) * dx(s) - hx(s) * dx(s)
  py = base_y(s) + (y - 0.5) * dy(s) - hy(s) * dy(s)
  return Point3(px, py, 0.0)
end

dual_points(s::AbstractCubicalComplex2D) = map(v -> dual_point(s, quad_to_coord(s, v)...), quads(s))

real_dual_point(s::AbstractCubicalComplex2D, x::Int, y::Int) = dual_point(s, x + hx(s), y + hy(s))

function dual_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align)
  if align == X_ALIGN
    return coord_to_edge(s, x, y, Y_ALIGN)
  elseif align == Y_ALIGN
    return coord_to_edge(s, x, y, X_ALIGN)
  else
    return 0
  end
end

# This function computes the length of the dual edge corresponding to the given primal edge
# This the same as the primal edges except on the boundary, where the dual edge is half the length of the primal edge
function dual_edge_len(s::AbstractCubicalComplex2D{FT}, x::Int, y::Int, align::Align) where FT <: AbstractFloat
  if align == X_ALIGN
    if y == 1 || y == ny(s)
      return 0.5 * dy(s)
    else
      return dy(s)
    end
  elseif align == Y_ALIGN
    if x == 1 || x == nx(s)
      return 0.5 * dx(s)
    else
      return dx(s)
    end
  else
    return zero(FT)
  end
end

dual_edge_len(s::AbstractCubicalComplex, e::Int) = dual_edge_len(s, edge_to_coord(s, e)...)

dual_quad(s::AbstractCubicalComplex2D, x::Int, y::Int) = coord_to_vert(s, x, y)

# This function computes the area of the dual quad corresponding to the given primal vertex
# This is the same as the primal quad area except on the boundary, where the dual quad area is half the area of the primal quad
# Also on the corners, the dual quad area is one quarter the area of the primal quad
function dual_quad_area(s::AbstractCubicalComplex2D, x::Int, y::Int)
  if (x == 1 || x == nx(s)) && (y == 1 || y == ny(s))
    return 0.25 * quad_area(s)
  elseif x == 1 || x == nx(s) || y == 1 || y == ny(s)
    return 0.5 * quad_area(s)
  else
    return quad_area(s)
  end
end

dual_quad_area(s::AbstractCubicalComplex2D, v::Int) = begin
  x, y = vert_to_coord(s, v)
  return dual_quad_area(s, x, y)
end

is_boundary_vert(s::AbstractCubicalComplex2D, x::Int, y::Int) = (x == 1 || x == nx(s) || y == 1 || y == ny(s))

is_left_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align) = align == Y_ALIGN && x == 1
is_right_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align) = align == Y_ALIGN && x == nx(s)
is_bottom_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align) = align == X_ALIGN && y == 1
is_top_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align) = align == X_ALIGN && y == ny(s)
function is_boundary_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align)
  return is_left_edge(s, x, y, align) ||
    is_right_edge(s, x, y, align) ||
    is_bottom_edge(s, x, y, align) ||
    is_top_edge(s, x, y, align)
end

# This function returns the two quads that are adjacent to the given edge, ordered with the quad on the left of the edge coming first
function edge_quads(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align)
  if align == X_ALIGN
    q1 = coord_to_quad(s, x, y - 1)
    q2 = coord_to_quad(s, x, y)
    return q1, q2
  elseif align == Y_ALIGN
    q1 = coord_to_quad(s, x - 1, y)
    q2 = coord_to_quad(s, x, y)
    return q1, q2
  else
    return 0, 0
  end
end

# This function returns the four edges that are adjacent to the given vertex, ordered with x-aligned edges coming before y-aligned edges and with edges ordered counterclockwise starting from the edge in the positive x direction
function vert_edges(s::AbstractCubicalComplex2D, x::Int, y::Int)
  e1 = coord_to_edge(s, x, y, X_ALIGN)
  e2 = coord_to_edge(s, x, y, Y_ALIGN)
  e3 = coord_to_edge(s, x - 1, y, Y_ALIGN)
  e4 = coord_to_edge(s, x, y - 1, X_ALIGN)
  return e1, e2, e3, e4
end

# This function returns the four quads that are adjacent to the given vertex, ordered with quads ordered counterclockwise starting from bottom left
function vert_quads(s::AbstractCubicalComplex2D, x::Int, y::Int)
  q1 = coord_to_quad(s, x - 1, y - 1)
  q2 = coord_to_quad(s, x, y - 1)
  q3 = coord_to_quad(s, x, y)
  q4 = coord_to_quad(s, x - 1, y)
  return q1, q2, q3, q4
end

function ghost_quads(s::AbstractCubicalComplex2D)
  function slabs(n_ax, n_b, h, to_idx)
      h == 0 && return (Int[], Int[], Int[], Int[])
      recv_low  = [to_idx(ax, b) for ax in 1:h,              b in 1:n_b][:]
      send_low  = [to_idx(ax, b) for ax in (h+1):(2h),       b in 1:n_b][:]
      send_high = [to_idx(ax, b) for ax in (n_ax-2h+1):(n_ax-h), b in 1:n_b][:]
      recv_high = [to_idx(ax, b) for ax in (n_ax-h+1):n_ax,  b in 1:n_b][:]
      return recv_low, send_low, send_high, recv_high
  end

  rl_ew, sl_ew, sh_ew, rh_ew = slabs(nxq(s), nyq(s), hx(s), (ax, b) -> coord_to_quad(s, ax, b))
  rl_ns, sl_ns, sh_ns, rh_ns = slabs(nyq(s), nxq(s), hy(s), (ax, b) -> coord_to_quad(s, b, ax))

  return (
      west  = (send = sl_ew, recv = rl_ew),
      east  = (send = sh_ew, recv = rh_ew),
      south = (send = sl_ns, recv = rl_ns),
      north = (send = sh_ns, recv = rh_ns),
  )
end