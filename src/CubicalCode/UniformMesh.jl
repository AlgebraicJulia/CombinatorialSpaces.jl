# Points will be given implicitly using the user defined size of domain and number of points in each direction.
# Edges will be given a coordinate based on their source point and their alignment (x or y).
# Quads will be given a coordinate based on their lower left point.

# Default edge orientation will be from smaller to greater vertex index, so x-aligned edges will be oriented in the positive x direction and y-aligned edges will be oriented in the positive y direction.
# Default quad orientation will be counterclockwise starting from the lower left vertex.

using GeometryBasics

import Base: show

abstract type AbstractCubicalComplex end

abstract type AbstractCubicalComplex2D <: AbstractCubicalComplex end

abstract type AbstractEmbeddedCubicalComplex2D{FT<:AbstractFloat} <: AbstractCubicalComplex2D end

struct UniformCubicalComplex2D{FT <: AbstractFloat} <: AbstractEmbeddedCubicalComplex2D{FT}
    nx::Int
    ny::Int

    dx::FT
    dy::FT

    halo_west::Int;  halo_east::Int
    halo_south::Int; halo_north::Int

    base_x::FT
    base_y::FT
end

struct PseudoCubicalMesh2D <: AbstractCubicalComplex2D
    nx::Int
    ny::Int

    halo_west::Int;  halo_east::Int
    halo_south::Int; halo_north::Int
end

@enum Align X_ALIGN Y_ALIGN Z_ALIGN

@enum GridSide EASTWEST NORTHSOUTH UPDOWN ALL

# Base point of real mesh will be (halo_x + 1, halo_y + 1)
# End point of real mesh will be (halo_x + nxr, halo_y + nyr)

base_x(s::UniformCubicalComplex2D) = s.base_x
base_y(s::UniformCubicalComplex2D) = s.base_y

halo_west(s::AbstractCubicalComplex2D)  = s.halo_west
halo_east(s::AbstractCubicalComplex2D)  = s.halo_east
halo_south(s::AbstractCubicalComplex2D) = s.halo_south
halo_north(s::AbstractCubicalComplex2D) = s.halo_north

nxr(s::AbstractCubicalComplex2D) = s.nx
nyr(s::AbstractCubicalComplex2D) = s.ny

nx(s::AbstractCubicalComplex2D) = nxr(s) + halo_west(s) + halo_east(s)
ny(s::AbstractCubicalComplex2D) = nyr(s) + halo_south(s) + halo_north(s)

dx(s::UniformCubicalComplex2D) = s.dx
dy(s::UniformCubicalComplex2D) = s.dy

nv(s::AbstractCubicalComplex2D) = nx(s) * ny(s)
nvr(s::AbstractCubicalComplex2D) = nxr(s) * nyr(s)

nxe(s::AbstractCubicalComplex2D) = nx(s) - 1
nye(s::AbstractCubicalComplex2D) = ny(s) - 1

nxe_r(s::AbstractCubicalComplex2D) = nxe(s) - halo_west(s) - halo_east(s)
nye_r(s::AbstractCubicalComplex2D) = nye(s) - halo_south(s) - halo_north(s)

nxedges(s::AbstractCubicalComplex2D) = nxe(s) * ny(s)
nyedges(s::AbstractCubicalComplex2D) = nx(s) * nye(s)

ne(s::AbstractCubicalComplex2D) = nxedges(s) + nyedges(s)

nxq(s::AbstractCubicalComplex2D) = nx(s) - 1
nyq(s::AbstractCubicalComplex2D) = ny(s) - 1
nquads(s::AbstractCubicalComplex2D) = nxq(s) * nyq(s)

nxqr(s::AbstractCubicalComplex2D) = nxq(s) - halo_west(s) - halo_east(s)
nyqr(s::AbstractCubicalComplex2D) = nyq(s) - halo_south(s) - halo_north(s)
nquadsr(s::AbstractCubicalComplex2D) = nxqr(s) * nyqr(s)

vertices(s::AbstractCubicalComplex2D) = 1:nv(s)
edges(s::AbstractCubicalComplex2D) = 1:ne(s)
quads(s::AbstractCubicalComplex2D) = 1:nquads(s)

top_edges(s::AbstractCubicalComplex2D) = coord_to_edge.(Ref(s), 1:nxe(s), Ref(ny(s)), Ref(X_ALIGN))
bottom_edges(s::AbstractCubicalComplex2D) = coord_to_edge.(Ref(s), 1:nxe(s), Ref(1), Ref(X_ALIGN))
left_edges(s::AbstractCubicalComplex2D) = coord_to_edge.(Ref(s), Ref(1), 1:nye(s), Ref(Y_ALIGN))
right_edges(s::AbstractCubicalComplex2D) = coord_to_edge.(Ref(s), Ref(nx(s)), 1:nye(s), Ref(Y_ALIGN))

function boundary_edges(s::AbstractCubicalComplex2D)
    return vcat(bottom_edges(s), top_edges(s), left_edges(s), right_edges(s))
end

function top_edges_real(s::AbstractCubicalComplex2D)
    y_top = halo_south(s) + nyr(s)
    return coord_to_edge.(Ref(s), (halo_west(s) + 1):(halo_west(s) + nxe_r(s)), Ref(y_top), Ref(X_ALIGN))
end

function bottom_edges_real(s::AbstractCubicalComplex2D)
    y_bot = halo_south(s) + 1
    return coord_to_edge.(Ref(s), (halo_west(s) + 1):(halo_west(s) + nxe_r(s)), Ref(y_bot), Ref(X_ALIGN))
end

function left_edges_real(s::AbstractCubicalComplex2D)
    x_left = halo_west(s) + 1
    return coord_to_edge.(Ref(s), Ref(x_left), (halo_south(s) + 1):(halo_south(s) + nye_r(s)), Ref(Y_ALIGN))
end

function right_edges_real(s::AbstractCubicalComplex2D)
    x_right = halo_west(s) + nxr(s)
    return coord_to_edge.(Ref(s), Ref(x_right), (halo_south(s) + 1):(halo_south(s) + nye_r(s)), Ref(Y_ALIGN))
end

function boundary_edges_real(s::AbstractCubicalComplex2D)
    return vcat(bottom_edges_real(s), top_edges_real(s), left_edges_real(s), right_edges_real(s))
end

coord_to_vert(s::AbstractCubicalComplex2D, x::Int, y::Int) = x + (y - 1) * nx(s)
function coord_to_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align)
    if align == X_ALIGN
        return x + (y - 1) * nxe(s)
    elseif align == Y_ALIGN
        return x + (y - 1) * nx(s) + nxedges(s)
    else
        throw(ArgumentError("Z_ALIGN is not valid for a 2D mesh (coord_to_edge called with align=$align)"))
    end
end
coord_to_quad(s::AbstractCubicalComplex2D, x::Int, y::Int) = (y - 1) * nxq(s) + x

is_X_aligned(e::Int, s::AbstractCubicalComplex2D) = e <= nxedges(s)
is_Y_aligned(e::Int, s::AbstractCubicalComplex2D) = e > nxedges(s)

function real_coord_to_vert(s::AbstractCubicalComplex2D, x::Int, y::Int)
    return coord_to_vert(s, x + halo_west(s), y + halo_south(s))
end
# This function takes a coordinate on the interior and maps it to a vertex index
real_coord_to_real_vert(s::AbstractCubicalComplex2D, x::Int, y::Int) = x + (y - 1) * nxr(s)
coord_to_real_coord(s::AbstractCubicalComplex2D, x::Int, y::Int) = (x - halo_west(s), y - halo_south(s))
function vert_to_real_vert(s::AbstractCubicalComplex2D, v::Int)
    x, y = vert_to_coord(s, v)
    x, y = coord_to_real_coord(s, x, y)
    return real_coord_to_real_vert(s, x, y)
end

function is_halo_vert(s::AbstractCubicalComplex2D, x::Int, y::Int)
    return x <= halo_west(s) || x > nxr(s) + halo_west(s) ||
           y <= halo_south(s) || y > nyr(s) + halo_south(s)
end

function is_halo_quad(s::AbstractCubicalComplex2D, x::Int, y::Int)
    return x <= halo_west(s) || x > nxr(s) + halo_west(s) - 1 ||
           y <= halo_south(s) || y > nyr(s) + halo_south(s) - 1
end

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
    x, y, align = edge_to_coord(s, e)   # bounds-checked above
    return align == X_ALIGN ? (x, y, Y_ALIGN) : (x, y, X_ALIGN)
end

function quad_to_coord(s::AbstractCubicalComplex2D, q::Int)
    y = div(q - 1, nxq(s)) + 1
    x = q - (y - 1) * nxq(s)
    return x, y
end

# Origin is at the first real point (non-halo)
function point(s::UniformCubicalComplex2D{FT}, x::Int, y::Int) where FT <: AbstractFloat
    px = base_x(s) + (x - 1 - halo_west(s)) * dx(s)
    py = base_y(s) + (y - 1 - halo_south(s)) * dy(s)
    return Point3(px, py, FT(0.0))
end

point(s::UniformCubicalComplex2D, v::Int) = point(s, vert_to_coord(s, v)...)

real_point(s::AbstractCubicalComplex2D, x::Int, y::Int) = point(s, x + halo_west(s), y + halo_south(s))

points(s::UniformCubicalComplex2D) = (point(s, v) for v in vertices(s))

spacing(len::FT, np::Int) where {FT <: AbstractFloat} = len / FT(np - 1)

function _validate_mesh_inputs_2d(nxr, nyr, lx, ly, halo_west, halo_east, halo_south, halo_north, base_x, base_y)
    # Point counts
    nxr >= 2 || throw(ArgumentError("nxr must be at least 2 (got $nxr)"))
    nyr >= 2 || throw(ArgumentError("nyr must be at least 2 (got $nyr)"))

    # Physical lengths
    isfinite(lx) && lx > 0 || throw(ArgumentError("lx must be finite and positive (got $lx)"))
    isfinite(ly) && ly > 0 || throw(ArgumentError("ly must be finite and positive (got $ly)"))

    # Halo widths
    halo_west  >= 0 || throw(ArgumentError("halo_west must be non-negative (got $halo_west)"))
    halo_east  >= 0 || throw(ArgumentError("halo_east must be non-negative (got $halo_east)"))
    halo_south >= 0 || throw(ArgumentError("halo_south must be non-negative (got $halo_south)"))
    halo_north >= 0 || throw(ArgumentError("halo_north must be non-negative (got $halo_north)"))

    # Halo widths must not exceed the real domain so that interior slabs exist
    halo_west + halo_east < nxr || throw(ArgumentError("total x halo ($(halo_west + halo_east)) must be less than nxr ($nxr)"))
    halo_south + halo_north < nyr || throw(ArgumentError("total y halo ($(halo_south + halo_north)) must be less than nyr ($nyr)"))

    # Base coordinates
    isfinite(base_x) || throw(ArgumentError("base_x must be finite (got $base_x)"))
    isfinite(base_y) || throw(ArgumentError("base_y must be finite (got $base_y)"))

    return nothing
end

function UniformCubicalComplex2D(nxr::Int, nyr::Int, lx::Real, ly::Real;
        halo_x::Int = 0, halo_y::Int = 0,
        halo_west::Int  = halo_x, halo_east::Int  = halo_x,
        halo_south::Int = halo_y, halo_north::Int = halo_y,
        base_x::Real = 0.0, base_y::Real = 0.0)

    FT = float(promote_type(typeof(lx), typeof(ly)))
    _lx, _ly         = FT(lx), FT(ly)
    _base_x, _base_y = FT(base_x), FT(base_y)

    _validate_mesh_inputs_2d(nxr, nyr, _lx, _ly, halo_west, halo_east, halo_south, halo_north, _base_x, _base_y)

    _dx = spacing(_lx, nxr)
    _dy = spacing(_ly, nyr)

    return UniformCubicalComplex2D{FT}(nxr, nyr, _dx, _dy, halo_west, halo_east, halo_south, halo_north, _base_x, _base_y)
end

UniformCubicalComplex(nx::Int, ny::Int, lx::Real, ly::Real; kwargs...) =
    UniformCubicalComplex2D(nx, ny, lx, ly; kwargs...)

    function PseudoCubicalMesh2D(nx::Int, ny::Int;
        halo_x::Int = 0, halo_y::Int = 0,
        halo_west::Int  = halo_x, halo_east::Int  = halo_x,
        halo_south::Int = halo_y, halo_north::Int = halo_y)

    nx >= 2 || throw(ArgumentError("nx must be at least 2 (got $nx)"))
    ny >= 2 || throw(ArgumentError("ny must be at least 2 (got $ny)"))
    halo_west  >= 0 || throw(ArgumentError("halo_west must be non-negative (got $halo_west)"))
    halo_east  >= 0 || throw(ArgumentError("halo_east must be non-negative (got $halo_east)"))
    halo_south >= 0 || throw(ArgumentError("halo_south must be non-negative (got $halo_south)"))
    halo_north >= 0 || throw(ArgumentError("halo_north must be non-negative (got $halo_north)"))
    halo_west + halo_east < nx || throw(ArgumentError(
        "total x halo ($(halo_west + halo_east)) must be less than nx ($nx)"))
    halo_south + halo_north < ny || throw(ArgumentError(
        "total y halo ($(halo_south + halo_north)) must be less than ny ($ny)"))

    return PseudoCubicalMesh2D(nx, ny, halo_west, halo_east, halo_south, halo_north)
end

PseudoCubicalMesh(nx::Int, ny::Int; kwargs...) = PseudoCubicalMesh2D(nx, ny; kwargs...)


# Basic show method for uniform mesh
function Base.show(io::IO, s::UniformCubicalComplex2D)
    println(io, "UniformCubicalComplex2D with dimensions: $(nx(s)) x $(ny(s))")
    println(io, "Spacing: dx = $(dx(s)), dy = $(dy(s))")
    println(io, "Halo:")
    println(io, " - halo_west = $(s.halo_west), halo_east = $(s.halo_east)")
    println(io, " - halo_south = $(s.halo_south), halo_north = $(s.halo_north)")
    return println(io, "Base point: ($(base_x(s)), $(base_y(s)))")
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
    else
        throw(ArgumentError("Z_ALIGN is not valid for a 2D mesh (tgt called with align=$align)"))
    end
end
tgt(s::AbstractCubicalComplex2D, e::Int) = tgt(s, edge_to_coord(s, e)...)

function edge_len(s::UniformCubicalComplex2D{FT}, align::Align) where {FT <: AbstractFloat}
    if align == X_ALIGN
        return dx(s)
    elseif align == Y_ALIGN
        return dy(s)
    else
        throw(ArgumentError("Z_ALIGN is not valid for a 2D mesh (edge_len called with align=$align)"))
    end
end

# TODO: Needs tests
function edge_vertex_offset(s::AbstractCubicalComplex2D, x::Int, y::Int,
                             align::Align, offset::Int)
    if align == X_ALIGN
        return coord_to_vert(s, x + offset, y)
    else  # Y_ALIGN
        return coord_to_vert(s, x, y + offset)
    end
end

edge_len(s::UniformCubicalComplex2D, x::Int, y::Int, align::Align) = edge_len(s, align)

edge_len(s::AbstractCubicalComplex2D, e::Int) = edge_len(s, edge_to_coord(s, e)...)

xedges(s::AbstractCubicalComplex2D, arr::AbstractVector) = @view arr[1:nxedges(s)]
yedges(s::AbstractCubicalComplex2D, arr::AbstractVector) = @view arr[(nxedges(s) + 1):end]

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

# TODO: Needs tests
# Given a quad, gives the edge offset by the given amount in the given direction
# An offset of zero will give either the left or bottom edge, depending on the direction
function quad_edge_offset(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align, offset::Int)
    if align == X_ALIGN
        return coord_to_edge(s, x, y + offset, X_ALIGN)
    elseif align == Y_ALIGN
        return coord_to_edge(s, x + offset, y, Y_ALIGN)
    else
        throw(ArgumentError(
            "Z_ALIGN is not valid for a 2D mesh (quad_edge_offset called with align=$align)"))
    end
end
quad_area(s::AbstractCubicalComplex2D) = dx(s) * dy(s)

function dual_point(s::UniformCubicalComplex2D{FT}, x::Int, y::Int) where FT <: AbstractFloat
    px = base_x(s) + (x - FT(0.5) - halo_west(s)) * dx(s)
    py = base_y(s) + (y - FT(0.5) - halo_south(s)) * dy(s)
    return Point3(px, py, FT(0.0))
end

dual_points(s::AbstractCubicalComplex2D) = map(v -> dual_point(s, quad_to_coord(s, v)...), quads(s))

real_dual_point(s::AbstractCubicalComplex2D, x::Int, y::Int) = dual_point(s, x + halo_west(s), y + halo_south(s))

function dual_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align)
    if align == X_ALIGN
        return coord_to_edge(s, x, y, Y_ALIGN)
    elseif align == Y_ALIGN
        return coord_to_edge(s, x, y, X_ALIGN)
    else
        throw(ArgumentError("Z_ALIGN is not valid for a 2D mesh (dual_edge called with align=$align)"))
    end
end

function dual_edge_len(s::UniformCubicalComplex2D{FT}, x::Int, y::Int,
                        align::Align) where {FT <: AbstractFloat}
    if align == X_ALIGN
        return y == 1 || y == ny(s) ? FT(0.5) * dy(s) : dy(s)
    elseif align == Y_ALIGN
        return x == 1 || x == nx(s) ? FT(0.5) * dx(s) : dx(s)
    else
        throw(ArgumentError("Z_ALIGN is not valid for a 2D mesh (dual_edge_len called with align=$align)"))
    end
end

dual_edge_len(s::AbstractCubicalComplex, e::Int) = dual_edge_len(s, edge_to_coord(s, e)...)

dual_quad(s::AbstractCubicalComplex2D, x::Int, y::Int) = coord_to_vert(s, x, y)

# This function computes the area of the dual quad corresponding to the given primal vertex
# This is the same as the primal quad area except on the boundary, where the dual quad area is half the area of the primal quad
# Also on the corners, the dual quad area is one quarter the area of the primal quad
function dual_quad_area(s::UniformCubicalComplex2D{FT}, x::Int, y::Int) where FT <: AbstractFloat
    if (x == 1 || x == nx(s)) && (y == 1 || y == ny(s))
        return FT(0.25) * quad_area(s)
    elseif x == 1 || x == nx(s) || y == 1 || y == ny(s)
        return FT(0.5) * quad_area(s)
    else
        return quad_area(s)
    end
end

dual_quad_area(s::AbstractCubicalComplex2D, v::Int) = begin
    x, y = vert_to_coord(s, v)
    return dual_quad_area(s, x, y)
end

function is_boundary_vert(s::AbstractCubicalComplex2D, x::Int, y::Int)
    return (x == 1 || x == nx(s) || y == 1 || y == ny(s))
end

is_left_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align) = align == Y_ALIGN && x == 1
function is_right_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align)
    return align == Y_ALIGN && x == nx(s)
end
function is_bottom_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align)
    return align == X_ALIGN && y == 1
end
function is_top_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align)
    return align == X_ALIGN && y == ny(s)
end
function is_boundary_edge(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align)
    return is_left_edge(s, x, y, align) || is_right_edge(s, x, y, align) || is_bottom_edge(s, x, y, align) || is_top_edge(s, x, y, align)
end

# This function returns the two quads that are adjacent to the given edge, ordered with the quad on the left of the edge coming first
function edge_quads(s::AbstractCubicalComplex2D, x::Int, y::Int, align::Align)
    if align == X_ALIGN
        q1 = y > 1          ? coord_to_quad(s, x, y - 1) : 0
        q2 = y <= nyq(s)    ? coord_to_quad(s, x, y)     : 0
        return q1, q2
    elseif align == Y_ALIGN
        q1 = x > 1          ? coord_to_quad(s, x - 1, y) : 0
        q2 = x <= nxq(s)    ? coord_to_quad(s, x, y)     : 0
        return q1, q2
    else
        throw(ArgumentError("Z_ALIGN is not valid for a 2D mesh (edge_quads called with align=$align)"))
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
    function low_slab(n_ax, n_b, h_low, to_idx)
        h_low == 0 && return (send = Int[], recv = Int[])
        return (
            send = [to_idx(ax, b) for ax in (h_low + 1):(2h_low), b in 1:n_b][:],
            recv = [to_idx(ax, b) for ax in 1:h_low,              b in 1:n_b][:],
        )
    end

    function high_slab(n_ax, n_b, h_high, to_idx)
        h_high == 0 && return (send = Int[], recv = Int[])
        return (
            send = [to_idx(ax, b) for ax in (n_ax - 2h_high + 1):(n_ax - h_high), b in 1:n_b][:],
            recv = [to_idx(ax, b) for ax in (n_ax - h_high + 1):n_ax,             b in 1:n_b][:],
        )
    end

    ew = (ax, b) -> coord_to_quad(s, ax, b)
    ns = (ax, b) -> coord_to_quad(s, b, ax)

    return (
        west  = low_slab( nxq(s), nyq(s), halo_west(s),  ew),
        east  = high_slab(nxq(s), nyq(s), halo_east(s),  ew),
        south = low_slab( nyq(s), nxq(s), halo_south(s), ns),
        north = high_slab(nyq(s), nxq(s), halo_north(s), ns),
    )
end
