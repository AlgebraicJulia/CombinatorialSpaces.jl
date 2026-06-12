using GeometryBasics

abstract type AbstractCubicalComplex3D{FT <: AbstractFloat} <: AbstractCubicalComplex end

struct UniformCubicalComplex3D{FT} <: AbstractCubicalComplex3D{FT}
    nx::Int
    ny::Int
    nz::Int

    dx::FT
    dy::FT
    dz::FT

    halo_x::Int
    halo_y::Int
    halo_z::Int

    base_x::FT
    base_y::FT
    base_z::FT
end

base_x(s::UniformCubicalComplex3D) = s.base_x
base_y(s::UniformCubicalComplex3D) = s.base_y
base_z(s::UniformCubicalComplex3D) = s.base_z

# Vertices

# This grabs the number of real points in the x and y directions, excluding halo points
nxr(s::AbstractCubicalComplex3D) = s.nx
nyr(s::AbstractCubicalComplex3D) = s.ny
nzr(s::AbstractCubicalComplex3D) = s.nz

hx(s::AbstractCubicalComplex3D) = s.halo_x
hy(s::AbstractCubicalComplex3D) = s.halo_y
hz(s::AbstractCubicalComplex3D) = s.halo_z

nx(s::AbstractCubicalComplex3D) = nxr(s) + 2 * hx(s)
ny(s::AbstractCubicalComplex3D) = nyr(s) + 2 * hy(s)
nz(s::AbstractCubicalComplex3D) = nzr(s) + 2 * hz(s)

dx(s::UniformCubicalComplex3D) = s.dx
dy(s::UniformCubicalComplex3D) = s.dy
dz(s::UniformCubicalComplex3D) = s.dz

nv(s::AbstractCubicalComplex3D) = nx(s) * ny(s) * nz(s)
nvr(s::AbstractCubicalComplex3D) = nxr(s) * nyr(s) * nzr(s)

# Edges

# Number of edges along an axis
nxe(s::AbstractCubicalComplex3D) = nx(s) - 1
nye(s::AbstractCubicalComplex3D) = ny(s) - 1
nze(s::AbstractCubicalComplex3D) = nz(s) - 1

nxe_r(s::AbstractCubicalComplex3D) = nxe(s) - 2 * hx(s)
nye_r(s::AbstractCubicalComplex3D) = nye(s) - 2 * hy(s)
nze_r(s::AbstractCubicalComplex3D) = nze(s) - 2 * hz(s)

# Total of axis-aligned edges
nxedges(s::AbstractCubicalComplex3D) = nxe(s) * ny(s) * nz(s)
nyedges(s::AbstractCubicalComplex3D) = nx(s) * nye(s) * nz(s)
nzedges(s::AbstractCubicalComplex3D) = nx(s) * ny(s) * nze(s)

ne(s::AbstractCubicalComplex3D) = nxedges(s) + nyedges(s) + nzedges(s)

# Quadrilaterals

# Number of quads along an axis
nxq(s::AbstractCubicalComplex3D) = nx(s) - 1
nyq(s::AbstractCubicalComplex3D) = ny(s) - 1
nzq(s::AbstractCubicalComplex3D) = nz(s) - 1

# Number of quads along a plane
nxyq(s::AbstractCubicalComplex3D) = nxq(s) * nyq(s)
nxzq(s::AbstractCubicalComplex3D) = nxq(s) * nzq(s)
nyzq(s::AbstractCubicalComplex3D) = nyq(s) * nzq(s)

# Total of axis-aligned quads
nxyquads(s::AbstractCubicalComplex3D) = nxyq(s) * nz(s)
nxzquads(s::AbstractCubicalComplex3D) = nxzq(s) * ny(s)
nyzquads(s::AbstractCubicalComplex3D) = nyzq(s) * nx(s)

nquads(s::AbstractCubicalComplex3D) = nxyquads(s) + nxzquads(s) + nyzquads(s)

hxq(s::AbstractCubicalComplex3D) = hx(s)
hyq(s::AbstractCubicalComplex3D) = hy(s)
hzq(s::AbstractCubicalComplex3D) = hz(s)
hzquads(s::AbstractCubicalComplex3D) = hz(s)

# Rectangular cuboids

nxb(s::AbstractCubicalComplex3D) = nx(s) - 1
nyb(s::AbstractCubicalComplex3D) = ny(s) - 1
nzb(s::AbstractCubicalComplex3D) = nz(s) - 1

nboidsr(s::AbstractCubicalComplex3D) = nxbr(s) * nybr(s) * nzbr(s)

nxbr(s::AbstractCubicalComplex3D) = nxr(s) - 1
nybr(s::AbstractCubicalComplex3D) = nyr(s) - 1
nzbr(s::AbstractCubicalComplex3D) = nzr(s) - 1

nxyb(s::AbstractCubicalComplex3D) = nxb(s) * nyb(s)
nxzb(s::AbstractCubicalComplex3D) = nxb(s) * nzb(s)
nyzb(s::AbstractCubicalComplex3D) = nyb(s) * nzb(s)

nboids(s::AbstractCubicalComplex3D) = nxb(s) * nyb(s) * nzb(s)

hxb(s::AbstractCubicalComplex3D) = hx(s)
hyb(s::AbstractCubicalComplex3D) = hy(s)
hzb(s::AbstractCubicalComplex3D) = hz(s)

vertices(s::AbstractCubicalComplex3D) = 1:nv(s)
edges(s::AbstractCubicalComplex3D) = 1:ne(s)
quads(s::AbstractCubicalComplex3D) = 1:nquads(s)
boids(s::AbstractCubicalComplex3D) = 1:nboids(s)

valid_xedge(s::AbstractCubicalComplex3D, x, y, z) = (1 <= x <= nxe(s)) && (1 <= y <= ny(s)) && (1 <= z <= nz(s))
valid_yedge(s::AbstractCubicalComplex3D, x, y, z) = (1 <= x <= nx(s)) && (1 <= y <= nye(s)) && (1 <= z <= nz(s))
valid_zedge(s::AbstractCubicalComplex3D, x, y, z) = (1 <= x <= nx(s)) && (1 <= y <= ny(s)) && (1 <= z <= nze(s))

valid_xyquad(s::AbstractCubicalComplex3D, x, y, z) = (1 <= x <= nxq(s)) && (1 <= y <= nyq(s)) && (1 <= z <= nz(s))
valid_xzquad(s::AbstractCubicalComplex3D, x, y, z) = (1 <= x <= nxq(s)) && (1 <= y <= ny(s)) && (1 <= z <= nzq(s))
valid_yzquad(s::AbstractCubicalComplex3D, x, y, z) = (1 <= x <= nx(s)) && (1 <= y <= nyq(s)) && (1 <= z <= nzq(s))

valid_boid(s::AbstractCubicalComplex3D, x, y, z) = (1 <= x <= nxb(s)) && (1 <= y <= nyb(s)) && (1 <= z <= nzb(s))

lx(s::UniformCubicalComplex3D) = nxe(s) * dx(s)
ly(s::UniformCubicalComplex3D) = nye(s) * dy(s)
lz(s::UniformCubicalComplex3D) = nze(s) * dz(s)

function UniformCubicalComplex3D(nxr::Int, nyr::Int, nzr::Int,
                                 lx::FT, ly::FT, lz::FT;
                                 halo_x::Int=0, halo_y::Int=0, halo_z::Int=0,
                                 base_x::FT=zero(FT), base_y::FT=zero(FT), base_z::FT=zero(FT)
                                 ) where FT <: AbstractFloat

    dx = spacing(lx, nxr)
    dy = spacing(ly, nyr)
    dz = spacing(lz, nzr)
    UniformCubicalComplex3D(
    nxr, nyr, nzr,
    dx, dy, dz,
    halo_x, halo_y, halo_z,
    base_x, base_y, base_z)
end

function Base.show(io::IO, s::UniformCubicalComplex3D)
    println(io, "UniformCubicalComplex3D with dimensions: $(nx(s)) x $(ny(s)) x $(nz(s))")
    println(io, "Spacing: dx = $(dx(s)), dy = $(dy(s)), dz = $(dz(s))")
    println(io, "Halo: halo_x = $(s.halo_x), halo_y = $(s.halo_y), halo_z = $(s.halo_z)")
    println(io, "Base point: ($(base_x(s)), $(base_y(s)), $(base_z(s)))")

    # TODO: Change these into real edge numbers
    lx = (nxr(s) - 1) * dx(s)
    ly = (nyr(s) - 1) * dy(s)
    lz = (nzr(s) - 1) * dz(s)
    println(io, "Physical domain: lx = $(lx), ly = $(ly), lz = $(lz)")
end

coord_to_vert(s::AbstractCubicalComplex3D, x::Int, y::Int, z::Int) = x + (y - 1) * nx(s) + (z - 1) * nx(s) * ny(s)

# First x-aligned, then y-aligned, then z-aligned
function coord_to_edge(s::AbstractCubicalComplex3D, x::Int, y::Int, z::Int, align::Align)
    if align == X_ALIGN
        return x + (y - 1) * nxe(s) + (z - 1) * nxe(s) * ny(s)
    elseif align == Y_ALIGN
        return x + (y - 1) * nx(s) + (z - 1) * nx(s) * nye(s) + nxedges(s)
    else # align == Z_ALIGN
        return  x + (y - 1) * nx(s) + (z - 1) * nx(s) * ny(s) + nxedges(s) + nyedges(s)
    end
end

function coord_to_quad(s::AbstractCubicalComplex3D, x::Int, y::Int, z::Int, align::Align)
    if align == Z_ALIGN
        return x + (y - 1) * nxq(s) + (z - 1) * nxq(s) * nyq(s)
    elseif align == Y_ALIGN
        return x + (y - 1) * nxq(s) + (z - 1) * nxq(s) * ny(s) + nxyquads(s)
    else # align == X_ALIGN
        return x + (y - 1) * nx(s) + (z - 1) * nx(s) * nye(s) + nxyquads(s) + nxzquads(s)
    end
end

coord_to_boid(s::AbstractCubicalComplex3D, x::Int, y::Int, z::Int) = x + (y - 1) * nxb(s) + (z - 1) * nxyb(s)

is_edge_X_aligned(e::Int, s::AbstractCubicalComplex3D) = e <= nxedges(s)
is_edge_Y_aligned(e::Int, s::AbstractCubicalComplex3D) = nxedges(s) < e <= nxedges(s) + nyedges(s)
is_edge_Z_aligned(e::Int, s::AbstractCubicalComplex3D) = e > nxedges(s) + nyedges(s)

function vert_to_coord(s::UniformCubicalComplex3D, idx::Int)
    idx0 = idx - 1
    nxy = nx(s) * ny(s)
    
    z = (idx0 ÷ nxy) + 1
    rem_z = idx0 % nxy
    y = (rem_z ÷ nx(s)) + 1
    x = (rem_z % nx(s)) + 1
    
    return (x, y, z)
end

function edge_to_coord(s::UniformCubicalComplex3D, idx::Int)
    if idx <= nxedges(s)
        align = X_ALIGN
        idx0 = idx - 1
        nxe_val = nxe(s)
        nxy_e = nxe_val * ny(s)
        
        z = (idx0 ÷ nxy_e) + 1
        rem_z = idx0 % nxy_e
        y = (rem_z ÷ nxe_val) + 1
        x = (rem_z % nxe_val) + 1
        
    elseif idx <= nxedges(s) + nyedges(s)
        align = Y_ALIGN
        idx0 = idx - nxedges(s) - 1
        nxe_val = nx(s)
        nxy_e = nxe_val * nye(s)
        
        z = (idx0 ÷ nxy_e) + 1
        rem_z = idx0 % nxy_e
        y = (rem_z ÷ nxe_val) + 1
        x = (rem_z % nxe_val) + 1
        
    else
        align = Z_ALIGN
        idx0 = idx - nxedges(s) - nyedges(s) - 1
        nxe_val = nx(s)
        nxy_e = nxe_val * ny(s)
        
        z = (idx0 ÷ nxy_e) + 1
        rem_z = idx0 % nxy_e
        y = (rem_z ÷ nxe_val) + 1
        x = (rem_z % nxe_val) + 1
    end
    
    return (x, y, z, align)
end

function quad_to_coord(s::UniformCubicalComplex3D, idx::Int)
    if idx <= nxyquads(s)
        align = Z_ALIGN
        idx0 = idx - 1
        nx_q = nxb(s)
        nxy_q = nxb(s) * nyb(s)
        
        z = (idx0 ÷ nxy_q) + 1
        rem_z = idx0 % nxy_q
        y = (rem_z ÷ nx_q) + 1
        x = (rem_z % nx_q) + 1
        
    elseif idx <= nxyquads(s) + nxzquads(s)
        align = Y_ALIGN
        idx0 = idx - nxyquads(s) - 1
        nx_q = nxb(s)
        nxy_q = nxb(s) * ny(s)
        
        z = (idx0 ÷ nxy_q) + 1
        rem_z = idx0 % nxy_q
        y = (rem_z ÷ nx_q) + 1
        x = (rem_z % nx_q) + 1
        
    else
        align = X_ALIGN
        idx0 = idx - nxyquads(s) - nxzquads(s) - 1
        nx_q = nx(s)
        nxy_q = nx(s) * nyb(s)
        
        z = (idx0 ÷ nxy_q) + 1
        rem_z = idx0 % nxy_q
        y = (rem_z ÷ nx_q) + 1
        x = (rem_z % nx_q) + 1
    end
    
    return (x, y, z, align)
end

function boid_to_coord(s::UniformCubicalComplex3D, idx::Int)
    idx0 = idx - 1
    nxy_b = nxb(s) * nyb(s)
    
    z = (idx0 ÷ nxy_b) + 1
    rem_z = idx0 % nxy_b
    y = (rem_z ÷ nxb(s)) + 1
    x = (rem_z % nxb(s)) + 1
    
    return (x, y, z)
end

function point(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int)
    px = base_x(s) + (x - 1) * dx(s) - hx(s) * dx(s)
    py = base_y(s) + (y - 1) * dy(s) - hy(s) * dy(s)
    pz = base_z(s) + (z - 1) * dz(s) - hz(s) * dz(s)
    return Point3(px, py, pz)
end
point(s::UniformCubicalComplex3D, v::Int) = point(s, vert_to_coord(s, v)...)

real_point(s::AbstractCubicalComplex3D, x::Int, y::Int, z::Int) = point(s, x + hx(s), y + hy(s), z + hz(s))

points(s::AbstractCubicalComplex3D) = map(v -> point(s, v), vertices(s))

# Edge methods

src(s::AbstractCubicalComplex3D, x::Int, y::Int, z::Int, align::Align) = coord_to_vert(s, x, y, z)
function src(s::AbstractCubicalComplex3D, e::Int)
    x, y, z, align = edge_to_coord(s, e)
    return src(s, x, y, z, align)
end

function tgt(s::AbstractCubicalComplex3D, x::Int, y::Int, z::Int, align::Align)
    if align == X_ALIGN
        return coord_to_vert(s, x + 1, y, z)
    elseif align == Y_ALIGN
        return coord_to_vert(s, x, y + 1, z)
    else # align == Z_ALIGN
        return coord_to_vert(s, x, y, z + 1)
    end
end
function tgt(s::AbstractCubicalComplex3D, e::Int)
    x, y, z, align = edge_to_coord(s, e)
    return tgt(s, x, y, z, align)
end

function edge_len(s::UniformCubicalComplex3D, align::Align)
    if align == X_ALIGN
        return dx(s)
    elseif align == Y_ALIGN
        return dy(s)
    else # align == Z_ALIGN
        return dz(s)
    end
end

edge_len(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int, align::Align) = edge_len(s, align)

xedges(s::AbstractCubicalComplex3D, arr::AbstractVector) = @view arr[1:nxedges(s)]
yedges(s::AbstractCubicalComplex3D, arr::AbstractVector) = @view arr[nxedges(s)+1:nxedges(s)+nyedges(s)]
zedges(s::AbstractCubicalComplex3D, arr::AbstractVector) = @view arr[nxedges(s)+nyedges(s)+1:end]

xyquads(s::AbstractCubicalComplex3D, arr::AbstractVector) = @view arr[1:nxyquads(s)]
xzquads(s::AbstractCubicalComplex3D, arr::AbstractVector) = @view arr[nxyquads(s)+1:nxyquads(s)+nxzquads(s)]
yzquads(s::AbstractCubicalComplex3D, arr::AbstractVector) = @view arr[nxyquads(s)+nxzquads(s)+1:end]

# This returns the vertices counterclockwise
# The ccw direction is determined by the perpendicular axis going from negative to positive
# The first vertex is the one with the smallest index
function quad_vertices(s::AbstractCubicalComplex3D, x::Int, y::Int, z::Int, align::Align)
    if align == Z_ALIGN
        v1 = coord_to_vert(s, x, y, z)
        v2 = coord_to_vert(s, x + 1, y, z)
        v3 = coord_to_vert(s, x + 1, y + 1, z)
        v4 = coord_to_vert(s, x, y + 1, z)
    elseif align == Y_ALIGN
        v1 = coord_to_vert(s, x, y, z)
        v2 = coord_to_vert(s, x, y, z + 1)
        v3 = coord_to_vert(s, x + 1, y, z + 1)
        v4 = coord_to_vert(s, x + 1, y, z)
    else # align == X_ALIGN
        v1 = coord_to_vert(s, x, y, z)
        v2 = coord_to_vert(s, x, y + 1, z)
        v3 = coord_to_vert(s, x, y + 1, z + 1)
        v4 = coord_to_vert(s, x, y, z + 1)
    end
    return (v1, v2, v3, v4)
end

# We provide the edges so that their sources line up with the quad_vertices
function quad_edges(s::AbstractCubicalComplex3D, x::Int, y::Int, z::Int, align::Align)
    if align == Z_ALIGN
        e1 = coord_to_edge(s, x, y, z, X_ALIGN)
        e2 = coord_to_edge(s, x + 1, y, z, Y_ALIGN)
        e3 = coord_to_edge(s, x, y + 1, z, X_ALIGN)
        e4 = coord_to_edge(s, x, y, z, Y_ALIGN)
    elseif align == Y_ALIGN
        e1 = coord_to_edge(s, x, y, z, Z_ALIGN)
        e2 = coord_to_edge(s, x, y, z + 1, X_ALIGN)
        e3 = coord_to_edge(s, x + 1, y, z, Z_ALIGN)
        e4 = coord_to_edge(s, x, y, z, X_ALIGN)
    else # align == X_ALIGN
        e1 = coord_to_edge(s, x, y, z, Y_ALIGN)
        e2 = coord_to_edge(s, x, y + 1, z, Z_ALIGN)
        e3 = coord_to_edge(s, x, y, z + 1, Y_ALIGN)
        e4 = coord_to_edge(s, x, y, z, Z_ALIGN)
    end
    return (e1, e2, e3, e4)
end

function boid_vertices(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int)
    # Lower Z plane (z)
    v1 = coord_to_vert(s, x, y, z)
    v2 = coord_to_vert(s, x + 1, y, z)
    v3 = coord_to_vert(s, x + 1, y + 1, z)
    v4 = coord_to_vert(s, x, y + 1, z)

    # Higher Z plane (z+1)
    v5 = coord_to_vert(s, x, y, z + 1)
    v6 = coord_to_vert(s, x + 1, y, z + 1)
    v7 = coord_to_vert(s, x + 1, y + 1, z + 1)
    v8 = coord_to_vert(s, x, y + 1, z + 1)

    return (v1, v2, v3, v4, v5, v6, v7, v8)
end

""" boid_quads(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int)
Returns the faces of a cuboid in order: Down, Up, South, North, West, East
"""
function boid_quads(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int)
    # Z-aligned faces (XY-quads)
    z_face1_idx = coord_to_quad(s, x, y, z, Z_ALIGN)
    z_face2_idx = coord_to_quad(s, x, y, z + 1, Z_ALIGN)

    # Y-aligned faces (XZ-quads)
    y_face1_idx = coord_to_quad(s, x, y, z, Y_ALIGN)
    y_face2_idx = coord_to_quad(s, x, y + 1, z, Y_ALIGN)

    # X-aligned faces (YZ-quads)
    x_face1_idx = coord_to_quad(s, x, y, z, X_ALIGN)
    x_face2_idx = coord_to_quad(s, x + 1, y, z, X_ALIGN)

    return (z_face1_idx, z_face2_idx, y_face1_idx, y_face2_idx, x_face1_idx, x_face2_idx)
end

function boid_edges(s::AbstractCubicalComplex3D, x::Int, y::Int, z::Int)
    e1 = coord_to_edge(s, x, y, z, X_ALIGN)
    e2 = coord_to_edge(s, x, y + 1, z, X_ALIGN)
    e3 = coord_to_edge(s, x, y, z + 1, X_ALIGN)
    e4 = coord_to_edge(s, x, y + 1, z + 1, X_ALIGN)

    e5 = coord_to_edge(s, x, y, z, Y_ALIGN)
    e6 = coord_to_edge(s, x + 1, y, z, Y_ALIGN)
    e7 = coord_to_edge(s, x, y, z + 1, Y_ALIGN)
    e8 = coord_to_edge(s, x + 1, y, z + 1, Y_ALIGN)

    e9 = coord_to_edge(s, x, y, z, Z_ALIGN)
    e10 = coord_to_edge(s, x + 1, y, z, Z_ALIGN)
    e11 = coord_to_edge(s, x, y + 1, z, Z_ALIGN)
    e12 = coord_to_edge(s, x + 1, y + 1, z, Z_ALIGN)

    return (e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12)
end

function quad_area(s::UniformCubicalComplex3D, align::Align)
    if align == Z_ALIGN
        # Z-aligned quad lies in the XY plane
        return dx(s) * dy(s)
    elseif align == Y_ALIGN
        # Y-aligned quad lies in the XZ plane
        return dx(s) * dz(s)
    elseif align == X_ALIGN
        # X-aligned quad lies in the YZ plane
        return dy(s) * dz(s)
    else
        error("Invalid alignment specified.")
    end
end

function boid_volume(s::UniformCubicalComplex3D)
    return dx(s) * dy(s) * dz(s)
end

function dual_point(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int)
    px = base_x(s) + (x - 0.5) * dx(s) - hx(s) * dx(s)
    py = base_y(s) + (y - 0.5) * dy(s) - hy(s) * dy(s)
    pz = base_z(s) + (z - 0.5) * dz(s) - hz(s) * dz(s)
    return Point3(px, py, pz)
end
dual_points(s::AbstractCubicalComplex3D) = map(v -> dual_point(s, boid_to_coord(s, v)...), boids(s))

function real_dual_point(s::UniformCubicalComplex3D, rx::Int, ry::Int, rz::Int)
    return dual_point(s, rx + hx(s), ry + hy(s), rz + hz(s))
end

real_coord_to_boid(s::AbstractCubicalComplex3D, rx::Int, ry::Int, rz::Int) =
    coord_to_boid(s, rx + hx(s), ry + hy(s), rz + hz(s))

function dual_edge_length(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int, align::Align)
    if align == Z_ALIGN
        # Z-aligned quad (XY plane) normal is along Z. 
        # Boundary is at the first and last Z-coordinates.
        if z == 1 || z == nz(s)
            return dz(s) / 2.0
        else
            return dz(s)
        end
    elseif align == Y_ALIGN
        # Y-aligned quad (XZ plane) normal is along Y.
        # Boundary is at the first and last Y-coordinates.
        if y == 1 || y == ny(s)
            return dy(s) / 2.0
        else
            return dy(s)
        end
    elseif align == X_ALIGN
        # X-aligned quad (YZ plane) normal is along X.
        # Boundary is at the first and last X-coordinates.
        if x == 1 || x == nx(s)
            return dx(s) / 2.0
        else
            return dx(s)
        end
    else
        error("Invalid alignment specified.")
    end
end

function dual_quad_area(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int, align::Align)
    lx = (x == 1 || x == nx(s)) ? dx(s) / 2.0 : dx(s)
    ly = (y == 1 || y == ny(s)) ? dy(s) / 2.0 : dy(s)
    lz = (z == 1 || z == nz(s)) ? dz(s) / 2.0 : dz(s)

    if align == Z_ALIGN
        return lx * ly
    elseif align == Y_ALIGN
        return lx * lz
    else # align == X_ALIGN
        return ly * lz
    end
end

function dual_boid_volume(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int)
    lx = (x == 1 || x == nx(s)) ? dx(s) / 2.0 : dx(s)
    ly = (y == 1 || y == ny(s)) ? dy(s) / 2.0 : dy(s)
    lz = (z == 1 || z == nz(s)) ? dz(s) / 2.0 : dz(s)
    
    return lx * ly * lz
end

"""
    quad_boids(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int, align::Align)

Return the indices of the two primal boids incident to the specified quad. The ordering 
rules below are respectful of the fact that when considering the dual elements,
namely the dual edge moving between dual vertices, the below should give us the src and tgt
combination in that order. We assume that the right-hand rule is positive orientation.

Ordering rules:
- Z-ALIGN (XY-quad): Down boid first, then up boid.
- Y-ALIGN (XZ-quad): South boid first, then north boid.
- X-ALIGN (YZ-quad): West boid first, then east boid.
"""
function quad_boids(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int, align::Align)    
    if align == Z_ALIGN
        b_lower_valid = valid_boid(s, x, y, z - 1)
        b_higher_valid = valid_boid(s, x, y, z)
        b_lower = b_lower_valid ? coord_to_boid(s, x, y, z - 1) : 0
        b_higher = b_higher_valid ? coord_to_boid(s, x, y, z) : 0
        
    elseif align == Y_ALIGN
        b_lower_valid = valid_boid(s, x, y - 1, z)
        b_higher_valid = valid_boid(s, x, y, z)
        b_lower = b_lower_valid ? coord_to_boid(s, x, y - 1, z) : 0
        b_higher = b_higher_valid ? coord_to_boid(s, x, y, z) : 0  

    else # align == X_ALIGN
        b_lower_valid = valid_boid(s, x - 1, y, z)
        b_higher_valid = valid_boid(s, x, y, z)
        b_lower = b_lower_valid ? coord_to_boid(s, x - 1, y, z) : 0
        b_higher = b_higher_valid ? coord_to_boid(s, x, y, z) : 0
    end

    return ((b_lower, b_higher), (b_lower_valid, b_higher_valid))
end

# TODO: Figure out if this dual edge ordering is useful or at least benign
# TODO: Check this code to make sure it is working as intended
"""
    edge_quads(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int, align::Align)

Return the indices of the four primal quads incident to the specified primal edge.
Assumes all coordinates fall within the valid interior of the mesh (no boundary checks).

Note that the ordering rules take a given start quad and then works around the edge in the
counterclockwise direction. This direction is defined as following the edge and using the
righthand rule.

Ordering rules:
- X-Aligned edge: Z-quad at lower y, Y-quad at lower z, Z-quad at higher y, Y-quad at higher z.
- Y-Aligned edge: Z-quad at higher x, X-quad at lower z, Z-quad at lower x, X-quad at higher z.
- Z-Aligned edge: X-quad at lower y, Y-quad at lower x, X-quad at higher y, Y-quad at higher x.
"""
function edge_quads(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int, align::Align)
    if align == X_ALIGN
        v1 = valid_xyquad(s, x, y - 1, z); q1 = v1 ? coord_to_quad(s, x, y - 1, z, Z_ALIGN) : 0
        v2 = valid_xzquad(s, x, y, z - 1); q2 = v2 ? coord_to_quad(s, x, y, z - 1, Y_ALIGN) : 0
        v3 = valid_xyquad(s, x, y, z);     q3 = v3 ? coord_to_quad(s, x, y, z, Z_ALIGN) : 0
        v4 = valid_xzquad(s, x, y, z);     q4 = v4 ? coord_to_quad(s, x, y, z, Y_ALIGN) : 0
    elseif align == Y_ALIGN
        v1 = valid_yzquad(s, x, y, z - 1); q1 = v1 ? coord_to_quad(s, x, y, z - 1, X_ALIGN) : 0
        v2 = valid_xyquad(s, x - 1, y, z); q2 = v2 ? coord_to_quad(s, x - 1, y, z, Z_ALIGN) : 0
        v3 = valid_yzquad(s, x, y, z);     q3 = v3 ? coord_to_quad(s, x, y, z, X_ALIGN) : 0
        v4 = valid_xyquad(s, x, y, z);     q4 = v4 ? coord_to_quad(s, x, y, z, Z_ALIGN) : 0       
    else # align == Z_ALIGN
        v1 = valid_xzquad(s, x - 1, y, z); q1 = v1 ? coord_to_quad(s, x - 1, y, z, Y_ALIGN) : 0
        v2 = valid_yzquad(s, x, y - 1, z); q2 = v2 ? coord_to_quad(s, x, y - 1, z, X_ALIGN) : 0
        v3 = valid_xzquad(s, x, y, z);     q3 = v3 ? coord_to_quad(s, x, y, z, Y_ALIGN) : 0
        v4 = valid_yzquad(s, x, y, z);     q4 = v4 ? coord_to_quad(s, x, y, z, X_ALIGN) : 0
    end

    return ((q1, q2, q3, q4), (v1, v2, v3, v4)) 
end

function edge_boids(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int, align::Align)
    if align == X_ALIGN
        # Boids around an X-aligned edge (y,z plane)
        v1 = valid_boid(s, x, y - 1, z - 1); b1_idx = v1 ? coord_to_boid(s, x, y - 1, z - 1) : 0
        v2 = valid_boid(s, x, y,     z - 1); b2_idx = v2 ? coord_to_boid(s, x, y,     z - 1) : 0
        v3 = valid_boid(s, x, y,     z);     b3_idx = v3 ? coord_to_boid(s, x, y,     z) : 0
        v4 = valid_boid(s, x, y - 1, z);     b4_idx = v4 ? coord_to_boid(s, x, y - 1, z) : 0
    elseif align == Y_ALIGN
        # Boids around a Y-aligned edge (x,z plane)
        v1 = valid_boid(s, x,     y, z - 1); b1_idx = v1 ? coord_to_boid(s, x,     y, z - 1) : 0
        v2 = valid_boid(s, x - 1, y, z - 1); b2_idx = v2 ? coord_to_boid(s, x - 1, y, z - 1) : 0
        v3 = valid_boid(s, x - 1, y, z);     b3_idx = v3 ? coord_to_boid(s, x - 1, y, z) : 0
        v4 = valid_boid(s, x,     y, z);     b4_idx = v4 ? coord_to_boid(s, x,     y, z) : 0
    else # align == Z_ALIGN
        # Boids around a Z-aligned edge (x,y plane)
        v1 = valid_boid(s, x - 1, y - 1, z); b1_idx = v1 ? coord_to_boid(s, x - 1, y - 1, z) : 0
        v2 = valid_boid(s, x,     y - 1, z); b2_idx = v2 ? coord_to_boid(s, x,     y - 1, z) : 0
        v3 = valid_boid(s, x,     y,     z); b3_idx = v3 ? coord_to_boid(s, x,     y,     z) : 0
        v4 = valid_boid(s, x - 1, y,     z); b4_idx = v4 ? coord_to_boid(s, x - 1, y,     z) : 0
    end

    indices = (b1_idx, b2_idx, b3_idx, b4_idx)
    validity = (v1, v2, v3, v4)
    return (indices, validity)
end

# TODO: Check this code to make sure it is working as intended
"""
    vertex_edges(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int)

Return the indices of the six primal edges incident to the specified primal vertex.
Assumes the vertex is in the valid interior of the mesh (no boundary checks).

Ordering rules:
1. Z-aligned (low, high): Z-edge at z-1, then Z-edge at z.
2. Y-aligned (south, north): Y-edge at y-1, then Y-edge at y.
3. X-aligned (west, east): X-edge at x-1 (West), then X-edge at x (East).
"""
function vertex_edges(s::UniformCubicalComplex3D, x::Int, y::Int, z::Int)
    v_low = valid_zedge(s, x, y, z - 1); e_low = v_low ? coord_to_edge(s, x, y, z - 1, Z_ALIGN) : 0
    v_high = valid_zedge(s, x, y, z); e_high = v_high ? coord_to_edge(s, x, y, z, Z_ALIGN) : 0
    v_south = valid_yedge(s, x, y - 1, z); e_south = v_south ? coord_to_edge(s, x, y - 1, z, Y_ALIGN) : 0
    v_north = valid_yedge(s, x, y, z); e_north = v_north ? coord_to_edge(s, x, y, z, Y_ALIGN) : 0
    v_west = valid_xedge(s, x - 1, y, z); e_west = v_west ? coord_to_edge(s, x - 1, y, z, X_ALIGN) : 0
    v_east = valid_xedge(s, x, y, z); e_east = v_east ? coord_to_edge(s, x, y, z, X_ALIGN) : 0

    indices = (e_low, e_high, e_south, e_north, e_west, e_east)
    validity = (v_low, v_high, v_south, v_north, v_west, v_east)
    return (indices, validity)
end

function primal_boundary_vertices(s::AbstractCubicalComplex3D, side::GridSide)
    if side == EASTWEST
        west = [coord_to_vert(s, 1, y, z) for y in 1:ny(s), z in 1:nz(s)][:]
        east = [coord_to_vert(s, nx(s), y, z) for y in 1:ny(s), z in 1:nz(s)][:]
        return (west, east)
    elseif side == NORTHSOUTH
        south = [coord_to_vert(s, x, 1, z) for x in 1:nx(s), z in 1:nz(s)][:]
        north = [coord_to_vert(s, x, ny(s), z) for x in 1:nx(s), z in 1:nz(s)][:]
        return (south, north)
    else # side == UPDOWN
        down = [coord_to_vert(s, x, y, 1) for x in 1:nx(s), y in 1:ny(s)][:]
        up = [coord_to_vert(s, x, y, nz(s)) for x in 1:nx(s), y in 1:ny(s)][:]
        return (down, up)
    end
  end
  
  function primal_boundary_quads(s::AbstractCubicalComplex3D, side::GridSide)
    if side == EASTWEST
        west = [coord_to_quad(s, 1, y, z, X_ALIGN) for y in 1:nyb(s), z in 1:nzb(s)][:]
        east = [coord_to_quad(s, nx(s), y, z, X_ALIGN) for y in 1:nyb(s), z in 1:nzb(s)][:]
        return (east, west)
    elseif side == NORTHSOUTH
        south = [coord_to_quad(s, x, 1, z, Y_ALIGN) for x in 1:nxb(s), z in 1:nzb(s)][:]
        north = [coord_to_quad(s, x, ny(s), z, Y_ALIGN) for x in 1:nxb(s), z in 1:nzb(s)][:]
        return (north, south)
    else # side == UPDOWN
        down = [coord_to_quad(s, x, y, 1, Z_ALIGN) for x in 1:nxb(s), y in 1:nyb(s)][:]
        up = [coord_to_quad(s, x, y, nz(s), Z_ALIGN) for x in 1:nxb(s), y in 1:nyb(s)][:]
        return (up, down)
    end
  end
  
  function primal_boundary_boids(s::AbstractCubicalComplex3D, side::GridSide)
    if side == EASTWEST
        west = [coord_to_boid(s, 1, y, z) for y in 1:nyb(s), z in 1:nzb(s)][:]
        east = [coord_to_boid(s, nxb(s), y, z) for y in 1:nyb(s), z in 1:nzb(s)][:]
        return (west, east)
    elseif side == NORTHSOUTH
        south = [coord_to_boid(s, x, 1, z) for x in 1:nxb(s), z in 1:nzb(s)][:]
        north = [coord_to_boid(s, x, nyb(s), z) for x in 1:nxb(s), z in 1:nzb(s)][:]
        return (south, north)
    else # side == UPDOWN
        down = [coord_to_boid(s, x, y, 1) for x in 1:nxb(s), y in 1:nyb(s)][:]
        up = [coord_to_boid(s, x, y, nzb(s)) for x in 1:nxb(s), y in 1:nyb(s)][:]
        return (down, up)
    end
  end
  
  # ── Ghost Boid Region Extraction ──────────────────────────────────────────────
#
# For a given side (EASTWEST, NORTHSOUTH, UPDOWN) and role, returns the flat
# vector of boid indices in that ghost/send slab.
#
# role:
#   :recv_low  – halo layer on the LOW  side (written by incoming MPI data)
#   :send_low  – real layer to send TO  the LOW  neighbour
#   :send_high – real layer to send TO  the HIGH neighbour
#   :recv_high – halo layer on the HIGH side (written by incoming MPI data)

function _boid_axis_info(s::AbstractCubicalComplex3D, side::GridSide)
    if side == EASTWEST
        return (nxb(s), nyb(s), nzb(s), hx(s),
                (ax, ay, az) -> coord_to_boid(s, ax, ay, az))
    elseif side == NORTHSOUTH
        return (nyb(s), nxb(s), nzb(s), hy(s),
                (ax, ay, az) -> coord_to_boid(s, ay, ax, az))
    else # UPDOWN
        return (nzb(s), nxb(s), nyb(s), hz(s),
                (ax, ay, az) -> coord_to_boid(s, ay, az, ax))
    end
end

function ghost_boids(s::AbstractCubicalComplex3D, side::GridSide, role::Symbol)
    # n_ax  : extent of the sliced axis
    # n_b   : extent of the first transverse axis
    # n_c   : extent of the second transverse axis
    # h     : halo depth on this axis
    # to_idx: coord_to_boid with axes permuted so ax is always the sliced one
    n_ax, n_b, n_c, h, to_idx = _boid_axis_info(s, side)

    h == 0 && return Int[]

    ax_range = if role == :recv_low
        1:h
    elseif role == :send_low
        (h + 1):(2h)
    elseif role == :send_high
        (n_ax - 2h + 1):(n_ax - h)
    elseif role == :recv_high
        (n_ax - h + 1):n_ax
    else
        error("Unknown ghost role $(repr(role)). " *
              "Valid roles: :recv_low, :send_low, :send_high, :recv_high")
    end

    return [to_idx(ax, b, c) for ax in ax_range, b in 1:n_b, c in 1:n_c][:]
end

function interior(::Val{3}, f::AbstractVector, s::AbstractCubicalComplex3D)
    indices = [coord_to_boid(s, x, y, z)
               for z in (hz(s)+1):(hz(s)+nzr(s)),
                   y in (hy(s)+1):(hy(s)+nyr(s)),
                   x in (hx(s)+1):(hx(s)+nxr(s))][:]
    return f[indices]
end