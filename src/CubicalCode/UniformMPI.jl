using MPI

include("UniformMesh.jl")
include("UniformMesh3D.jl")

# ── Concrete types ─────────────────────────────────────────────────────────────

### MPI_TOPOLOGY ###
struct MPITopology{C}
    world_comm::Union{MPI.Comm,Nothing}
    cart_comm::Union{MPI.Comm,Nothing}
    intercomm::Union{MPI.Comm,Nothing}
    cart_rank::Int
    is_output::Bool
    cache::C # Will be either WorkerCache or OutputCache
end

output(topo::MPITopology) = topo.is_output
worker(topo::MPITopology) = !output(topo)

output_leader(topo::MPITopology) = output(topo) && topo.cart_rank == 0
worker_leader(topo::MPITopology) = worker(topo) && topo.cart_rank == 0

### WORKER CACHE ###
struct WorkerCache{N}
    cart_nranks::Int
    neighbors::NamedTuple
    lm_dims::NTuple{N,Int}
    lm_gm_offsets::NTuple{N,Int}
    gm_dims::NTuple{N,Int}
end

function WorkerCache{N}(cart_comm::Union{MPI.Comm,Nothing}, cart_coords, m_dims, w_dims) where {N}
    lm_dims = ntuple(i -> worker_mesh_size(cart_coords[i], m_dims[i], w_dims[i]), N)
    lm_gm_offsets = ntuple(i -> worker_mesh_offset(cart_coords[i], m_dims[i], w_dims[i]), N)

    if isnothing(cart_comm)
        cart_nranks = -1
        neighbors = NamedTuple()
    else
        cart_nranks = MPI.Comm_size(cart_comm)
        neighbors = build_neighbors(cart_comm, Val(N))
    end

    return WorkerCache{N}(cart_nranks, neighbors, lm_dims, lm_gm_offsets, m_dims)
end

function worker_mesh_size(cart_coord::Int, m_dim::Int, w_dim::Int)
    m_dual_dim = m_dim - 1 # Quad for 2D, Boid for 3D
    base_m_dim = m_dual_dim ÷ w_dim
    rem = m_dual_dim % w_dim
    return tile_size(cart_coord, base_m_dim, rem) + 1
end

function worker_mesh_offset(cart_coord::Int, m_dim::Int, w_dim::Int)
    m_dual_dim = m_dim - 1 # Quad for 2D, Boid for 3D
    base_m_dim = m_dual_dim ÷ w_dim
    rem = m_dual_dim % w_dim
    return tile_offset(cart_coord, base_m_dim, rem)
end

function Base.show(io::IO, c::WorkerCache{N}) where {N}
    println(io, "WorkerCache{$N}")
    println(io, "  cart_nranks : $(c.cart_nranks)")
    println(io, "  lm_dims     : $(c.lm_dims)  ($(prod(c.lm_dims .- 1)) dual cells)")
    println(io, "  lm_gm_offsets: $(c.lm_gm_offsets)")
    println(io, "  gm_dims     : $(c.gm_dims)  ($(prod(c.gm_dims .- 1)) global dual cells)")
    return print(io, "  neighbors   : $(keys(c.neighbors))")
end

# ── Worker mesh construction ──────────────────────────────────────────────────

function worker_mesh(topo::MPITopology{WorkerCache{N}}, L::NTuple{N,<:Real}; halo::Union{Int,NTuple{N,Int}} = 1) where {N}
    cache = topo.cache
    gm_dual = ntuple(i -> cache.gm_dims[i] - 1, N)
    lm_dims = cache.lm_dims
    offsets = cache.lm_gm_offsets
    halos = halo isa Int ? ntuple(_ -> halo, N) : halo

    FT = eltype(L)
    lengths = ntuple(i -> FT(lm_dims[i] - 1) * L[i] / gm_dual[i], N)
    bases = ntuple(i -> FT(offsets[i]) * L[i] / gm_dual[i], N)

    return UniformCubicalComplex(lm_dims..., lengths...; _mesh_kwargs(Val(N), halos, bases)...)
end

function _mesh_kwargs(::Val{2}, halos, bases)
    return (halo_x = halos[1], halo_y = halos[2], base_x = bases[1], base_y = bases[2])
end

function _mesh_kwargs(::Val{3}, halos, bases)
    return (halo_x = halos[1], halo_y = halos[2], halo_z = halos[3], base_x = bases[1], base_y = bases[2], base_z = bases[3])
end

# TODO: Remove me
_build_ghosts(::Val{2}, s) = ghost_quads(s)
_build_ghosts(::Val{3}, s) = ghost_boids(s)

### OUTPUT CACHE ###
struct OutputWorkerCache{N}
    lm_dims::NTuple{N,Int}
    lm_gm_offsets::NTuple{N,Int}
    mesh::AbstractCubicalComplex
end

struct OutputCache{N}
    nworkers::Int
    worker_caches::Vector{OutputWorkerCache{N}}
    lm_om_offsets::Vector{NTuple{N,Int}} # Offset of worker mesh from output mesh origin

    lt_dims::NTuple{N,Int} # Number of workers in each dim
    om_dims::NTuple{N,Int} # Number of vertices in each dim
    om_gm_offsets::NTuple{N,Int} # Offset of mesh from global mesh origin in vertices
    om_mesh::AbstractCubicalComplex
end

function OutputCache{N}(intercomm::Union{MPI.Comm,Nothing}, cart_coords, base_tiles, rems, test_arr = nothing) where {N}
    lt_dims = ntuple(i -> tile_size(cart_coords[i], base_tiles[i], rems[i]), N)
    nworkers = prod(lt_dims)
    len = length(OutputWorkerCache{N})

    # For testing only
    flat = if isnothing(intercomm)
        @assert nworkers * len == length(test_arr) "Test_arr should be $nworkers x $len, or $(nworkers * len)"
        test_arr
    else
        gather_workercaches(intercomm, nworkers, len)
    end

    worker_caches = [deserialize(OutputWorkerCache{N}, flat[((w - 1) * len + 1):(w * len)]) for w in 1:nworkers]

    om_dims = output_mesh_dimensions(worker_caches, lt_dims)

    om_gm_offsets = output_mesh_offsets(worker_caches)
    lm_om_offsets = local_mesh_worker_offsets(worker_caches)

    return OutputCache{N}(nworkers, worker_caches, lm_om_offsets, lt_dims, om_dims, om_gm_offsets, PseudoCubicalMesh(om_dims...))
end

function gather_workercaches(intercomm::MPI.Comm, nworkers::Int, len::Int)
    mpi_root = Ref{Cint}(MPI.API.MPI_ROOT[])
    flat = zeros(Int32, nworkers * len)
    MPI.API.MPI_Gather(C_NULL, Cint(0), MPI.Datatype(Int32), flat, Cint(len), MPI.Datatype(Int32), mpi_root[], intercomm)

    return flat
end

# XXX: This is making assumptions about the order the worker_caches are stored
# Namely, it assumes MPI ordering of going Z -> Y -> X 
function output_mesh_dimensions(worker_caches::Vector{OutputWorkerCache{N}}, lt_dims::NTuple{N,Int}) where {N}
    lt_dims = reverse(lt_dims)
    reverse(ntuple(N) do i
        stride = prod(lt_dims[1:(i - 1)])
        dim_sum = sum(worker_caches[stride * coord + 1].lm_dims[N - i + 1] for coord in 0:(lt_dims[i] - 1))
        return dim_sum - (lt_dims[i] - 1)
    end)
end

function output_mesh_offsets(worker_caches::Vector{OutputWorkerCache{N}}) where {N}
    return first(worker_caches).lm_gm_offsets
end
function local_mesh_worker_offsets(worker_caches::Vector{OutputWorkerCache{N}}) where {N}
    om_gm_offsets = output_mesh_offsets(worker_caches)
    return map(cache -> cache.lm_gm_offsets .- om_gm_offsets, worker_caches)
end

function Base.show(io::IO, c::OutputCache{N}) where {N}
    println(io, "OutputCache{$N}")
    println(io, "  nworkers     : $(c.nworkers)")
    println(io, "  lt_dims      : $(c.lt_dims)  (worker grid per output)")
    println(io, "  om_dims      : $(c.om_dims)  ($(prod(c.om_dims .- 1)) dual cells)")
    println(io, "  om_gm_offsets: $(c.om_gm_offsets)")
    println(io, "  worker_caches:")
    for (i, wc) in enumerate(c.worker_caches)
        lo = c.lm_om_offsets[i]
        hi = lo .+ (wc.lm_dims .- 1)
        println(io, "    [$i] lm_dims=$(wc.lm_dims)  gm_offset=$(wc.lm_gm_offsets)  om_range=$(lo).→.$(hi)")
    end
    return print(io, "  om_gm_offsets cover: $(c.om_gm_offsets) → $(c.om_gm_offsets .+ c.om_dims .- 1)")
end

### OUTPUT WORKER CACHE
function OutputWorkerCache(lm_dims::NTuple{N,Int}, lm_gm_offsets::NTuple{N,Int}) where {N}
    return OutputWorkerCache{N}(lm_dims, lm_gm_offsets, PseudoCubicalMesh(lm_dims...))
end

OutputWorkerCache(cache::WorkerCache) = OutputWorkerCache(cache.lm_dims, cache.lm_gm_offsets)

Base.length(::Type{OutputWorkerCache{N}}) where {N} = 2N
Base.length(::OutputWorkerCache{N}) where {N} = 2N

function serialize(cache::OutputWorkerCache{N}) where {N}
    return Int32[cache.lm_dims..., cache.lm_gm_offsets...]
end

function deserialize(::Type{OutputWorkerCache{N}}, buf::AbstractVector{Int32}) where {N}
    sizes = ntuple(i -> Int(buf[i]), N)
    offsets = ntuple(i -> Int(buf[N + i]), N)
    return OutputWorkerCache{N}(sizes, offsets, PseudoCubicalMesh(sizes...))
end

function _send_mesh_metadata(intercomm::MPI.Comm, cache::WorkerCache{N}) where {N}
    buf = serialize(OutputWorkerCache(cache))

    return MPI.API.MPI_Gather(buf, Cint(length(OutputWorkerCache{N})), MPI.Datatype(Int32), C_NULL, Cint(0), MPI.Datatype(Int32), Cint(0), intercomm)
end

function Base.show(io::IO, c::OutputWorkerCache{N}) where {N}
    println(io, "OutputWorkerCache{$N}")
    println(io, "  lm_dims      : $(c.lm_dims)  ($(prod(c.lm_dims .- 1)) dual cells)")
    return print(io, "  lm_gm_offsets: $(c.lm_gm_offsets)")
end

### BUILD THE MPI_TOPOLOGY ###

# TODO: This could probably be cleaned up with a struct that stores all dimensions
# and computes base_sizes and rems for mesh/worker and output/worker combos
function MPITopology(m_dims::NTuple{N,Int}, w_dims::NTuple{N,Int}, o_dims::NTuple{N,Int}; periods::NTuple{N,Bool} = ntuple(_ -> false, N)) where {N}

    # TODO: Fix these, non-boolean?
    # @assert all(m_dims .> 0) "Mesh dimensions as given $m_dims are reading negative dimensions"
    # @assert all(w_dims .> 0) "Worker dimensions as given $w_dims are reading negative dimensions"
    # @assert all(o_dims .> 0) "Output dimensions as given $o_dims are reading negative dimensions"

    # @assert all(m_dims .=> w_dims) "Expecting larger mesh dimensions than worker dimensions"
    # @assert all(w_dims .=> o_dims) "Expecting larger worker dimensions than output dimensions"

    world_comm, cart_comm, is_output = _build_comms(w_dims, o_dims, periods)
    intercomm, base_tiles, rems = _build_intercomm(world_comm, cart_comm, is_output, w_dims, o_dims, N)

    cart_rank = MPI.Comm_rank(cart_comm)
    cart_coords = MPI.Cart_coords(cart_comm)

    if is_output
        cache = OutputCache{N}(intercomm, cart_coords, base_tiles, rems)
    else
        cache = WorkerCache{N}(cart_comm, cart_coords, m_dims, w_dims)
        _send_mesh_metadata(intercomm, cache)
    end

    return MPITopology(world_comm, cart_comm, intercomm, cart_rank, is_output, cache)
end

MPITopology(cache, is_output::Bool) = MPITopology(nothing, nothing, nothing, -1, is_output, cache)

function _build_comms(w_dims::NTuple{N,Int}, o_dims::NTuple{N,Int}, periods::NTuple{N,Bool}) where {N}
    world_comm = MPI.COMM_WORLD
    world_rank = MPI.Comm_rank(world_comm)
    world_size = MPI.Comm_size(world_comm)

    nworkers = prod(w_dims)
    noutput = prod(o_dims)

    @assert nworkers > 0 "Need at least 1 worker process"
    @assert noutput > 0 "Need at least 1 output process"
    @assert world_size == nworkers + noutput "world_size must equal nworkers + noutput"

    is_output = world_rank >= nworkers
    group_comm = MPI.Comm_split(world_comm, is_output ? 1 : 0, world_rank)

    dims = is_output ? o_dims : w_dims
    cart_comm = MPI.Cart_create(group_comm, collect(Cint, dims), collect(Cint, periods), false)

    MPI.Barrier(world_comm)

    return (world_comm, cart_comm, is_output)
end

function _build_intercomm(world_comm, cart_comm, is_output, w_dims, o_dims, N)
    world_rank = MPI.Comm_rank(world_comm)

    cart_rank = MPI.Comm_rank(cart_comm)
    cart_coords = MPI.Cart_coords(cart_comm)

    base_tiles = ntuple(i -> w_dims[i] ÷ o_dims[i], N)
    rems = ntuple(i -> w_dims[i] % o_dims[i], N)

    pair_color = _compute_pair_color(is_output, cart_rank, cart_coords, base_tiles, rems, o_dims, N)

    pair_comm = MPI.Comm_split(world_comm, pair_color, world_rank)
    pair_size = MPI.Comm_size(pair_comm)
    pair_side_comm = MPI.Comm_split(pair_comm, is_output ? 1 : 0, world_rank)

    output_pair_rank = pair_size - 1
    local_leader = 0
    remote_leader = is_output ? 0 : output_pair_rank

    intercomm_ref = Ref{MPI.MPI_Comm}()
    MPI.API.MPI_Intercomm_create(pair_side_comm, local_leader, pair_comm, remote_leader, pair_color, intercomm_ref)
    intercomm = MPI.Comm(intercomm_ref[])

    MPI.Barrier(world_comm)

    return (intercomm, base_tiles, rems)
end

function _compute_pair_color(is_output::Bool, cart_rank::Int, cart_coords, base_tiles, rems, o_dims, N)
    if is_output
        return cart_rank
    else
        o_coords = ntuple(i -> owning_output_coord(cart_coords[i], base_tiles[i], rems[i], o_dims[i]), N)
        return o_coord_to_idx(o_coords, o_dims)
    end
end

### WORKER HELPERS ### 

function build_neighbors(cart_comm::MPI.Comm, ::Val{2})
    west, east = MPI.Cart_shift(cart_comm, 0, 1)
    south, north = MPI.Cart_shift(cart_comm, 1, 1)
    return (west = west, east = east, south = south, north = north)
end

function build_neighbors(cart_comm::MPI.Comm, ::Val{3})
    west, east = MPI.Cart_shift(cart_comm, 0, 1)
    south, north = MPI.Cart_shift(cart_comm, 1, 1)
    down, up = MPI.Cart_shift(cart_comm, 2, 1)
    return (west = west, east = east, south = south, north = north, down = down, up = up)
end

build_neighbors(cart_comm::MPI.Comm, N::Int) = build_neighbors(cart_comm, Val(N))

# TODO: Remove me
# ── Halo Exchange ─────────────────────────────────────────────────────────────

function exchange_quads!(f::AbstractVector, ghosts::NamedTuple, topo::MPITopology{WorkerCache{N}}, side::GridSide) where {N}
    if side == EASTWEST
        low, high = :west, :east
    elseif side == NORTHSOUTH
        low, high = :south, :north
    else
        error("exchange_quads!: GridSide $(repr(side)) is not valid for a 2D mesh. " * "Valid sides: EASTWEST, NORTHSOUTH")
    end

    neighbors = topo.cache.neighbors

    send_low = f[ghosts[low].send]
    send_high = f[ghosts[high].send]
    recv_low = similar(send_low)
    recv_high = similar(send_high)

    MPI.Sendrecv!(send_low, neighbors[low], 0, recv_high, neighbors[high], 0, topo.cart_comm)
    MPI.Sendrecv!(send_high, neighbors[high], 1, recv_low, neighbors[low], 1, topo.cart_comm)

    f[ghosts[low].recv] .= recv_low
    f[ghosts[high].recv] .= recv_high

    return nothing
end

function exchange_quads!(f::AbstractVector, ghosts::NamedTuple, topo::MPITopology{<:OutputCache}, side::GridSide)
    return error("exchange_quads! called on output process")
end

function exchange_quads_all!(f::AbstractVector, ghosts::NamedTuple, topo::MPITopology{WorkerCache{N}}) where {N}
    exchange_quads!(f, ghosts, topo, EASTWEST)
    exchange_quads!(f, ghosts, topo, NORTHSOUTH)
    return nothing
end

function exchange_quads_all!(f::AbstractVector, ghosts::NamedTuple, topo::MPITopology{<:OutputCache})
    return error("exchange_quads_all! called on output process")
end

function exchange_boids!(f::AbstractVector, ghosts::NamedTuple, topo::MPITopology{WorkerCache{N}}, side::GridSide) where {N}
    if side == EASTWEST
        low, high = :west, :east
    elseif side == NORTHSOUTH
        low, high = :south, :north
    elseif side == UPDOWN
        low, high = :down, :up
    else
        error("exchange_boids!: GridSide $(repr(side)) is not valid. " * "Valid sides: EASTWEST, NORTHSOUTH, UPDOWN")
    end

    neighbors = topo.cache.neighbors

    send_low = f[ghosts[low].send]
    send_high = f[ghosts[high].send]
    recv_low = similar(send_low)
    recv_high = similar(send_high)

    MPI.Sendrecv!(send_low, neighbors[low], 0, recv_high, neighbors[high], 0, topo.cart_comm)
    MPI.Sendrecv!(send_high, neighbors[high], 1, recv_low, neighbors[low], 1, topo.cart_comm)

    f[ghosts[low].recv] .= recv_low
    f[ghosts[high].recv] .= recv_high

    return nothing
end

function exchange_boids!(f::AbstractVector, ghosts::NamedTuple, topo::MPITopology{<:OutputCache}, side::GridSide)
    return error("exchange_boids! called on output process")
end

function exchange_boids_all!(f::AbstractVector, ghosts::NamedTuple, topo::MPITopology{WorkerCache{N}}) where {N}
    exchange_boids!(f, ghosts, topo, EASTWEST)
    exchange_boids!(f, ghosts, topo, NORTHSOUTH)
    exchange_boids!(f, ghosts, topo, UPDOWN)
    return nothing
end

function exchange_boids_all!(f::AbstractVector, ghosts::NamedTuple, topo::MPITopology{<:OutputCache})
    return error("exchange_boids_all! called on output process")
end

# ── Output helpers ────────────────────────────────────────────────────────────

# Computes the actual worker tile size at that Cartesian coordinate (any dimension)
tile_size(rank_coord, base_tile, rem) = base_tile + (rank_coord < rem ? 1 : 0)

# Comptutes the worker offset 
tile_offset(rank_coord, base_tile, rem) = rank_coord * base_tile + min(rank_coord, rem)

# Finds worker's output partner by simulation
function owning_output_coord(w_coord, base_tile, rem, n_output)
    for o in 0:(n_output - 1)
        if w_coord < tile_offset(o + 1, base_tile, rem)
            return o
        end
    end
    return n_output - 1
end

function o_coord_to_idx(o_coords, o_dims)
    return foldl((acc, (oc, od)) -> acc * od + oc, zip(o_coords, o_dims); init = 0)
end

function data_range(local_tile_origin, local_tile_dim)
    return (local_tile_origin + 1):(local_tile_origin + local_tile_dim)
end