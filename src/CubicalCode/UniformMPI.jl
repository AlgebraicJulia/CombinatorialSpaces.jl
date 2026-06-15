using MPI

# ── Abstract base ──────────────────────────────────────────────────────────────

abstract type AbstractMPITopology{N} end

# ── Concrete types ─────────────────────────────────────────────────────────────

struct WorkerTopology{N} <: AbstractMPITopology{N}
    world_comm  :: MPI.Comm
    cart_comm   :: MPI.Comm
    intercomm   :: MPI.Comm
    is_output   :: Bool
    cart_rank   :: Int
    cart_nranks :: Int
    neighbors   :: NamedTuple
end

struct OutputTopology{N} <: AbstractMPITopology{N}
    world_comm      :: MPI.Comm
    cart_comm       :: MPI.Comm
    intercomm       :: MPI.Comm
    is_output       :: Bool
    cart_rank       :: Int
    cart_nranks     :: Int
    data_ranges     :: NTuple{N, UnitRange{Int}}
    local_tile_dims :: NTuple{N, Int}
end

output(topo::AbstractMPITopology) = topo.is_output
worker(topo::AbstractMPITopology) = !output(topo)

output_leader(topo::AbstractMPITopology) = output(topo) && topo.cart_rank == 0
worker_leader(topo::AbstractMPITopology) = worker(topo) && topo.cart_rank == 0

function build_worker_output_topology(worker_dims::NTuple{N, Int},
                                       output_dims::NTuple{N, Int};
                                       periods::NTuple{N, Bool} = ntuple(_ -> false, N)) where N

    
    world_comm = MPI.COMM_WORLD
    world_rank = MPI.Comm_rank(world_comm)
    world_size = MPI.Comm_size(world_comm)

    nworkers = prod(worker_dims)
    noutput  = prod(output_dims)

    @assert nworkers > 0 "Need at least 1 worker process"
    @assert noutput  > 0 "Need at least 1 output process"
    @assert world_size == nworkers + noutput "world_size must equal nworkers + noutput"

    # ── Split into worker / output groups ─────────────────────────────────────
    is_output = world_rank >= nworkers
    color     = is_output ? 1 : 0

    group_comm = MPI.Comm_split(world_comm, color, world_rank)

    worker_leader = 0
    output_leader = nworkers

    # ── Cartesian topology within each group ──────────────────────────────────
    dims      = is_output ? output_dims : worker_dims
    cart_comm = MPI.Cart_create(group_comm, collect(Cint, dims),
                                collect(Cint, periods), false)
    cart_rank   = MPI.Comm_rank(cart_comm)
    cart_nranks = MPI.Comm_size(cart_comm)
    cart_coords = MPI.Cart_coords(cart_comm)  # 0-indexed, length N

    MPI.Barrier(world_comm)

    # ── Broadcast both sets of dims to all processes ───────────────────────────
    worker_dims = Vector{Cint}(is_output ? zeros(Int, N) : collect(dims))
    output_dims = Vector{Cint}(is_output ? collect(dims) : zeros(Int, N))

    MPI.Bcast!(worker_dims, worker_leader, world_comm)
    MPI.Bcast!(output_dims, output_leader, world_comm)

    # ── Per-axis tiling arithmetic ─────────────────────────────────────────────
    base_tiles = ntuple(i -> worker_dims[i] ÷ output_dims[i], N)
    rems       = ntuple(i -> worker_dims[i] % output_dims[i], N)
    
    # pair_color is simply the local output rank. Workers need to derive this
    pair_color = if is_output
        cart_rank
    else
        # Map worker cart coord to output cart coord
        o_coords = ntuple(i -> owning_output_coord(cart_coords[i], base_tiles[i], rems[i], output_dims[i]), N)

        # TODO: May just want to use MPI for this but we don't have output's cart_comm here
        # 2D: oy + ox * Oy  3D: ox * Oy * Oz + oy * Oz + oz 
        o_coord_to_idx(o_coords, output_dims)
    end

    # ── Sub-communicators per pair ─────────────────────────────────────────────
    pair_comm      = MPI.Comm_split(world_comm, pair_color, world_rank)
    pair_size      = MPI.Comm_size(pair_comm)

    pair_side_color = is_output ? 1 : 0
    pair_side_comm  = MPI.Comm_split(pair_comm, pair_side_color, world_rank)

    # ── Intercomm ──────────────────────────────────────────────────────────────
    # Output process is highest world_rank in the pair → highest pair_rank
    output_pair_rank = pair_size - 1
    local_leader     = 0
    remote_leader    = is_output ? 0 : output_pair_rank

    intercomm_ref = Ref{MPI.MPI_Comm}()
    MPI.API.MPI_Intercomm_create(
        pair_side_comm, local_leader,
        pair_comm,      remote_leader,
        pair_color,     # unique tag per output process
        intercomm_ref
    )
    intercomm = MPI.Comm(intercomm_ref[])

    MPI.Barrier(world_comm)

    # ── Construct and return the appropriate topology type ────────────────────
    if is_output
        local_tile_dims = ntuple(i -> tile_size(cart_coords[i], base_tiles[i], rems[i]), N)
        local_tile_origins = ntuple(i -> tile_offset(cart_coords[i], base_tiles[i], rems[i]), N)
        ranges = ntuple(i -> data_range(local_tile_origins[i], local_tile_dims[i]), N)

        return OutputTopology{N}(
            world_comm, cart_comm, intercomm,
            true,
            cart_rank, cart_nranks,
            ranges, local_tile_dims
        )
    else
        neighbors = build_neighbors(cart_comm, N)

        return WorkerTopology{N}(
            world_comm, cart_comm, intercomm,
            false,
            cart_rank, cart_nranks,
            neighbors
        )
    end
end
 
# ── Worker helpers ────────────────────────────────────────────────────────────

const AXIS_NAMES_2D = (:west, :east, :south, :north)
const AXIS_NAMES_3D = (:west, :east, :south, :north, :down, :up)

function build_neighbors(cart_comm::MPI.Comm, ::Val{2})
    west,  east  = MPI.Cart_shift(cart_comm, 0, 1)
    south, north = MPI.Cart_shift(cart_comm, 1, 1)
    return (west=west, east=east, south=south, north=north)
end

function build_neighbors(cart_comm::MPI.Comm, ::Val{3})
    west,  east  = MPI.Cart_shift(cart_comm, 0, 1)
    south, north = MPI.Cart_shift(cart_comm, 1, 1)
    down,  up    = MPI.Cart_shift(cart_comm, 2, 1)
    return (west=west, east=east, south=south, north=north, down=down, up=up)
end

build_neighbors(cart_comm::MPI.Comm, N::Int) = build_neighbors(cart_comm, Val(N))

function exchange_quads!(f::AbstractVector, ghosts::NamedTuple,
                         topo::WorkerTopology{2}, side::GridSide)
    if side == EASTWEST
        low, high = :west, :east
    elseif side == NORTHSOUTH
        low, high = :south, :north
    else
        error("exchange_quads!: GridSide $(repr(side)) is not valid for a 2D mesh. " *
              "Valid sides: EASTWEST, NORTHSOUTH")
    end

    send_low  = f[ghosts[low].send]
    send_high = f[ghosts[high].send]
    recv_low  = similar(send_low)
    recv_high = similar(send_high)

    MPI.Sendrecv!(send_low,  topo.neighbors[low],  0,
                  recv_high, topo.neighbors[high], 0,
                  topo.cart_comm)
    MPI.Sendrecv!(send_high, topo.neighbors[high], 1,
                  recv_low,  topo.neighbors[low],  1,
                  topo.cart_comm)

    f[ghosts[low].recv]  .= recv_low
    f[ghosts[high].recv] .= recv_high

    return nothing
end

function exchange_quads_all!(f::AbstractVector, ghosts::NamedTuple, topo::WorkerTopology{2})
    exchange_quads!(f, ghosts, topo, EASTWEST)
    exchange_quads!(f, ghosts, topo, NORTHSOUTH)
end

# ── Output helpers ────────────────────────────────────────────────────────────

# Computes the actual worker tile size at that Cartesian coordinate (any dimension)
tile_size(rank_coord, base_tile, rem) = base_tile + (rank_coord < rem ? 1 : 0)

# Comptutes the worker offset 
tile_offset(rank_coord, base_tile, rem) = rank_coord * base_tile + min(rank_coord, rem)

# Finds worker's output partner by simulation
function owning_output_coord(w_coord, base_tile, rem, n_output)
    for o in 0:n_output-1
        if w_coord < tile_offset(o + 1, base_tile, rem)
            return o
        end
    end
    return n_output - 1
end

o_coord_to_idx(o_coords, output_dims) = foldl((acc, (oc, od)) -> acc * od + oc, zip(o_coords, output_dims); init = 0)

data_range(local_tile_origin, local_tile_dim) = (local_tile_origin + 1):(local_tile_origin + local_tile_dim)