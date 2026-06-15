using MPI

include("UniformMesh.jl")
include("UniformMesh3D.jl")

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
    local_mesh_sizes :: NTuple{N, Int}
    local_mesh_offsets :: NTuple{N, Int}
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
    worker_meshdims    :: NTuple{N, Vector{Int}}   # per-worker mesh point count per axis
    worker_meshoffsets   :: NTuple{N, Vector{Int}}   # per-worker global offset per axis
    output_mesh_dims :: NTuple{N, Int}
end

output(topo::AbstractMPITopology) = topo.is_output
worker(topo::AbstractMPITopology) = !output(topo)

output_leader(topo::AbstractMPITopology) = output(topo) && topo.cart_rank == 0
worker_leader(topo::AbstractMPITopology) = worker(topo) && topo.cart_rank == 0

function build_worker_output_topology(mesh_dims::NTuple{N, Int},
                                      worker_dims::NTuple{N, Int},
                                      output_dims::NTuple{N, Int};
                                      periods::NTuple{N, Bool} = ntuple(_ -> false, N)) where N

    
    world_comm, cart_comm, is_output = _build_comms(worker_dims, output_dims, periods)
    intercomm, base_tiles, rems = _build_intercomm(world_comm, cart_comm, is_output, worker_dims, output_dims, N)

    cart_rank = MPI.Comm_rank(cart_comm)
    cart_nranks = MPI.Comm_size(cart_comm)
    cart_coords = MPI.Cart_coords(cart_comm)

    size_ref = Ref{Cint}(0)
    MPI.API.MPI_Comm_remote_size(intercomm, size_ref)
    remote_size = Int(size_ref[])
    
    if is_output
        worker_meshdims, worker_meshoffsets = _recv_mesh_metadata(intercomm, remote_size, N)

        local_tile_dims = ntuple(i -> tile_size(cart_coords[i], base_tiles[i], rems[i]), N)
        local_tile_origins = ntuple(i -> tile_offset(cart_coords[i], base_tiles[i], rems[i]), N)
        ranges = ntuple(i -> data_range(local_tile_origins[i], local_tile_dims[i]), N)

        # TODO: Check that this logic works
        # Build a reference Cart comm over the local tile to convert coords to linear index
        tile_cart = MPI.Cart_create(MPI.COMM_SELF, collect(Cint, local_tile_dims),
        collect(Cint, ntuple(_ -> false, N)), false)

        # For each axis, sum worker_meshdims along all coords in that axis
        output_mesh_dims = ntuple(N) do i
            sum(0:local_tile_dims[i]-1) do coord
                # Build a representative coord tuple with this axis varying, others fixed at 0
                coords = ntuple(j -> j == i ? coord : 0, N)
                idx = MPI.Cart_rank(tile_cart, coords) + 1   # 1-indexed
                topo.worker_meshdims[i][idx]
            end
        end
    
        return OutputTopology{N}(
            world_comm, cart_comm, intercomm,
            true,
            cart_rank, cart_nranks,
            ranges, local_tile_dims,
            worker_meshdims, worker_meshoffsets,
            output_mesh_dims,
        )
    else
        mesh_quad_dims = mesh_dims .- 1
        local_mesh_sizes, local_mesh_offsets = _send_mesh_metadata(intercomm, cart_coords, mesh_quad_dims, worker_dims, N)

        neighbors = build_neighbors(cart_comm, Val(N))
    
        return WorkerTopology{N}(
            world_comm, cart_comm, intercomm,
            false,
            cart_rank, cart_nranks,
            neighbors,
            local_mesh_sizes, local_mesh_offsets,
        )
    end
end

function _build_comms(worker_dims::NTuple{N, Int}, output_dims::NTuple{N, Int}, periods::NTuple{N, Bool}) where N

    MPI.Init()

    world_comm = MPI.COMM_WORLD
    world_rank = MPI.Comm_rank(world_comm)
    world_size = MPI.Comm_size(world_comm)

    nworkers = prod(worker_dims)
    noutput  = prod(output_dims)

    @assert nworkers > 0 "Need at least 1 worker process"
    @assert noutput  > 0 "Need at least 1 output process"
    @assert world_size == nworkers + noutput "world_size must equal nworkers + noutput"

    is_output  = world_rank >= nworkers
    group_comm = MPI.Comm_split(world_comm, is_output ? 1 : 0, world_rank)

    dims      = is_output ? output_dims : worker_dims
    cart_comm = MPI.Cart_create(group_comm, collect(Cint, dims),
                                collect(Cint, periods), false)

    MPI.Barrier(world_comm)

    return (world_comm, cart_comm, is_output)
end

function _build_intercomm(world_comm, cart_comm, is_output, w_dims, o_dims, N)

    world_rank  = MPI.Comm_rank(world_comm)
    cart_coords = MPI.Cart_coords(cart_comm)

    base_tiles = ntuple(i -> w_dims[i] ÷ o_dims[i], N)
    rems       = ntuple(i -> w_dims[i] % o_dims[i], N)

    cart_rank  = MPI.Comm_rank(cart_comm)

    pair_color = if is_output
        cart_rank
    else
        o_coords = ntuple(i -> owning_output_coord(cart_coords[i], base_tiles[i], rems[i], o_dims[i]), N)
        foldl((acc, (oc, od)) -> acc * od + oc, zip(o_coords, o_dims); init = 0)
    end

    pair_comm      = MPI.Comm_split(world_comm, pair_color, world_rank)
    pair_size      = MPI.Comm_size(pair_comm)
    pair_side_comm = MPI.Comm_split(pair_comm, is_output ? 1 : 0, world_rank)

    output_pair_rank = pair_size - 1
    local_leader     = 0
    remote_leader    = is_output ? 0 : output_pair_rank

    intercomm_ref = Ref{MPI.MPI_Comm}()
    MPI.API.MPI_Intercomm_create(
        pair_side_comm, local_leader,
        pair_comm,      remote_leader,
        pair_color,
        intercomm_ref
    )
    intercomm = MPI.Comm(intercomm_ref[])

    MPI.Barrier(world_comm)

    return (intercomm, base_tiles, rems)
end

# Worker generates local mesh sizes can communicates to partner output
function _send_mesh_metadata(intercomm::MPI.Comm, cart_coords, mesh_quad_dims, worker_dims, N)
    local_mesh_sizes   = ntuple(i -> tile_size(cart_coords[i],
                                               mesh_quad_dims[i] ÷ worker_dims[i],
                                               mesh_quad_dims[i] % worker_dims[i]), N) .+ 1
    local_mesh_offsets = ntuple(i -> tile_offset(cart_coords[i],
                                                  mesh_quad_dims[i] ÷ worker_dims[i],
                                                  mesh_quad_dims[i] % worker_dims[i]), N)

    MPI.API.MPI_Gather(
        collect(Int32, local_mesh_sizes),   Cint(N), MPI.Datatype(Int32),
        C_NULL,                             Cint(0), MPI.Datatype(Int32),
        Cint(0), intercomm
    )
    MPI.API.MPI_Gather(
        collect(Int32, local_mesh_offsets), Cint(N), MPI.Datatype(Int32),
        C_NULL,                             Cint(0), MPI.Datatype(Int32),
        Cint(0), intercomm
    )

    return local_mesh_sizes, local_mesh_offsets
end

# Output waits for workers to report local mesh sizes
function _recv_mesh_metadata(intercomm::MPI.Comm, remote_size::Int, N)
    mpi_root = Ref{Cint}(MPI.API.MPI_ROOT[])

    sizes_flat   = zeros(Int32, remote_size * N)
    offsets_flat = zeros(Int32, remote_size * N)

    MPI.API.MPI_Gather(
        C_NULL, Cint(0), MPI.Datatype(Int32),
        sizes_flat,   Cint(N), MPI.Datatype(Int32),
        mpi_root[], intercomm
    )
    MPI.API.MPI_Gather(
        C_NULL, Cint(0), MPI.Datatype(Int32),
        offsets_flat, Cint(N), MPI.Datatype(Int32),
        mpi_root[], intercomm
    )

    worker_meshdims  = ntuple(i -> Int.(sizes_flat[i:N:end]),   N)
    worker_meshoffsets = ntuple(i -> Int.(offsets_flat[i:N:end]), N)

    return worker_meshdims, worker_meshoffsets
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