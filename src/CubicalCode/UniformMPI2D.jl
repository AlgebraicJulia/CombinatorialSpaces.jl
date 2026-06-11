using MPI

struct MPITopology{N}
    comm      :: MPI.Comm
    rank      :: Int
    nranks    :: Int
    neighbors :: NamedTuple
end

function MPITopology{2}(comm::MPI.Comm;
                        periodic :: NTuple{2, Bool} = (false, false),
                        dims     :: NTuple{2, Int}  = (0, 0))
    rank   = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    cdims   = Cint[dims[1], dims[2]]
    MPI.Dims_create!(nranks, cdims)

    periods   = Cint[periodic[1], periodic[2]]
    cart_comm = MPI.Cart_create(comm, cdims, periods, false)

    west,  east  = MPI.Cart_shift(cart_comm, 0, 1)
    south, north = MPI.Cart_shift(cart_comm, 1, 1)

    return MPITopology{2}(cart_comm, rank, nranks,
                          (west=west, east=east, south=south, north=north))
end

const SIDE_TO_NEIGHBORS_2D = Dict(
    EASTWEST   => (:west,  :east),
    NORTHSOUTH => (:south, :north),
)

function exchange_quads!(f::AbstractVector, ghosts::NamedTuple,
                         topo::MPITopology{2}, side::GridSide)
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
                  topo.comm)
    MPI.Sendrecv!(send_high, topo.neighbors[high], 1,
                  recv_low,  topo.neighbors[low],  1,
                  topo.comm)

    f[ghosts[low].recv]  .= recv_low
    f[ghosts[high].recv] .= recv_high

    return nothing
end

function exchange_quads_all!(f::AbstractVector, ghosts::NamedTuple, topo::MPITopology{2})
    exchange_quads!(f, ghosts, topo, EASTWEST)
    exchange_quads!(f, ghosts, topo, NORTHSOUTH)
end