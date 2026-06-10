using MPI

struct MPITopology
    comm      :: MPI.Comm
    rank      :: Int
    nranks    :: Int
    neighbors :: NamedTuple{(:west, :east, :south, :north, :down, :up), NTuple{6, Int}}
end

function MPITopology(comm::MPI.Comm)
    rank   = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    dims   = Cint[0, 0, 0]
    MPI.Dims_create!(nranks, dims)
    periods   = Cint[0, 0, 0]
    cart_comm = MPI.Cart_create(comm, dims, periods, false)
    coords    = MPI.Cart_coords(cart_comm)
    
    neighbor(dim, disp) = MPI.Cart_shift(cart_comm, dim, disp)

    west,  east  = neighbor(0, 1)
    south, north = neighbor(1, 1)
    down,  up    = neighbor(2, 1)

    MPITopology(cart_comm, rank, nranks,
                (west=west, east=east, south=south, north=north, down=down, up=up))
end

const SIDE_TO_AXIS = Dict(
    EASTWEST   => 0,
    NORTHSOUTH => 1,
    UPDOWN     => 2,
)

const SIDE_TO_NEIGHBORS = Dict(
    EASTWEST   => (:west,  :east),
    NORTHSOUTH => (:south, :north),
    UPDOWN     => (:down,  :up),
)

function exchange_boids!(f::AbstractVector, s::AbstractCubicalComplex3D,
                         topo::MPITopology, side::GridSide)
    low_nbr, high_nbr = SIDE_TO_NEIGHBORS[side]
    rank_low  = topo.neighbors[low_nbr]
    rank_high = topo.neighbors[high_nbr]

    send_low  = f[ghost_boids(s, side, :send_low)]
    send_high = f[ghost_boids(s, side, :send_high)]
    recv_low  = similar(send_low)
    recv_high = similar(send_high)

    # Send to low neighbor, receive from high neighbor
    MPI.Sendrecv!(send_low,  rank_low,  0,
                  recv_high, rank_high, 0,
                  topo.comm)

    # Send to high neighbor, receive from low neighbor
    MPI.Sendrecv!(send_high, rank_high, 1,
                  recv_low,  rank_low,  1,
                  topo.comm)

    f[ghost_boids(s, side, :recv_low)]  .= recv_low
    f[ghost_boids(s, side, :recv_high)] .= recv_high

    return nothing
end

function exchange_boids_all!(f::AbstractVector, s::AbstractCubicalComplex3D,
                              topo::MPITopology)
    for side in (EASTWEST, NORTHSOUTH, UPDOWN)
        exchange_boids!(f, s, topo, side)
    end
end