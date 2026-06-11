include(joinpath("@__DIR__", "UniformMPI2D.jl"))

function MPITopology{3}(comm::MPI.Comm;
                        periodic :: NTuple{3, Bool} = (false, false, false),
                        dims     :: NTuple{3, Int}  = (0, 0, 0))
    rank   = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    cdims   = Cint[dims[1], dims[2], dims[3]]
    MPI.Dims_create!(nranks, cdims)

    periods   = Cint[periodic[1], periodic[2], periodic[3]]
    cart_comm = MPI.Cart_create(comm, cdims, periods, false)

    west,  east  = MPI.Cart_shift(cart_comm, 0, 1)
    south, north = MPI.Cart_shift(cart_comm, 1, 1)
    down,  up    = MPI.Cart_shift(cart_comm, 2, 1)

    return MPITopology{3}(cart_comm, rank, nranks,
                          (west=west, east=east, south=south,
                           north=north, down=down, up=up))
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