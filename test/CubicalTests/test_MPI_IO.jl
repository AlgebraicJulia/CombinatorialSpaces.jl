using MPI
using HDF5

include("../../src/CubicalCode/UniformMPI.jl")
include("../../src/CubicalCode/UniformIO.jl")

MPI.Init()

# ── Global configuration ───────────────────────────────────────────────────────
const m_dims = (33, 33)       # Global mesh in vertices
const w_dims = (1, 1)         # Worker process grid
const o_dims = (1, 1)         # Output process grid
const FILEPATH = "test_output.h5"

# ── DataStream: single Quad{2} datum ──────────────────────────────────────────
const stream = DataStream(
    "test",
    FILEPATH,
    [Datum{Quad, 2}("quads", "fields", Float64)]
)

# ── Build topology ─────────────────────────────────────────────────────────────
topo = build_worker_output_topology(m_dims, w_dims, o_dims)

# ── Worker branch ──────────────────────────────────────────────────────────────
if worker(topo)
    cache = topo.cache  # WorkerCache{2}

    # Generate synthetic data: linear index over local quad mesh
    # Quads are (lm_dims .- 1) in each dimension
    quad_dims = cache.lm_dims .- 1
    nquads_    = prod(quad_dims)

    # Fill with global quad index so we can verify placement in the output
    data = Float64[
        (cache.lm_gm_offsets[1] + ((q-1) % quad_dims[1])) * (m_dims[2] - 1) +
        (cache.lm_gm_offsets[2] + ((q-1) ÷ quad_dims[1])) + 1.0
        for q in 1:nquads_
    ]

    # Participate in each Gatherv! over the intercomm
    for datum in stream.data
        gather!(data, datum, topo)
    end

# ── Output branch ──────────────────────────────────────────────────────────────
else
    cache   = topo.cache  # OutputCache{2}
    handler = DataHandler(stream, topo)

    # Output rank 0 creates the HDF5 file collectively
    create_hdf5!(handler, m_dims)

    MPI.Barrier(topo.cart_comm)

    # All output processes write their tile at time step 0
    write_output!(handler, 0)

    MPI.Barrier(topo.cart_comm)

    # Output rank 0 opens the file and prints the result
    if output_leader(topo)
        h5open(FILEPATH, "r") do h5
            data = h5["fields/quads"][1, :, :]
            println("── Quad data at time step 0 ──────────────────────")
            println("Size: ", size(data))
            println("Min:  ", minimum(data))
            println("Max:  ", maximum(data))
            println("Data:")
            show(stdout, "text/plain", data)
            println()
        end
    end
end

MPI.Finalize()