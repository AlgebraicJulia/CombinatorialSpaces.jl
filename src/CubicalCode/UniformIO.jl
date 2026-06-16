using MPI
using HDF5

# More types can be added later on
abstract type AbstractMeshType end
struct Vert <: AbstractMeshType end
struct Edge <: AbstractMeshType end
struct Quad <: AbstractMeshType end
struct Boid <: AbstractMeshType end

# This logically represents a piece of data (density, temperature, velocity, etc.)
struct Datum{M <: AbstractMeshType, N}
    name      :: String
    groupname :: String
    entrytype :: DataType
end

# This logically represnts a stream of data either meant to be read out or saved
struct DataStream
    tag::String
    filepath::String
    data::AbstractVector{Datum}
end

# ^ The two above are totally user created

mutable struct GathervCache{T}
    recv_buffer::Vector{T}
    recv_counts::Vector{Cint}
    displs::Vector{Cint}
end


function GathervCache(datum::Datum{Quad, 2}, cache::OutputCache{2})
    T = datum.entrytype

    recv_counts = Cint[
        (wc.lm_dims[1] - 1) * (wc.lm_dims[2] - 1)
        for wc in cache.worker_caches
    ]

    displs = Cint[0; cumsum(recv_counts)[1:end-1]]

    total = Int(sum(recv_counts))
    recv_buffer = Vector{T}(undef, total)

    return GathervCache{T}(recv_buffer, recv_counts, displs)
end

# This is meant has the actual data handler. When passed into a function, it'll
# handle the actual operation of reading/writing the data in the proper way
# It'll be in charge of all communication from worker to output
# There will be a init phase where the worker's will output data into the WorkerCache
# This'll be used to populate the OutputCache which will be used to be able to process the data correctly
# Normally, the DataHandler will busy wait until a command arrives from the workers
# This'll allow it to decide which action to take open/close/read/write
# For a write, it'll loop over the data in datastream and activate the buffers
# It'll do the gather and then pass the data into the buffer which will handle proper loading and arrangement
# The buffer, once the data is arranged, will allow the handler to simply write the data into the HDF5
# TODO: This could use the "tag" of the datastream to have a Dict of multiple datastreams

mutable struct DataHandler{N}
    stream        :: DataStream
    topo          :: MPITopology{OutputCache}
    gatherv_caches:: AbstractVector{GathervCache}
    tile_buffers  :: AbstractVector{AbstractArray} # Temp holding for data from Gatherv
    time_step     :: Int
end

function DataHandler(stream::DataStream, topo::MPITopology{OutputCache{N}}) where N
    cache = topo.cache

    gatherv_caches = map(stream.data) do datum
        T = datum.entrytype
        recv_counts = Cint[mesh_count(datum, wc.lm_dims) for wc in cache.worker_caches]
        displs = Cint[0; cumsum(recv_counts)[1:end-1]]
        recv_buffer = Vector{T}(undef, Int(sum(recv_counts)))
        GathervCache{T}(recv_buffer, recv_counts, displs)
    end

    tile_buffers = map(datum -> tile_buffer(datum, cache.om_dims), stream.data)

    return DataHandler{N}(stream, topo, gatherv_caches, tile_buffers, -Inf)
end

function create_hdf5!(handler::DataHandler{N}) where N
    h5open(handler.stream.filepath, "w", cart_comm, MPI.Info()) do h5

        for datum in handler.stream.data

            if !haskey(h5, datum.groupname)
                create_group(h5, datum.groupname)
            end
            grp = h5[datum.groupname]

            all_dims = global_datum_dims(datum, gm_dims)

            for (i, spatial_dims) in enumerate(all_dims)
                dims    = (1,              spatial_dims...)
                maxdims = (HDF5.UNLIMITED, spatial_dims...)
                chunk   = (1,              spatial_dims...)

                dset_name = length(all_dims) == 1 ? datum.name : datum.name * "_$i"

                HDF5.create_dataset(grp, dset_name, datum.entrytype,
                    HDF5.dataspace(dims, max_dims=maxdims);
                    chunk=chunk,
                    dxpl_mpio=:collective)
            end
        end
    end
end

function write_output!(handler::DataHandler{N}, time_step::Int) where N
    cache = handler.topo.cache

    # 1. Fire all Gatherv! operations as fast as possible
    gather!(handler)

    # 2. Scatter each recv_buffer into its structured tile_buffers
    for (datum, gcache, tbufs) in zip(handler.stream.data, handler.gatherv_caches, handler.tile_buffers)
        scatter_to_tile!(datum, gcache, tbufs, cache)
    end

    # 3. Open HDF5 file in collective mode and write all tile buffers
    h5open(handler.stream.filepath, "r+", topo.cart_comm, MPI.Info()) do h5
        for (datum, tbufs) in zip(handler.stream.data, handler.tile_buffers)
            write_tile!(h5, datum, tbufs, cache, time_step)
        end
    end

    handler.time_step = time_step # TODO: Deal with this time step better
end

function write_output!(data_arrays::Vector{<:AbstractVector}, stream::DataStream,
                       topo::MPITopology{WorkerCache{N}}) where N
    for (data, datum) in zip(data_arrays, stream.data)
        gather!(data, datum, topo)
    end
end

function gather!(handler::DataHandler{N}) where N
    mpi_root = Ref{Cint}(MPI.API.MPI_ROOT[])

    for (datum, gcache) in zip(handler.stream.data, handler.gatherv_caches)
        GC.@preserve gcache begin
            MPI.API.MPI_Gatherv(
                C_NULL, Cint(0), MPI.Datatype(datum.entrytype),
                gcache.recv_buffer, gcache.recv_counts, gcache.displs, MPI.Datatype(datum.entrytype),
                mpi_root[], handler.topo.intercomm
            )
        end
    end
end

function gather!(data::AbstractVector{T}, datum::Datum{M, N}, topo::MPITopology{WorkerCache{N}}) where {T, M, N}
    GC.@preserve data begin
        MPI.API.MPI_Gatherv(
            data, Cint(length(data)), MPI.Datatype(datum.entrytype),
            C_NULL, C_NULL, C_NULL,   MPI.Datatype(datum.entrytype),
            Cint(0), topo.intercomm
        )
    end
end

# Quad 2D: dual in both dims, single tile buffer
function scatter_to_tile!(datum::Datum{Quad, 2}, gcache::GathervCache, tbufs::Vector, cache::OutputCache{2})
    for (wc, wc_offset, src_start) in zip(cache.worker_caches, cache.lm_om_offsets,
                                           [0; cumsum([mesh_count(datum, wc.lm_dims) for wc in cache.worker_caches])[1:end-1]])
        dims   = wc.lm_dims .- 1
        n      = prod(dims)
        ranges = ntuple(i -> wc_offset[i]+1 : wc_offset[i]+dims[i], 2)
        tbufs[1][ranges...] .= reshape(gcache.recv_buffer[src_start+1 : src_start+n], dims)
    end
end

# Quad{2}, Boid: dual-sized, same vertex origin offset
function write_tile!(h5loc, datum::Union{Datum{Quad, 2}, Datum{Boid, N}},
                     tbufs::Vector, cache::OutputCache{N}, time_step::Int) where N
    offset = cache.om_gm_offsets
    count  = size(tbufs[1])
    ranges = (time_step:time_step, ntuple(j -> offset[j]+1 : offset[j]+count[j], N)...)
    h5loc[datum.groupname * "/" * datum.name][ranges...] = tbufs[1]
end

# TODO: This could probably use mesh functionality and be cleaner

# Vertices: vertex-sized in all dims
mesh_count(::Datum{Vert, N}, om_dims::NTuple{N, Int}) where N =
    prod(om_dims)

# Boids: dual in all dims
mesh_count(::Datum{Boid, N}, om_dims::NTuple{N, Int}) where N =
    prod(om_dims .- 1)

# Edges: dual in 1 dim, vertex in the rest → N families
mesh_count(::Datum{Edge, N}, om_dims::NTuple{N, Int}) where N =
    sum(1:N) do i
        prod(ntuple(j -> j == i ? om_dims[j] - 1 : om_dims[j], N))
    end

# Quads in 2D: dual in both dims
mesh_count(::Datum{Quad, 2}, om_dims::NTuple{2, Int}) =
    prod(om_dims .- 1)

# Quads in 3D: dual in 2 dims, vertex in 1 → 3 families
mesh_count(::Datum{Quad, 3}, om_dims::NTuple{3, Int}) =
    sum(1:3) do i
        prod(ntuple(j -> j == i ? om_dims[j] : om_dims[j] - 1, 3))
    end

# Vert: single N-D array of vertex-sized dims
tile_buffer(datum::Datum{Vert, N}, om_dims::NTuple{N, Int}) where N =
    [Array{datum.entrytype}(undef, om_dims...)]

# Boid: single N-D array of dual-sized dims
tile_buffer(datum::Datum{Boid, N}, om_dims::NTuple{N, Int}) where N =
    [Array{datum.entrytype}(undef, (om_dims .- 1)...)]

# Quad 2D: single dual-sized 2D array
tile_buffer(datum::Datum{Quad, 2}, om_dims::NTuple{2, Int}) =
    [Array{datum.entrytype}(undef, (om_dims .- 1)...)]

# Quad 3D: one array per axis-aligned face family (XY, XZ, YZ)
tile_buffer(datum::Datum{Quad, 3}, om_dims::NTuple{3, Int}) =
    [Array{datum.entrytype}(undef, ntuple(j -> j == i ? om_dims[j] : om_dims[j] - 1, 3)...)
     for i in 1:3]

# Edge 2D: one array per axis-aligned edge family
tile_buffer(datum::Datum{Edge, 2}, om_dims::NTuple{2, Int}) =
    [Array{datum.entrytype}(undef, ntuple(j -> j == i ? om_dims[j] - 1 : om_dims[j], 2)...)
     for i in 1:2]

# Edge 3D: one array per axis-aligned edge family
tile_buffer(datum::Datum{Edge, 3}, om_dims::NTuple{3, Int}) =
    [Array{datum.entrytype}(undef, ntuple(j -> j == i ? om_dims[j] - 1 : om_dims[j], 3)...)
     for i in 1:3]

# Returns a Vector of dimension tuples, one per family
function global_datum_dims(::Datum{Vert, N}, gm_dims::NTuple{N, Int}) where N
    [gm_dims]
end

function global_datum_dims(::Datum{Boid, N}, gm_dims::NTuple{N, Int}) where N
    [gm_dims .- 1]
end

function global_datum_dims(::Datum{Quad, 2}, gm_dims::NTuple{2, Int})
    [gm_dims .- 1]
end

function global_datum_dims(::Datum{Quad, 3}, gm_dims::NTuple{3, Int})
    [ntuple(j -> j == i ? gm_dims[j] : gm_dims[j] - 1, 3) for i in 1:3]
end

function global_datum_dims(::Datum{Edge, N}, gm_dims::NTuple{N, Int}) where N
    [ntuple(j -> j == i ? gm_dims[j] - 1 : gm_dims[j], N) for i in 1:N]
end