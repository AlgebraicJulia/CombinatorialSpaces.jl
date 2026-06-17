using MPI
using HDF5

include("UniformMesh.jl")
include("UniformMesh3D.jl")
include("UniformMPI.jl")

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

Datum{M, N}() where {M <: AbstractMeshType, N} = Datum{M, N}("", "", Float64)

Datum{Boid, 2}(args...) = 
    error("Datum{Boid, 2} is invalid: Boid is a 3-cell and only exists in 3D. " * 
    "Did you mean Datum{Quad, 2} or Datum{Boid, 3}?")


# This logically represnts a stream of data either meant to be read out or saved
struct DataStream
    tag::String
    filepath::String
    data::AbstractVector{Datum}
end

DataStream(datum::Datum) = DataStream("", "", [datum])
DataStream(data::AbstractVector{Datum}) = DataStream("", "", data)

# ^ The two above are totally user created

mutable struct GathervCache{T}
    recv_buffer::Vector{T}
    recv_counts::Vector{Cint}
    displs::Vector{Cint}
end

function GathervCache(datum::Datum, cache::OutputCache{N}) where N
    T = datum.entrytype

    recv_counts = Cint[mesh_count(datum, wc.mesh) for wc in cache.worker_caches]
    displs      = Cint[0; cumsum(recv_counts)[1:end-1]]
    recv_buffer = Vector{T}(undef, Int(sum(recv_counts)))

    return GathervCache{T}(recv_buffer, recv_counts, displs)
end

mutable struct DataHandler{N}
    stream        :: DataStream
    topo          :: MPITopology{OutputCache{N}}
    gatherv_caches:: AbstractVector{GathervCache}
    tile_buffers  :: AbstractVector{AbstractArray} # Temp holding for data from Gatherv
    save_step     :: Int
end

out_cache(h::DataHandler)      = h.topo.cache
out_cart_comm(h::DataHandler)  = h.topo.cart_comm
intercomm(h::DataHandler)  = h.topo.intercomm
is_output(h::DataHandler)  = h.topo.is_output
om_mesh(h::DataHandler)    = out_cache(h).om_mesh
om_gm_offsets(h::DataHandler) = out_cache(h).om_gm_offsets
worker_caches(h::DataHandler) = out_cache(h).worker_caches
lm_om_offsets(h::DataHandler) = out_cache(h).lm_om_offsets
metadata(h::DataHandler) = h.save_step

function DataHandler(stream::DataStream, topo::MPITopology{OutputCache{N}}) where N
    cache = topo.cache

    gatherv_caches = map(datum -> GathervCache(datum, cache), stream.data)
    tile_buffers = map(datum -> tile_buffer(datum, cache.om_mesh), stream.data)

    return DataHandler{N}(stream, topo, gatherv_caches, tile_buffers, 1)
end

function DataHandler(stream::DataStream, cache::OutputCache{N}) where N
    topo = MPITopology(cache, true)
    return DataHandler(stream, topo)
end

function _build_gatherv_counts(datum::Datum, worker_caches::Vector{OutputWorkerCache{N}}) where N
    recv_counts = Cint[mesh_count(datum, wc.mesh) for wc in worker_caches]
    displs      = Cint[0; cumsum(recv_counts)[1:end-1]]
    return recv_counts, displs
end

function create_hdf5!(handler::DataHandler{N}, gm_dims::NTuple{N, Int}) where N
    gm = PseudoCubicalMesh(gm_dims...)
    h5open(handler.stream.filepath, "w", out_cart_comm(handler), MPI.Info()) do h5
        for datum in handler.stream.data
            if !haskey(h5, datum.groupname)
                create_group(h5, datum.groupname)
            end
            grp = h5[datum.groupname]

            for (name, spatial_dims) in zip(datum_dset_names(datum), datum_dims(datum, gm))
                dims    = tuple(1, spatial_dims...)
                maxdims = tuple(-1, spatial_dims...)
                chunk   = tuple(1, spatial_dims...)

                HDF5.create_dataset(grp, name, datum.entrytype,
                    HDF5.dataspace(dims, max_dims=maxdims);
                    chunk=chunk,
                    dxpl_mpio=:collective)
            end
        end
    end
end

function write_output!(handler::DataHandler{N}) where N
    gather!(handler)

    for (datum, gcache, tbuf) in zip(handler.stream.data, handler.gatherv_caches, handler.tile_buffers)
        scatter_to_tile!(datum, gcache, tbuf, handler)
    end

    h5open(handler.stream.filepath, "r+", out_cart_comm(handler), MPI.Info()) do h5
        for (datum, datum_tbufs) in zip(handler.stream.data, handler.tile_buffers)
            dsets = open_dsets(h5, datum)
            for (dset, tbuf) in zip(dsets, datum_tbufs)
                extent_dims, _ = HDF5.get_extent_dims(HDF5.dataspace(dset))
                time_dim, spatial_dims... = extent_dims
                if time_dim < handler.save_step
                    HDF5.set_extent_dims(dset, tuple(handler.save_step, spatial_dims...))
                end
                write_tile!(dset, tbuf, handler)
            end
        end
    end

    handler.save_step += 1
end

nfamilies(::Datum{Vert, N}) where N = 1
nfamilies(::Datum{Boid, N}) where N = 1
nfamilies(::Datum{Quad, 2})         = 1
nfamilies(::Datum{Quad, 3})         = 3
nfamilies(::Datum{Edge, N}) where N = N

function datum_dset_names(datum::Datum)
    n = nfamilies(datum)
    return n == 1 ? [datum.name] :
                    [datum.name * "_$i" for i in 1:n]
end

datum_dset_paths(datum::Datum) = (datum.groupname * "/") .* datum_dset_names(datum)
open_dsets(h5loc, datum::Datum) = [h5loc[path] for path in datum_dset_paths(datum)]

function write_output!(data_arrays::Vector{<:AbstractVector}, stream::DataStream, topo::MPITopology{WorkerCache{N}}) where N
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

function scatter_to_tile!(datum::Datum, gcache::GathervCache, tbufs::Vector, handler::DataHandler{N}) where N
    src_starts = [0; cumsum([mesh_count(datum, wc.mesh) for wc in worker_caches(handler)])[1:end-1]]

    for (wc, wc_offset, src_start) in zip(worker_caches(handler), lm_om_offsets(handler), src_starts)
        for (tbuf, dims) in zip(tbufs, datum_dims(datum, wc.mesh))
            _scatter_worker_chunk!(tbuf, gcache.recv_buffer, wc_offset, src_start, dims)
        end
    end
end

function _scatter_worker_chunk!(tbuf::AbstractArray, recv_buffer::AbstractVector, wc_offset::NTuple{N, Int},
                                 src_start::Int, dims::NTuple{N, Int}) where N
    n      = prod(dims)
    ranges = ntuple(i -> wc_offset[i]+1 : wc_offset[i]+dims[i], N)
    tbuf[ranges...] .= reshape(recv_buffer[src_start+1 : src_start+n], dims)
end

function write_tile!(dset, tbuf::Array, handler::DataHandler{N}) where N
    ranges = _hyperslab_ranges(out_cache(handler), size(tbuf))
    dset[handler.save_step, ranges...] = tbuf
    return dset
end

function _hyperslab_ranges(cache::OutputCache{N}, count::NTuple{N,Int}) where N
    offset = cache.om_gm_offsets
    return ntuple(j -> offset[j]+1 : offset[j]+count[j], N)
end

# ── datum_dims ────────────────────────────────────────────────────────────────
# This returns a vector to handle aligned elements seperatly (edges in 2D/3D, or quads in 3D)

datum_dims(::Datum{Vert, 2}, m::AbstractCubicalComplex2D) = [(nx(m), ny(m))]
datum_dims(::Datum{Vert, 3}, m::AbstractCubicalComplex3D) = [(nx(m), ny(m), nz(m))]

datum_dims(::Datum{Edge, 2}, m::AbstractCubicalComplex2D) =
    [(nxe(m), ny(m)),   # X family
     (nx(m),  nye(m))]  # Y family
datum_dims(::Datum{Edge, 3}, m::AbstractCubicalComplex3D) =
    [(nxe(m), ny(m),  nz(m)),   # X family
     (nx(m),  nye(m), nz(m)),   # Y family
     (nx(m),  ny(m),  nze(m))]  # Z family

datum_dims(::Datum{Quad, 2}, m::AbstractCubicalComplex2D) = [(nxq(m), nyq(m))]
datum_dims(::Datum{Quad, 3}, m::AbstractCubicalComplex3D) =
    [(nxq(m), nyq(m), nz(m)),   # XY family
     (nxq(m), ny(m),  nzq(m)),  # XZ family
     (nx(m),  nyq(m), nzq(m))]  # YZ family

datum_dims(::Datum{Boid, 2}, m::AbstractCubicalComplex2D) = [(nxq(m), nyq(m))]
datum_dims(::Datum{Boid, 3}, m::AbstractCubicalComplex3D) = [(nxb(m), nyb(m), nzb(m))]

# ── tile_buffer ───────────────────────────────────────────────────────────────

function tile_buffer(datum::Datum, m::AbstractCubicalComplex)
    [Array{datum.entrytype}(undef, dims...) for dims in datum_dims(datum, m)]
end

# ── mesh_count ────────────────────────────────────────────────────────────────

function mesh_count(datum::Datum, m::AbstractCubicalComplex)
    sum(prod(dims) for dims in datum_dims(datum, m))
end