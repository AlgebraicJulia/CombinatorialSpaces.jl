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
struct Datum{M<:AbstractMeshType,N}
    name::String
    groupname::String
    entrytype::DataType
end

Datum{M,N}() where {M<:AbstractMeshType,N} = Datum{M,N}("", "", Float64)

function Datum{Boid,2}(args...)
    return error("Datum{Boid, 2} is invalid: Boid is a 3-cell and only exists in 3D. " * "Did you mean Datum{Quad, 2} or Datum{Boid, 3}?")
end

nfamilies(::Datum{Vert,N}) where {N} = 1
nfamilies(::Datum{Boid,N}) where {N} = 1
nfamilies(::Datum{Quad,2}) = 1
nfamilies(::Datum{Quad,3}) = 3
nfamilies(::Datum{Edge,N}) where {N} = N

function datum_dset_names(datum::Datum)
    n = nfamilies(datum)
    return n == 1 ? [datum.name] : [datum.name * "_$i" for i in 1:n]
end

datum_dset_paths(datum::Datum) = (datum.groupname * "/") .* datum_dset_names(datum)
open_dsets(h5loc, datum::Datum) = [h5loc[path] for path in datum_dset_paths(datum)]

datum_dims(::Datum{Vert,2}, m::AbstractCubicalComplex2D) = [(nx(m), ny(m))]
datum_dims(::Datum{Vert,3}, m::AbstractCubicalComplex3D) = [(nx(m), ny(m), nz(m))]

datum_dims(::Datum{Edge,2}, m::AbstractCubicalComplex2D) = [
    (nxe(m), ny(m)),   # X family
    (nx(m), nye(m)),
]  # Y family
function datum_dims(::Datum{Edge,3}, m::AbstractCubicalComplex3D)
    return [
        (nxe(m), ny(m), nz(m)),   # X family
        (nx(m), nye(m), nz(m)),   # Y family
        (nx(m), ny(m), nze(m)),
    ]  # Z family
end  # Z family

datum_dims(::Datum{Quad,2}, m::AbstractCubicalComplex2D) = [(nxq(m), nyq(m))]
function datum_dims(::Datum{Quad,3}, m::AbstractCubicalComplex3D)
    return [
        (nxq(m), nyq(m), nz(m)),   # XY family
        (nxq(m), ny(m), nzq(m)),  # XZ family
        (nx(m), nyq(m), nzq(m)),
    ]  # YZ family
end  # YZ family

datum_dims(::Datum{Boid,2}, m::AbstractCubicalComplex2D) = [(nxq(m), nyq(m))]
datum_dims(::Datum{Boid,3}, m::AbstractCubicalComplex3D) = [(nxb(m), nyb(m), nzb(m))]

function tile_buffer(datum::Datum, m::AbstractCubicalComplex)
    return [Array{datum.entrytype}(undef, dims...) for dims in datum_dims(datum, m)]
end

function mesh_count(datum::Datum, m::AbstractCubicalComplex)
    return sum(prod(dims) for dims in datum_dims(datum, m))
end

# This logically represnts a stream of data either meant to be read out or saved
struct DataStream
    tag::String
    filepath::String
    data::AbstractVector{Datum}
end

DataStream(datum::Datum) = DataStream("", "", [datum])
DataStream(data::AbstractVector{Datum}) = DataStream("", "", data)

abstract type AbstractMetaData end  # stub, filled later

# ── Signalling ────────────────────────────────────────────────────────────────

@enum SignalTag::Int32 begin
    SIGNAL_WRITE = 0
    SIGNAL_DONE = -1
end

function worker_to_output(tag::SignalTag, topo::MPITopology{<:WorkerCache})
    buf = Ref{Int32}(Int32(tag))
    MPI.Send(buf, topo.intercomm; dest = 0)
    # println("WORKER $(MPI.Comm_rank(topo.cart_comm)) sent $(SignalTag(buf[]))")
    return nothing
end

function output_from_worker(topo::MPITopology{<:OutputCache})
    buf = Ref{Int32}(Int32(0))
    MPI.Recv!(buf, topo.intercomm; source = 0)
    # println("OUTPUT $(MPI.Comm_rank(topo.cart_comm)) got $(SignalTag(buf[]))")
    return SignalTag(buf[])
end

mutable struct GathervCache{T}
    recv_buffer::Vector{T}
    recv_counts::Vector{Cint}
    displs::Vector{Cint}
end

function GathervCache(datum::Datum, cache::OutputCache{N}) where {N}
    T = datum.entrytype

    recv_counts = Cint[mesh_count(datum, wc.mesh) for wc in cache.worker_caches]
    displs = Cint[0; cumsum(recv_counts)[1:(end - 1)]]
    recv_buffer = Vector{T}(undef, Int(sum(recv_counts)))

    return GathervCache{T}(recv_buffer, recv_counts, displs)
end

mutable struct DataHandler{N}
    stream::DataStream
    topo::MPITopology{OutputCache{N}}
    gatherv_caches::AbstractVector{GathervCache}
    tile_buffers::AbstractVector{AbstractArray} # Temp holding for data from Gatherv
end

out_cache(h::DataHandler) = h.topo.cache
out_cart_comm(h::DataHandler) = h.topo.cart_comm
intercomm(h::DataHandler) = h.topo.intercomm
is_output(h::DataHandler) = h.topo.is_output
om_mesh(h::DataHandler) = out_cache(h).om_mesh
om_gm_offsets(h::DataHandler) = out_cache(h).om_gm_offsets
worker_caches(h::DataHandler) = out_cache(h).worker_caches
lm_om_offsets(h::DataHandler) = out_cache(h).lm_om_offsets
metadata(h::DataHandler) = h.save_step

function DataHandler(stream::DataStream, topo::MPITopology{OutputCache{N}}) where {N}
    cache = topo.cache

    gatherv_caches = map(datum -> GathervCache(datum, cache), stream.data)
    tile_buffers = map(datum -> tile_buffer(datum, cache.om_mesh), stream.data)

    return DataHandler{N}(stream, topo, gatherv_caches, tile_buffers)
end

function DataHandler(stream::DataStream, cache::OutputCache{N}) where {N}
    topo = MPITopology(cache, true)
    return DataHandler(stream, topo)
end

function _build_gatherv_counts(datum::Datum, worker_caches::Vector{OutputWorkerCache{N}}) where {N}
    recv_counts = Cint[mesh_count(datum, wc.mesh) for wc in worker_caches]
    displs = Cint[0; cumsum(recv_counts)[1:(end - 1)]]
    return recv_counts, displs
end

function create_hdf5!(handler::DataHandler{N}, gm_dims::NTuple{N,Int}) where {N}
    gm = PseudoCubicalMesh(gm_dims...)
    h5open(handler.stream.filepath, "w", out_cart_comm(handler), MPI.Info()) do h5
        for datum in handler.stream.data
            if !haskey(h5, datum.groupname)
                create_group(h5, datum.groupname)
            end
            grp = h5[datum.groupname]
            for (name, spatial_dims) in zip(datum_dset_names(datum), datum_dims(datum, gm))
                dims = tuple(0, spatial_dims...)
                maxdims = tuple(-1, spatial_dims...)
                chunk = tuple(1, spatial_dims...)
                HDF5.create_dataset(grp, name, datum.entrytype, HDF5.dataspace(dims; max_dims = maxdims); chunk = chunk, dxpl_mpio = :collective)
            end
        end
    end
end

# Output-side
function write_output!(handler::DataHandler{N}) where {N}
    gather!(handler)

    for (datum, gcache, tbuf) in zip(handler.stream.data, handler.gatherv_caches, handler.tile_buffers)
        scatter_to_tile!(datum, gcache, tbuf, handler)
    end

    h5open(handler.stream.filepath, "r+", out_cart_comm(handler), MPI.Info()) do h5
        for (datum, datum_tbufs) in zip(handler.stream.data, handler.tile_buffers)
            dsets = open_dsets(h5, datum)
            for (dset, tbuf) in zip(dsets, datum_tbufs)
                time_dim, spatial_dims... = HDF5.get_extent_dims(HDF5.dataspace(dset))[1]
                HDF5.set_extent_dims(dset, (time_dim + 1, spatial_dims...))
                write_tile!(dset, tbuf, time_dim + 1, handler)
            end
        end
    end
end

# Worker-side
function send_output!(data_arrays::Vector{<:AbstractVector}, stream::DataStream, topo::MPITopology{<:WorkerCache})
    worker_to_output(SIGNAL_WRITE, topo)
    for (data, datum) in zip(data_arrays, stream.data)
        gather!(data, datum, topo)
    end
end

function gather!(handler::DataHandler{N}) where {N}
    mpi_root = Ref{Cint}(MPI.API.MPI_ROOT[])

    for (datum, gcache) in zip(handler.stream.data, handler.gatherv_caches)
        GC.@preserve gcache begin
            MPI.API.MPI_Gatherv(
                C_NULL,
                Cint(0),
                MPI.Datatype(datum.entrytype),
                gcache.recv_buffer,
                gcache.recv_counts,
                gcache.displs,
                MPI.Datatype(datum.entrytype),
                mpi_root[],
                handler.topo.intercomm,
            )
        end
    end
end

function gather!(data::AbstractVector{T}, datum::Datum{M,N}, topo::MPITopology{WorkerCache{N}}) where {T,M,N}
    GC.@preserve data begin
        MPI.API.MPI_Gatherv(data, Cint(length(data)), MPI.Datatype(datum.entrytype), C_NULL, C_NULL, C_NULL, MPI.Datatype(datum.entrytype), Cint(0), topo.intercomm)
    end
end

function scatter_to_tile!(datum::Datum, gcache::GathervCache, tbufs::Vector, handler::DataHandler{N}) where {N}
    src_starts = [0; cumsum([mesh_count(datum, wc.mesh) for wc in worker_caches(handler)])[1:(end - 1)]]

    for (wc, wc_offset, src_start) in zip(worker_caches(handler), lm_om_offsets(handler), src_starts)
        family_offset = src_start # For when a Datum has multiple buffers (e.g. Edge in 2D/3D, Quad in 3D)
        for (tbuf, dims) in zip(tbufs, datum_dims(datum, wc.mesh))
            _scatter_worker_chunk!(tbuf, gcache.recv_buffer, wc_offset, family_offset, dims)
            family_offset += prod(dims)
        end
    end
end

# TODO: Currently, wc_offset works in vertex offset from origin.
# While this offset should work for any element, this is based on a design
# choice and may need to be generalized if behavior changes.
function _scatter_worker_chunk!(tbuf::AbstractArray, recv_buffer::AbstractVector, wc_offset::NTuple{N,Int}, src_start::Int, dims::NTuple{N,Int}) where {N}
    n = prod(dims)
    ranges = ntuple(i -> (wc_offset[i] + 1):(wc_offset[i] + dims[i]), N)
    return tbuf[ranges...] .= reshape(recv_buffer[(src_start + 1):(src_start + n)], dims)
end

function write_tile!(dset, tbuf::Array, step::Int, handler::DataHandler{N}) where {N}
    ranges = _hyperslab_ranges(out_cache(handler), size(tbuf))
    return dset[step, ranges...] = tbuf
end

function _hyperslab_ranges(cache::OutputCache{N}, count::NTuple{N,Int}) where {N}
    offset = cache.om_gm_offsets
    return ntuple(j -> (offset[j] + 1):(offset[j] + count[j]), N)
end

struct ExchangeHandler{N,TN,FT}
    topo::MPITopology{WorkerCache{N}}
    stream::DataStream
    ghosts::NamedTuple

    # 2N send and recv buffers ordered: west/east/south/north[/down/up]
    send_bufs::NTuple{TN,Vector{FT}}
    recv_bufs::NTuple{TN,Vector{FT}}

    # 2N persistent send and recv requests
    send_reqs::NTuple{TN,MPI.Request}
    recv_reqs::NTuple{TN,MPI.Request}
end

const AXIS_NAMES_2D = (:west, :east, :south, :north)
const AXIS_NAMES_3D = (:west, :east, :south, :north, :down, :up)

function ExchangeHandler(stream::DataStream, topo::MPITopology{WorkerCache{N}}, s::AbstractCubicalComplex) where {N}
    @assert !isempty(stream.data) "ExchangeHandler: stream must contain at least one datum"

    FT = stream.data[1].entrytype
    @assert all(d.entrytype == FT for d in stream.data) "ExchangeHandler: all datums must share the same entrytype, got $(unique(d.entrytype for d in stream.data))"

    # TODO: This will have to change to just check the element dimension
    valid_type = N == 2 ? Datum{Quad,2} : Datum{Boid,3}
    @assert all(d isa valid_type for d in stream.data) "ExchangeHandler: all datums must be $(valid_type) for a $(N)D mesh"

    ghosts = _build_ghosts(Val(N), s)

    face_names = N == 2 ? AXIS_NAMES_2D : AXIS_NAMES_3D   # NTuple{2N, Symbol}

    # TODO: This assumes that we're dealing with the same datums
    # This code will have to change when dealing with multi-datum types
    n_vars = length(stream.data)
    buf_sizes = ntuple(i -> length(ghosts[face_names[i]].send) * n_vars, 2N) # 2N for each pair of faces (e.g. west/east)

    send_bufs = ntuple(i -> Vector{FT}(undef, buf_sizes[i]), 2N)
    recv_bufs = ntuple(i -> Vector{FT}(undef, buf_sizes[i]), 2N)

    # W -> 0, E -> 1, S -> 2, N -> 3, D -> 4, U -> 5
    nb = topo.cache.neighbors
    send_reqs = ntuple(2N) do i
        axis = (i - 1) ÷ 2
        tag = isodd(i) ? 2 * axis : 2 * axis + 1
        return MPI.Send_init(send_bufs[i], nb[face_names[i]], tag, topo.cart_comm)
    end

    # W -> 1, E -> 0, S -> 3, N -> 2, D -> 5, U -> 4
    recv_reqs = ntuple(2N) do i
        axis = (i - 1) ÷ 2
        tag = isodd(i) ? 2 * axis + 1 : 2 * axis
        return MPI.Recv_init(recv_bufs[i], nb[face_names[i]], tag, topo.cart_comm)
    end
    return ExchangeHandler{N,2N,FT}(topo, stream, ghosts, send_bufs, recv_bufs, send_reqs, recv_reqs)
end

function exchange!(handler::ExchangeHandler{N,TN,FT}, vars::NamedTuple) where {N,TN,FT}
    face_names = N == 2 ? AXIS_NAMES_2D : AXIS_NAMES_3D

    fields = ntuple(j -> vars[Symbol(handler.stream.data[j].name)], length(handler.stream.data))
    send_slabs = ntuple(i -> handler.ghosts[face_names[i]].send, 2N)
    recv_slabs = ntuple(i -> handler.ghosts[face_names[i]].recv, 2N)

    _exchange_axis!(handler, fields, send_slabs, recv_slabs, 1)
    _exchange_axis!(handler, fields, send_slabs, recv_slabs, 2)
    N == 3 && _exchange_axis!(handler, fields, send_slabs, recv_slabs, 3)

    return nothing
end

function _exchange_axis!(handler::ExchangeHandler, fields::NTuple, send_slabs::NTuple, recv_slabs::NTuple, axis::Int)
    low_idx = 2 * axis - 1
    high_idx = 2 * axis

    low_send = handler.send_bufs[low_idx]
    high_send = handler.send_bufs[high_idx]
    low_recv = handler.recv_bufs[low_idx]
    high_recv = handler.recv_bufs[high_idx]

    low_send_req = handler.send_reqs[low_idx]
    high_send_req = handler.send_reqs[high_idx]
    low_recv_req = handler.recv_reqs[low_idx]
    high_recv_req = handler.recv_reqs[high_idx]

    MPI.Start(low_recv_req)
    MPI.Start(high_recv_req)

    _pack_face!(low_send, send_slabs[low_idx], fields)
    _pack_face!(high_send, send_slabs[high_idx], fields)

    MPI.Start(low_send_req)
    MPI.Start(high_send_req)

    MPI.Wait(low_recv_req)
    MPI.Wait(high_recv_req)
    MPI.Wait(low_send_req)
    MPI.Wait(high_send_req)

    _unpack_face!(low_recv, recv_slabs[low_idx], fields)
    _unpack_face!(high_recv, recv_slabs[high_idx], fields)

    return nothing
end

function _pack_face!(buf::Vector, slab::Vector, fields::NTuple)
    n_cell = length(slab)
    for (j, field) in enumerate(fields)
        offset = (j - 1) * n_cell
        for k in 1:n_cell
            buf[offset + k] = field[slab[k]]
        end
    end
    return nothing
end

function _unpack_face!(buf::Vector, slab::Vector, fields::NTuple)
    n_cell = length(slab)
    for (j, field) in enumerate(fields)
        offset = (j - 1) * n_cell
        for k in 1:n_cell
            field[slab[k]] = buf[offset + k]
        end
    end
    return nothing
end

function close!(handler::ExchangeHandler{N,TN,FT}) where {N,TN,FT}
    ntuple(2N) do i
        MPI.free(handler.send_reqs[i])
        MPI.free(handler.recv_reqs[i])
        return nothing
    end
    return nothing
end