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

### GhostRegion ###

ghost_index(::Datum{Vert,N}) where {N} = 1
ghost_index(::Datum{Edge,N}) where {N} = 2
ghost_index(::Datum{Quad,N}) where {N} = 3
ghost_index(::Datum{Boid,N}) where {N} = 4

meshtype_index(::Val{1}) = Vert
meshtype_index(::Val{2}) = Edge
meshtype_index(::Val{3}) = Quad
meshtype_index(::Val{4}) = Boid

struct GhostRegion{M<:AbstractMeshType,N}
    send::AbstractVector{AbstractVector{Int32}}
    recv::AbstractVector{AbstractVector{Int32}}
end

function GhostRegion(::Type{Datum{Quad,2}}, s::AbstractCubicalComplex2D)
    hx_ = hxq(s)
    hy_ = hyq(s)
    nxq_ = nxq(s)
    nyq_ = nyq(s)
    nxqr_ = nxqr(s)
    nyqr_ = nyqr(s)

    # X-axis: transverse range is interior y only (no y-halo)
    rl_x = Int32[coord_to_quad(s, ax, b) for ax in 1:hx_, b in (hy_ + 1):(hy_ + nyqr_)][:]
    sl_x = Int32[coord_to_quad(s, ax, b) for ax in (hx_ + 1):(2hx_), b in (hy_ + 1):(hy_ + nyqr_)][:]

    sh_x = Int32[coord_to_quad(s, ax, b) for ax in (nxq_ - 2hx_ + 1):(nxq_ - hx_), b in (hy_ + 1):(hy_ + nyqr_)][:]
    rh_x = Int32[coord_to_quad(s, ax, b) for ax in (nxq_ - hx_ + 1):nxq_, b in (hy_ + 1):(hy_ + nyqr_)][:]

    # Y-axis: transverse range is full x including x-halo (corners now covered)
    rl_y = Int32[coord_to_quad(s, b, ax) for ax in 1:hy_, b in 1:nxq_][:]
    sl_y = Int32[coord_to_quad(s, b, ax) for ax in (hy_ + 1):(2hy_), b in 1:nxq_][:]

    sh_y = Int32[coord_to_quad(s, b, ax) for ax in (nyq_ - 2hy_ + 1):(nyq_ - hy_), b in 1:nxq_][:]
    rh_y = Int32[coord_to_quad(s, b, ax) for ax in (nyq_ - hy_ + 1):nyq_, b in 1:nxq_][:]

    send = [sl_x, sh_x, sl_y, sh_y]
    recv = [rl_x, rh_x, rl_y, rh_y]

    return GhostRegion{Quad,2}(send, recv)
end

function GhostRegion(::Type{Datum{Boid,3}}, s::AbstractCubicalComplex3D)
    hx_ = hxb(s)
    hy_ = hyb(s)
    hz_ = hzb(s)
    nxb_ = nxb(s)
    nyb_ = nyb(s)
    nzb_ = nzb(s)
    nxbr_ = nxbr(s)
    nybr_ = nybr(s)
    nzbr_ = nzbr(s)

    # X-axis: transverse is real y and real z only
    rl_x = Int32[coord_to_boid(s, ax, b, c) for ax in 1:hx_, b in (hy_ + 1):(hy_ + nybr_), c in (hz_ + 1):(hz_ + nzbr_)][:]
    sl_x = Int32[coord_to_boid(s, ax, b, c) for ax in (hx_ + 1):(2hx_), b in (hy_ + 1):(hy_ + nybr_), c in (hz_ + 1):(hz_ + nzbr_)][:]
    sh_x = Int32[coord_to_boid(s, ax, b, c) for ax in (nxb_ - 2hx_ + 1):(nxb_ - hx_), b in (hy_ + 1):(hy_ + nybr_), c in (hz_ + 1):(hz_ + nzbr_)][:]
    rh_x = Int32[coord_to_boid(s, ax, b, c) for ax in (nxb_ - hx_ + 1):nxb_, b in (hy_ + 1):(hy_ + nybr_), c in (hz_ + 1):(hz_ + nzbr_)][:]

    # Y-axis: transverse is full x (x-halo filled), real z only
    rl_y = Int32[coord_to_boid(s, b, ax, c) for ax in 1:hy_, b in 1:nxb_, c in (hz_ + 1):(hz_ + nzbr_)][:]
    sl_y = Int32[coord_to_boid(s, b, ax, c) for ax in (hy_ + 1):(2hy_), b in 1:nxb_, c in (hz_ + 1):(hz_ + nzbr_)][:]
    sh_y = Int32[coord_to_boid(s, b, ax, c) for ax in (nyb_ - 2hy_ + 1):(nyb_ - hy_), b in 1:nxb_, c in (hz_ + 1):(hz_ + nzbr_)][:]
    rh_y = Int32[coord_to_boid(s, b, ax, c) for ax in (nyb_ - hy_ + 1):nyb_, b in 1:nxb_, c in (hz_ + 1):(hz_ + nzbr_)][:]

    # Z-axis: transverse is full x and full y (both halos filled)
    rl_z = Int32[coord_to_boid(s, b, c, ax) for ax in 1:hz_, b in 1:nxb_, c in 1:nyb_][:]
    sl_z = Int32[coord_to_boid(s, b, c, ax) for ax in (hz_ + 1):(2hz_), b in 1:nxb_, c in 1:nyb_][:]
    sh_z = Int32[coord_to_boid(s, b, c, ax) for ax in (nzb_ - 2hz_ + 1):(nzb_ - hz_), b in 1:nxb_, c in 1:nyb_][:]
    rh_z = Int32[coord_to_boid(s, b, c, ax) for ax in (nzb_ - hz_ + 1):nzb_, b in 1:nxb_, c in 1:nyb_][:]

    send = [sl_x, sh_x, sl_y, sh_y, sl_z, sh_z]
    recv = [rl_x, rh_x, rl_y, rh_y, rl_z, rh_z]

    return GhostRegion{Boid,3}(send, recv)
end

@enum Face begin
    WEST = 1
    EAST = 2
    SOUTH = 3
    NORTH = 4
    DOWN = 5
    UP = 6
end

low_face(side::GridSide) = Face(2 * (Int(side) + 1) - 1)
high_face(side::GridSide) = Face(2 * (Int(side) + 1))

GhostRegion(::Type{Datum{Vert,N}}, s) where {N} = GhostRegion{Vert,N}(Int32[], Int32[])   # stub
GhostRegion(::Type{Datum{Edge,N}}, s) where {N} = GhostRegion{Edge,N}(Int32[], Int32[])   # stub
GhostRegion(::Type{Datum{Quad,3}}, s) where {} = GhostRegion{Quad,3}(Int32[], Int32[])   # stub

send_slab(g::GhostRegion, face::Face) = send_slab(g, Int(face))
recv_slab(g::GhostRegion, face::Face) = recv_slab(g, Int(face))

send_slab(g::GhostRegion, i::Int) = g.send[i]
recv_slab(g::GhostRegion, i::Int) = g.recv[i]

### ExchangeHandler ###

# Vector over each Datum
struct FaceBuffer
    slabs::Vector{Vector{Int32}}
    cell_lens::Vector{Int}
end

function FaceBuffer(ghosts::AbstractVector, stream::DataStream, face::Face, sendrecv::Symbol)
    get_slab = (sendrecv == :recv ? recv_slab : send_slab)
    slabs = Vector{Int32}[]
    cell_lens = Int[]
    for datum in stream.data
        g = ghosts[ghost_index(datum)]
        slab = get_slab(g, face)
        push!(slabs, slab)
        push!(cell_lens, length(slab))
    end
    return FaceBuffer(slabs, cell_lens)
end

struct ExchangeHandler{N,FT}
    topo::MPITopology{WorkerCache{N}}
    stream::DataStream
    ghosts::Vector{GhostRegion} # length N + 1

    # One for each face
    send_face::Vector{FaceBuffer}   # length 2N
    recv_face::Vector{FaceBuffer}   # length 2N

    send_bufs::Vector{Vector{FT}}   # length 2N
    recv_bufs::Vector{Vector{FT}}   # length 2N

    send_reqs::Vector{MPI.Request}  # length 2N
    recv_reqs::Vector{MPI.Request}  # length 2N
end

const AXIS_NAMES_2D = (:west, :east, :south, :north)
const AXIS_NAMES_3D = (:west, :east, :south, :north, :down, :up)

function ExchangeHandler(stream::DataStream, topo::MPITopology{WorkerCache{N}}, s::AbstractCubicalComplex) where {N}
    @assert !isempty(stream.data) "ExchangeHandler: stream must contain at least one datum"

    FT = stream.data[1].entrytype
    @assert all(d.entrytype == FT for d in stream.data) "ExchangeHandler: all datums must share the same entrytype, got $(unique(d.entrytype for d in stream.data))"

    ghosts = map(1:(N + 1)) do i
        elemtype = meshtype_index(Val(i))
        return GhostRegion(Datum{elemtype,N}, s)  # lower slots stubbed
    end

    face_names = N == 2 ? AXIS_NAMES_2D : AXIS_NAMES_3D
    faces = Face.(1:(2N))

    # FaceBuffer for each face (info for packing/unpacking) (all datums covered)
    send_face = [FaceBuffer(ghosts, stream, face, :send) for face in faces]
    recv_face = [FaceBuffer(ghosts, stream, face, :recv) for face in faces]

    # Buffer size for each face
    buf_sizes = [sum(fb.cell_lens) for fb in send_face]

    # Send and recv buffers for each face
    send_bufs = map(i -> Vector{FT}(undef, buf_sizes[i]), 1:(2N))
    recv_bufs = map(i -> Vector{FT}(undef, buf_sizes[i]), 1:(2N))

    # W -> 0, E -> 1, S -> 2, N -> 3, D -> 4, U -> 5
    nb = topo.cache.neighbors
    send_reqs = map(1:(2N)) do i
        axis = (i - 1) ÷ 2
        tag = isodd(i) ? 2 * axis : 2 * axis + 1
        return MPI.Send_init(send_bufs[i], nb[face_names[i]], tag, topo.cart_comm)
    end

    # W -> 1, E -> 0, S -> 3, N -> 2, D -> 5, U -> 4
    recv_reqs = map(1:(2N)) do i
        axis = (i - 1) ÷ 2
        tag = isodd(i) ? 2 * axis + 1 : 2 * axis
        return MPI.Recv_init(recv_bufs[i], nb[face_names[i]], tag, topo.cart_comm)
    end
    return ExchangeHandler{N,FT}(topo, stream, ghosts, send_face, recv_face, send_bufs, recv_bufs, send_reqs, recv_reqs)
end

face_index(f::Face) = Int(f)

send_buf(handler::ExchangeHandler, f::Face) = handler.send_bufs[face_index(f)]
recv_buf(handler::ExchangeHandler, f::Face) = handler.recv_bufs[face_index(f)]
send_req(handler::ExchangeHandler, f::Face) = handler.send_reqs[face_index(f)]
recv_req(handler::ExchangeHandler, f::Face) = handler.recv_reqs[face_index(f)]
send_face(handler::ExchangeHandler, f::Face) = handler.send_face[face_index(f)]
recv_face(handler::ExchangeHandler, f::Face) = handler.recv_face[face_index(f)]

nfaces(::ExchangeHandler{N,FT}) where {N,FT} = 2 * N
faces(::ExchangeHandler{N,FT}) where {N,FT} = 1:(2N)
facenames(handler::ExchangeHandler) = Face.(faces(handler))

function exchange!(handler::ExchangeHandler{N,FT}, vars::NamedTuple) where {N,FT}
    fields = map(datum -> vars[Symbol(datum.name)], handler.stream.data)

    _exchange_axis!(handler, fields, EASTWEST)
    _exchange_axis!(handler, fields, NORTHSOUTH)
    N == 3 && _exchange_axis!(handler, fields, UPDOWN)

    return nothing
end

function _exchange_axis!(handler::ExchangeHandler, fields::AbstractVector, side::GridSide)
    low = low_face(side)
    high = high_face(side)

    MPI.Start(recv_req(handler, low))
    MPI.Start(recv_req(handler, high))

    _pack_face!(send_buf(handler, low), send_face(handler, low), fields)
    _pack_face!(send_buf(handler, high), send_face(handler, high), fields)

    MPI.Start(send_req(handler, low))
    MPI.Start(send_req(handler, high))

    MPI.Wait(recv_req(handler, low))
    MPI.Wait(recv_req(handler, high))
    MPI.Wait(send_req(handler, low))
    MPI.Wait(send_req(handler, high))

    _unpack_face!(recv_buf(handler, low), recv_face(handler, low), fields)
    _unpack_face!(recv_buf(handler, high), recv_face(handler, high), fields)

    return nothing
end

function _pack_face!(buf::Vector, fb::FaceBuffer, fields::AbstractVector)
    offset = 0
    for (field, slab, cell_len) in zip(fields, fb.slabs, fb.cell_lens)
        for k in 1:cell_len
            buf[offset + k] = field[slab[k]]
        end
        offset += cell_len
    end
    return nothing
end

function _unpack_face!(buf::Vector, fb::FaceBuffer, fields::AbstractVector)
    offset = 0
    for (field, slab, cell_len) in zip(fields, fb.slabs, fb.cell_lens)
        for k in 1:cell_len
            field[slab[k]] = buf[offset + k]
        end
        offset += cell_len
    end
    return nothing
end

function close!(handler::ExchangeHandler{N,FT}) where {N,FT}
    for i in faces(handler)
        MPI.free(handler.send_reqs[i])
        MPI.free(handler.recv_reqs[i])
    end
    return nothing
end