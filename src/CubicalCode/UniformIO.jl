using MPI
using HDF5
using Adapt
using KernelAbstractions

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

datum_dims(::Datum{Vert,2}, s::AbstractCubicalComplex2D) = [(nx(s), ny(s))]
datum_dims(::Datum{Vert,3}, s::AbstractCubicalComplex3D) = [(nx(s), ny(s), nz(s))]

datum_dims(::Datum{Edge,2}, s::AbstractCubicalComplex2D) = [
    (nxe(s), ny(s)),   # X family
    (nx(s), nye(s)),
]  # Y family
function datum_dims(::Datum{Edge,3}, s::AbstractCubicalComplex3D)
    return [
        (nxe(s), ny(s), nz(s)),   # X family
        (nx(s), nye(s), nz(s)),   # Y family
        (nx(s), ny(s), nze(s)),
    ]  # Z family
end  # Z family

datum_dims(::Datum{Quad,2}, s::AbstractCubicalComplex2D) = [(nxq(s), nyq(s))]
function datum_dims(::Datum{Quad,3}, s::AbstractCubicalComplex3D)
    return [
        (nxq(s), nyq(s), nz(s)),   # XY family
        (nxq(s), ny(s), nzq(s)),  # XZ family
        (nx(s), nyq(s), nzq(s)),
    ]  # YZ family
end  # YZ family

datum_dims(::Datum{Boid,2}, s::AbstractCubicalComplex2D) = [(nxq(s), nyq(s))]
datum_dims(::Datum{Boid,3}, s::AbstractCubicalComplex3D) = [(nxb(s), nyb(s), nzb(s))]

function tile_buffer(datum::Datum, s::AbstractCubicalComplex)
    return [Array{datum.entrytype}(undef, dims...) for dims in datum_dims(datum, s)]
end

function mesh_count(datum::Datum, s::AbstractCubicalComplex)
    return sum(prod(dims) for dims in datum_dims(datum, s))
end

# This logically represnts a stream of data either meant to be read out or saved
struct DataStream
    tag::String
    filepath::String
    data::AbstractVector{Datum}
end

DataStream(datum::Datum) = DataStream("", "", [datum])
DataStream(data::AbstractVector{Datum}) = DataStream("", "", data)
DataStream(data::AbstractVector) = DataStream("", "", Vector{Datum}(data)) # Convenience for users

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
function send_output!(data_arrays::AbstractVector{<:AbstractVector}, stream::DataStream, topo::MPITopology{<:WorkerCache})
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

function Adapt.adapt_structure(backend, gr::GhostRegion{M,N}) where {M,N}
    GhostRegion{M,N}(
        map(s -> adapt(backend, s), gr.send),
        map(s -> adapt(backend, s), gr.recv),
    )
end

# TODO: This needs to be tested
function GhostRegion(::Type{Datum{Vert,2}}, s::AbstractCubicalComplex2D)
    hw_ = halo_west(s);  he_ = halo_east(s)
    hs_ = halo_south(s); hn_ = halo_north(s)
    nx_ = nx(s)
    ny_ = ny(s)
    nyr_ = nyr(s)

    # ── EASTWEST pass (slice in x, interior-y transverse only) ────────────
    ew_yt = (hs_ + 1):(hs_ + nyr_)

    sl_ew = Int32[coord_to_vert(s, ax, b) for ax in (hw_ + 1):(2hw_),            b in ew_yt][:]
    rh_ew = Int32[coord_to_vert(s, ax, b) for ax in (nx_ - he_ + 1):nx_,         b in ew_yt][:]
    sh_ew = Int32[coord_to_vert(s, ax, b) for ax in (nx_ - 2he_ + 1):(nx_ - he_), b in ew_yt][:]
    rl_ew = Int32[coord_to_vert(s, ax, b) for ax in 1:hw_,                        b in ew_yt][:]

    # ── NORTHSOUTH pass (slice in y, full-x transverse) ───────────────────
    sl_ns = Int32[coord_to_vert(s, b, ax) for ax in (hs_ + 1):(2hs_),            b in 1:nx_][:]
    rh_ns = Int32[coord_to_vert(s, b, ax) for ax in (ny_ - hn_ + 1):ny_,         b in 1:nx_][:]
    sh_ns = Int32[coord_to_vert(s, b, ax) for ax in (ny_ - 2hn_ + 1):(ny_ - hn_), b in 1:nx_][:]
    rl_ns = Int32[coord_to_vert(s, b, ax) for ax in 1:hs_,                        b in 1:nx_][:]

    send = [sl_ew, sh_ew, sl_ns, sh_ns]
    recv = [rl_ew, rh_ew, rl_ns, rh_ns]

    return GhostRegion{Vert,2}(send, recv)
end

# TODO: Pretty sure we can remove the [:] if we use multiple "for" statements
function GhostRegion(::Type{Datum{Quad,2}}, s::AbstractCubicalComplex2D)
    hw_ = halo_west(s);  he_ = halo_east(s)
    hs_ = halo_south(s); hn_ = halo_north(s)
    nxq_ = nxq(s)
    nyq_ = nyq(s)
    nxqr_ = nxqr(s)
    nyqr_ = nyqr(s)

    # X-axis: transverse range is interior y only (no y-halo)
    rl_x = Int32[coord_to_quad(s, ax, b) for ax in 1:hw_,                          b in (hs_ + 1):(hs_ + nyqr_)][:]
    sl_x = Int32[coord_to_quad(s, ax, b) for ax in (hw_ + 1):(2hw_),               b in (hs_ + 1):(hs_ + nyqr_)][:]

    sh_x = Int32[coord_to_quad(s, ax, b) for ax in (nxq_ - 2he_ + 1):(nxq_ - he_), b in (hs_ + 1):(hs_ + nyqr_)][:]
    rh_x = Int32[coord_to_quad(s, ax, b) for ax in (nxq_ - he_ + 1):nxq_,          b in (hs_ + 1):(hs_ + nyqr_)][:]

    # Y-axis: transverse range is full x including x-halo (corners now covered)
    rl_y = Int32[coord_to_quad(s, b, ax) for ax in 1:hs_,                          b in 1:nxq_][:]
    sl_y = Int32[coord_to_quad(s, b, ax) for ax in (hs_ + 1):(2hs_),               b in 1:nxq_][:]

    sh_y = Int32[coord_to_quad(s, b, ax) for ax in (nyq_ - 2hn_ + 1):(nyq_ - hn_), b in 1:nxq_][:]
    rh_y = Int32[coord_to_quad(s, b, ax) for ax in (nyq_ - hn_ + 1):nyq_,          b in 1:nxq_][:]

    send = [sl_x, sh_x, sl_y, sh_y]
    recv = [rl_x, rh_x, rl_y, rh_y]

    return GhostRegion{Quad,2}(send, recv)
end

# TODO: Test me!
function GhostRegion(::Type{Datum{Quad,3}}, s::AbstractCubicalComplex3D)
    hw_  = halo_west(s);  he_  = halo_east(s)
    hs_  = halo_south(s); hn_  = halo_north(s)
    hd_  = halo_down(s);  hu_  = halo_up(s)
    nx_  = nx(s);  ny_  = ny(s);  nz_  = nz(s)
    nxq_ = nxq(s); nyq_ = nyq(s); nzq_ = nzq(s)
    nxr_ = nxr(s); nyr_ = nyr(s); nzr_ = nzr(s)
    nxbr_ = nxbr(s); nybr_ = nybr(s); nzbr_ = nzbr(s)

    # TODO: These seem good but check
    # ── EASTWEST pass (slice in x) ────────────────────────────────────────────
    ew_yt = (hs_ + 1):(hs_ + nybr_)
    ew_zt = (hd_ + 1):(hd_ + nzbr_)

    # Z-aligned (XY): x is tangential (effectively x-edges in 2D)
    sl_z_ew = Int32[coord_to_quad(s, ax, b, c, Z_ALIGN) for ax in (hw_ + 1):(2hw_), b in ew_yt, c in ew_zt][:]
    rh_z_ew = Int32[coord_to_quad(s, ax, b, c, Z_ALIGN) for ax in (nxq_ - he_ + 1):nxq_, b in ew_yt, c in ew_zt][:]
    sh_z_ew = Int32[coord_to_quad(s, ax, b, c, Z_ALIGN) for ax in (nxq_ - 2he_ + 1):(nxq_ - he_), b in ew_yt, c in ew_zt][:]
    rl_z_ew = Int32[coord_to_quad(s, ax, b, c, Z_ALIGN) for ax in 1:hw_, b in ew_yt, c in ew_zt][:]

    # Y-aligned (XZ): x is tangential (effectively y-edges in 2D)
    sl_y_ew = Int32[coord_to_quad(s, ax, b, c, Y_ALIGN) for ax in (hw_ + 1):(2hw_), b in ew_yt, c in ew_zt][:]
    rh_y_ew = Int32[coord_to_quad(s, ax, b, c, Y_ALIGN) for ax in (nxq_ - he_ + 1):nxq_, b in ew_yt, c in ew_zt][:]
    sh_y_ew = Int32[coord_to_quad(s, ax, b, c, Y_ALIGN) for ax in (nxq_ - 2he_ + 1):(nxq_ - he_), b in ew_yt, c in ew_zt][:]
    rl_y_ew = Int32[coord_to_quad(s, ax, b, c, Y_ALIGN) for ax in 1:hw_, b in ew_yt, c in ew_zt][:]

    # X-aligned (YZ): x is the NORMAL axis 
    sl_x_ew = Int32[coord_to_quad(s, ax, b, c, X_ALIGN) for ax in (hw_ + 1):(2hw_ + 1), b in ew_yt, c in ew_zt][:]
    rh_x_ew = Int32[coord_to_quad(s, ax, b, c, X_ALIGN) for ax in (nx_ - he_):nx_, b in ew_yt, c in ew_zt][:]
    sh_x_ew = Int32[coord_to_quad(s, ax, b, c, X_ALIGN) for ax in (nx_ - 2he_):(nx_ - he_ - 1), b in ew_yt, c in ew_zt][:]
    rl_x_ew = Int32[coord_to_quad(s, ax, b, c, X_ALIGN) for ax in 1:hw_, b in ew_yt, c in ew_zt][:]

    sl_ew = vcat(sl_z_ew, sl_y_ew, sl_x_ew)
    rh_ew = vcat(rh_z_ew, rh_y_ew, rh_x_ew)
    sh_ew = vcat(sh_z_ew, sh_y_ew, sh_x_ew)
    rl_ew = vcat(rl_z_ew, rl_y_ew, rl_x_ew)

    # TODO: These seem good but check
    # ── NORTHSOUTH pass (slice in y, full-x transverse) ───────────────────────
    ns_zt   = (hd_ + 1):(hd_ + nzbr_)
    
    # Z-aligned (XY): y is tangential (effectively x-edges in 2D)
    ns_xt_z = 1:nxq_
    sl_z_ns = Int32[coord_to_quad(s, b, ax, c, Z_ALIGN) for ax in (hs_ + 1):(2hs_), b in ns_xt_z, c in ns_zt][:]
    rh_z_ns = Int32[coord_to_quad(s, b, ax, c, Z_ALIGN) for ax in (nyq_ - hn_ + 1):nyq_, b in ns_xt_z, c in ns_zt][:]
    sh_z_ns = Int32[coord_to_quad(s, b, ax, c, Z_ALIGN) for ax in (nyq_ - 2hn_ + 1):(nyq_ - hn_), b in ns_xt_z, c in ns_zt][:]
    rl_z_ns = Int32[coord_to_quad(s, b, ax, c, Z_ALIGN) for ax in 1:hs_, b in ns_xt_z, c in ns_zt][:]

    # Y-aligned (XZ): y is the NORMAL axis
    ns_xt_y = 1:nxq_
    sl_y_ns = Int32[coord_to_quad(s, b, ax, c, Y_ALIGN) for ax in (hs_ + 1):(2hs_ + 1), b in ns_xt_y, c in ns_zt][:]
    rh_y_ns = Int32[coord_to_quad(s, b, ax, c, Y_ALIGN) for ax in (ny_ - hn_):ny_, b in ns_xt_y, c in ns_zt][:]
    sh_y_ns = Int32[coord_to_quad(s, b, ax, c, Y_ALIGN) for ax in (ny_ - 2hn_):(ny_ - hn_ - 1), b in ns_xt_y, c in ns_zt][:]
    rl_y_ns = Int32[coord_to_quad(s, b, ax, c, Y_ALIGN) for ax in 1:hs_, b in ns_xt_y, c in ns_zt][:]

    # X-aligned (YZ): y is tangential (effectively y-edges in 2D)
    ns_xt_x = 1:nx_
    sl_x_ns = Int32[coord_to_quad(s, b, ax, c, X_ALIGN) for ax in (hs_ + 1):(2hs_), b in ns_xt_x, c in ns_zt][:]
    rh_x_ns = Int32[coord_to_quad(s, b, ax, c, X_ALIGN) for ax in (nyq_ - hn_ + 1):nyq_, b in ns_xt_x, c in ns_zt][:]
    sh_x_ns = Int32[coord_to_quad(s, b, ax, c, X_ALIGN) for ax in (nyq_ - 2hn_ + 1):(nyq_ - hn_), b in ns_xt_x, c in ns_zt][:]
    rl_x_ns = Int32[coord_to_quad(s, b, ax, c, X_ALIGN) for ax in 1:hs_, b in ns_xt_x, c in ns_zt][:]

    sl_ns = vcat(sl_z_ns, sl_y_ns, sl_x_ns)
    rh_ns = vcat(rh_z_ns, rh_y_ns, rh_x_ns)
    sh_ns = vcat(sh_z_ns, sh_y_ns, sh_x_ns)
    rl_ns = vcat(rl_z_ns, rl_y_ns, rl_x_ns)

    # ── UPDOWN pass (slice in z, full-x and full-y transverse) ────────────────
    
    # Z-aligned (XY): z is the NORMAL axis
    ud_xt_z = 1:nxq_
    ud_yt_z = 1:nyq_
    sl_z_ud = Int32[coord_to_quad(s, b, c, ax, Z_ALIGN) for ax in (hd_ + 1):(2hd_ + 1), b in ud_xt_z, c in ud_yt_z][:]
    rh_z_ud = Int32[coord_to_quad(s, b, c, ax, Z_ALIGN) for ax in (nz_ - hu_):nz_, b in ud_xt_z, c in ud_yt_z][:]
    sh_z_ud = Int32[coord_to_quad(s, b, c, ax, Z_ALIGN) for ax in (nz_ - 2hu_):(nz_ - hu_ - 1), b in ud_xt_z, c in ud_yt_z][:]
    rl_z_ud = Int32[coord_to_quad(s, b, c, ax, Z_ALIGN) for ax in 1:hd_, b in ud_xt_z, c in ud_yt_z][:]

    # Y-aligned (XZ): z is tangential (effectively x-edges in 2D)
    ud_xt_y = 1:nxq_
    ud_yt_y = 1:ny_
    sl_y_ud = Int32[coord_to_quad(s, b, c, ax, Y_ALIGN) for ax in (hd_ + 1):(2hd_), b in ud_xt_y, c in ud_yt_y][:]
    rh_y_ud = Int32[coord_to_quad(s, b, c, ax, Y_ALIGN) for ax in (nzq_ - hu_ + 1):nzq_, b in ud_xt_y, c in ud_yt_y][:]
    sh_y_ud = Int32[coord_to_quad(s, b, c, ax, Y_ALIGN) for ax in (nzq_ - 2hu_ + 1):(nzq_ - hu_), b in ud_xt_y, c in ud_yt_y][:]
    rl_y_ud = Int32[coord_to_quad(s, b, c, ax, Y_ALIGN) for ax in 1:hd_, b in ud_xt_y, c in ud_yt_y][:]

    # X-aligned (YZ): z is tangential (effectively y-edges in 2D)
    ud_xt_x = 1:nx_
    ud_yt_x = 1:nyq_
    sl_x_ud = Int32[coord_to_quad(s, b, c, ax, X_ALIGN) for ax in (hd_ + 1):(2hd_), b in ud_xt_x, c in ud_yt_x][:]
    rh_x_ud = Int32[coord_to_quad(s, b, c, ax, X_ALIGN) for ax in (nzq_ - hu_ + 1):nzq_, b in ud_xt_x, c in ud_yt_x][:]
    sh_x_ud = Int32[coord_to_quad(s, b, c, ax, X_ALIGN) for ax in (nzq_ - 2hu_ + 1):(nzq_ - hu_), b in ud_xt_x, c in ud_yt_x][:]
    rl_x_ud = Int32[coord_to_quad(s, b, c, ax, X_ALIGN) for ax in 1:hd_, b in ud_xt_x, c in ud_yt_x][:]

    sl_ud = vcat(sl_z_ud, sl_y_ud, sl_x_ud)
    rh_ud = vcat(rh_z_ud, rh_y_ud, rh_x_ud)
    sh_ud = vcat(sh_z_ud, sh_y_ud, sh_x_ud)
    rl_ud = vcat(rl_z_ud, rl_y_ud, rl_x_ud)

    send = [sl_ew, sh_ew, sl_ns, sh_ns, sl_ud, sh_ud]
    recv = [rl_ew, rh_ew, rl_ns, rh_ns, rl_ud, rh_ud]

    return GhostRegion{Quad,3}(send, recv)
end

function GhostRegion(::Type{Datum{Boid,3}}, s::AbstractCubicalComplex3D)
    hw_ = halo_west(s);  he_ = halo_east(s)
    hs_ = halo_south(s); hn_ = halo_north(s)
    hd_ = halo_down(s);  hu_ = halo_up(s)
    nxb_  = nxb(s)
    nyb_  = nyb(s)
    nzb_  = nzb(s)
    nxbr_ = nxbr(s)
    nybr_ = nybr(s)
    nzbr_ = nzbr(s)

    # X-axis: transverse is real y and real z only
    rl_x = Int32[coord_to_boid(s, ax, b, c) for ax in 1:hw_,                            b in (hs_ + 1):(hs_ + nybr_), c in (hd_ + 1):(hd_ + nzbr_)][:]
    sl_x = Int32[coord_to_boid(s, ax, b, c) for ax in (hw_ + 1):(2hw_),                 b in (hs_ + 1):(hs_ + nybr_), c in (hd_ + 1):(hd_ + nzbr_)][:]
    sh_x = Int32[coord_to_boid(s, ax, b, c) for ax in (nxb_ - 2he_ + 1):(nxb_ - he_),  b in (hs_ + 1):(hs_ + nybr_), c in (hd_ + 1):(hd_ + nzbr_)][:]
    rh_x = Int32[coord_to_boid(s, ax, b, c) for ax in (nxb_ - he_ + 1):nxb_,           b in (hs_ + 1):(hs_ + nybr_), c in (hd_ + 1):(hd_ + nzbr_)][:]

    # Y-axis: transverse is full x (x-halo filled), real z only
    rl_y = Int32[coord_to_boid(s, b, ax, c) for ax in 1:hs_,                            b in 1:nxb_, c in (hd_ + 1):(hd_ + nzbr_)][:]
    sl_y = Int32[coord_to_boid(s, b, ax, c) for ax in (hs_ + 1):(2hs_),                 b in 1:nxb_, c in (hd_ + 1):(hd_ + nzbr_)][:]
    sh_y = Int32[coord_to_boid(s, b, ax, c) for ax in (nyb_ - 2hn_ + 1):(nyb_ - hn_),  b in 1:nxb_, c in (hd_ + 1):(hd_ + nzbr_)][:]
    rh_y = Int32[coord_to_boid(s, b, ax, c) for ax in (nyb_ - hn_ + 1):nyb_,           b in 1:nxb_, c in (hd_ + 1):(hd_ + nzbr_)][:]

    # Z-axis: transverse is full x and full y (both halos filled)
    rl_z = Int32[coord_to_boid(s, b, c, ax) for ax in 1:hd_,                            b in 1:nxb_, c in 1:nyb_][:]
    sl_z = Int32[coord_to_boid(s, b, c, ax) for ax in (hd_ + 1):(2hd_),                 b in 1:nxb_, c in 1:nyb_][:]
    sh_z = Int32[coord_to_boid(s, b, c, ax) for ax in (nzb_ - 2hu_ + 1):(nzb_ - hu_),  b in 1:nxb_, c in 1:nyb_][:]
    rh_z = Int32[coord_to_boid(s, b, c, ax) for ax in (nzb_ - hu_ + 1):nzb_,           b in 1:nxb_, c in 1:nyb_][:]

    send = [sl_x, sh_x, sl_y, sh_y, sl_z, sh_z]
    recv = [rl_x, rh_x, rl_y, rh_y, rl_z, rh_z]

    return GhostRegion{Boid,3}(send, recv)
end

# Left/bottom edges are real and are exchanged with right/top which are halo
function GhostRegion(::Type{Datum{Edge,2}}, s::AbstractCubicalComplex2D)
    hw_ = halo_west(s);  he_ = halo_east(s)
    hs_ = halo_south(s); hn_ = halo_north(s)
    nx_   = nx(s)
    ny_   = ny(s)
    nxe_  = nxe(s)
    nye_  = nye(s)
    nxqr_ = nxqr(s)
    nyqr_ = nyqr(s)

    # ── EASTWEST pass (slice in x, interior-y transverse only) ────────────────
    # X-edges
    x_ew_yt = (hs_ + 1):(hs_ + nyqr_)

    sl_x_ew = Int32[coord_to_edge(s, ax, b, X_ALIGN) for ax in (hw_ + 1):(2hw_),         b in x_ew_yt][:]
    rh_x_ew = Int32[coord_to_edge(s, ax, b, X_ALIGN) for ax in (nxe_ - he_ + 1):nxe_,   b in x_ew_yt][:]
    sh_x_ew = Int32[coord_to_edge(s, ax, b, X_ALIGN) for ax in (nxe_ - 2he_ + 1):(nxe_ - he_), b in x_ew_yt][:]
    rl_x_ew = Int32[coord_to_edge(s, ax, b, X_ALIGN) for ax in 1:hw_,                    b in x_ew_yt][:]

    # Y-edges (normal axis: +1 extension on recv/send)
    y_ew_yt = (hs_ + 1):(hs_ + nyqr_)

    sl_y_ew = Int32[coord_to_edge(s, ax, b, Y_ALIGN) for ax in (hw_ + 1):(2hw_ + 1),     b in y_ew_yt][:]
    rh_y_ew = Int32[coord_to_edge(s, ax, b, Y_ALIGN) for ax in (nx_ - he_):nx_,           b in y_ew_yt][:]
    sh_y_ew = Int32[coord_to_edge(s, ax, b, Y_ALIGN) for ax in (nx_ - 2he_):(nx_ - he_ - 1), b in y_ew_yt][:]
    rl_y_ew = Int32[coord_to_edge(s, ax, b, Y_ALIGN) for ax in 1:hw_,                    b in y_ew_yt][:]

    sl_ew = vcat(sl_x_ew, sl_y_ew)
    rl_ew = vcat(rl_x_ew, rl_y_ew)
    sh_ew = vcat(sh_x_ew, sh_y_ew)
    rh_ew = vcat(rh_x_ew, rh_y_ew)

    # ── NORTHSOUTH pass (slice in y, full-x transverse) ───────────────────────
    # Y-edges
    sl_y_ns = Int32[coord_to_edge(s, b, ax, Y_ALIGN) for ax in (hs_ + 1):(2hs_),         b in 1:nx_][:]
    rh_y_ns = Int32[coord_to_edge(s, b, ax, Y_ALIGN) for ax in (nye_ - hn_ + 1):nye_,    b in 1:nx_][:]
    sh_y_ns = Int32[coord_to_edge(s, b, ax, Y_ALIGN) for ax in (nye_ - 2hn_ + 1):(nye_ - hn_), b in 1:nx_][:]
    rl_y_ns = Int32[coord_to_edge(s, b, ax, Y_ALIGN) for ax in 1:hs_,                    b in 1:nx_][:]

    # X-edges (normal axis: +1 extension on recv/send)
    sl_x_ns = Int32[coord_to_edge(s, b, ax, X_ALIGN) for ax in (hs_ + 1):(2hs_ + 1),    b in 1:nxe_][:]
    rh_x_ns = Int32[coord_to_edge(s, b, ax, X_ALIGN) for ax in (ny_ - hn_):ny_,          b in 1:nxe_][:]
    sh_x_ns = Int32[coord_to_edge(s, b, ax, X_ALIGN) for ax in (ny_ - 2hn_):(ny_ - hn_ - 1), b in 1:nxe_][:]
    rl_x_ns = Int32[coord_to_edge(s, b, ax, X_ALIGN) for ax in 1:hs_,                    b in 1:nxe_][:]

    sl_ns = vcat(sl_y_ns, sl_x_ns)
    rl_ns = vcat(rl_y_ns, rl_x_ns)
    sh_ns = vcat(sh_y_ns, sh_x_ns)
    rh_ns = vcat(rh_y_ns, rh_x_ns)

    send = [sl_ew, sh_ew, sl_ns, sh_ns]
    recv = [rl_ew, rh_ew, rl_ns, rh_ns]

    return GhostRegion{Edge,2}(send, recv)
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

# TODO: These stubs don't work if included in the data stream
GhostRegion(::Type{Datum{Vert,3}}, s) = GhostRegion{Vert,3}(Int32[], Int32[])   # stub
GhostRegion(::Type{Datum{Edge,N}}, s) where {N} = GhostRegion{Edge,N}(Int32[], Int32[])   # stub

send_slab(g::GhostRegion, face::Face) = send_slab(g, Int(face))
recv_slab(g::GhostRegion, face::Face) = recv_slab(g, Int(face))

send_slab(g::GhostRegion, i::Int) = g.send[i]
recv_slab(g::GhostRegion, i::Int) = g.recv[i]

### ExchangeHandler ###

# Vector over each Datum
struct FaceBuffer
    slabs    :: AbstractVector{AbstractVector{Int32}}        # CPU index arrays, host-side only
    cell_lens:: AbstractVector{Int}
end

function Adapt.adapt_structure(backend, fb::FaceBuffer)
    FaceBuffer(
        map(s -> adapt(backend, s), fb.slabs),
        fb.cell_lens,
    )
end

function FaceBuffer(ghosts::AbstractVector, stream::DataStream, face::Face, sendrecv::Symbol, backend = CPU())
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
    send_face::AbstractVector{FaceBuffer}   # length 2N
    recv_face::AbstractVector{FaceBuffer}   # length 2N

    send_bufs::AbstractVector{AbstractVector{FT}}   # length 2N
    recv_bufs::AbstractVector{AbstractVector{FT}}   # length 2N

    send_reqs::AbstractVector{MPI.Request}  # length 2N
    recv_reqs::AbstractVector{MPI.Request}  # length 2N
end

const AXIS_NAMES_2D = (:west, :east, :south, :north)
const AXIS_NAMES_3D = (:west, :east, :south, :north, :down, :up)

# TODO: Having the buffers be GPUArrays means we need GPU-Aware MPI
function ExchangeHandler(stream::DataStream, topo::MPITopology{WorkerCache{N}}, s::AbstractCubicalComplex; backend = CPU()) where {N}
    @assert !isempty(stream.data) "ExchangeHandler: stream must contain at least one datum"

    FT = stream.data[1].entrytype
    @assert all(d.entrytype == FT for d in stream.data) "ExchangeHandler: all datums must share the same entrytype, got $(unique(d.entrytype for d in stream.data))"

    ghosts = map(1:(N + 1)) do i
        elemtype = meshtype_index(Val(i))
        gr = GhostRegion(Datum{elemtype,N}, s)  # lower slots stubbed
        return adapt(backend, gr)
    end

    face_names = N == 2 ? AXIS_NAMES_2D : AXIS_NAMES_3D
    faces = Face.(1:(2N))

    # FaceBuffer for each face (info for packing/unpacking) (all datums covered)
    send_face = [adapt(backend, FaceBuffer(ghosts, stream, face, :send)) for face in faces]
    recv_face = [adapt(backend, FaceBuffer(ghosts, stream, face, :recv)) for face in faces]

    # Buffer size for each face
    send_buf_sizes = [sum(fb.cell_lens) for fb in send_face]
    recv_buf_sizes = [sum(fb.cell_lens) for fb in recv_face]

    # Send and recv buffers for each face
    send_bufs = map(i -> KernelAbstractions.zeros(backend, FT, send_buf_sizes[i]), 1:(2N))
    recv_bufs = map(i -> KernelAbstractions.zeros(backend, FT, recv_buf_sizes[i]), 1:(2N))

    nb = topo.cache.neighbors

    # Tag the communication with the source's face

    # Send
    # W -> 0, E -> 1, S -> 2, N -> 3, D -> 4, U -> 5

    # Recv
    # W -> 1, E -> 0, S -> 3, N -> 2, D -> 5, U -> 4

    low_faces = Face.(1:2:(2N))
    high_faces = Face.(2:2:(2N))

    send_reqs = Vector{MPI.Request}(undef, 2N)
    recv_reqs = Vector{MPI.Request}(undef, 2N)

    for (low, high) in zip(low_faces, high_faces)
        il = Int(low)
        ih = Int(high)
        send_reqs[il] = MPI.Send_init(send_bufs[il], nb[face_names[il]], il, topo.cart_comm)
        recv_reqs[ih] = MPI.Recv_init(recv_bufs[ih], nb[face_names[ih]], il, topo.cart_comm)

        send_reqs[ih] = MPI.Send_init(send_bufs[ih], nb[face_names[ih]], ih, topo.cart_comm)
        recv_reqs[il] = MPI.Recv_init(recv_bufs[il], nb[face_names[il]], ih, topo.cart_comm)
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

    _unpack_face!(fields, recv_face(handler, low), recv_buf(handler, low))
    _unpack_face!(fields, recv_face(handler, high), recv_buf(handler, high))

    MPI.Wait(send_req(handler, low))
    MPI.Wait(send_req(handler, high))

    return nothing
end

# TODO: For benchmark testing
function test_exchange!(handler::ExchangeHandler{N,FT}, vars::NamedTuple) where {N,FT}
    fields = map(datum -> vars[Symbol(datum.name)], handler.stream.data)

    _test_exchange_axis!(handler, fields, EASTWEST)
    _test_exchange_axis!(handler, fields, NORTHSOUTH)
    N == 3 && _test_exchange_axis!(handler, fields, UPDOWN)

    return nothing
end

function _test_exchange_axis!(handler::ExchangeHandler{N,FT}, fields::AbstractVector, side::GridSide) where {N,FT}
    low  = low_face(side)
    high = high_face(side)
    il   = Int(low)
    ih   = Int(high)

    nb         = handler.topo.cache.neighbors
    face_names = N == 2 ? AXIS_NAMES_2D : AXIS_NAMES_3D

    _pack_face!(handler.send_bufs[il], handler.send_face[il], fields)
    _pack_face!(handler.send_bufs[ih], handler.send_face[ih], fields)

    MPI.Sendrecv!(handler.send_bufs[il], nb[face_names[il]], il,
                  handler.recv_bufs[ih], nb[face_names[ih]], il,
                  handler.topo.cart_comm)

    MPI.Sendrecv!(handler.send_bufs[ih], nb[face_names[ih]], ih,
                  handler.recv_bufs[il], nb[face_names[il]], ih,
                  handler.topo.cart_comm)

    _unpack_face!(fields, handler.recv_face[il], handler.recv_bufs[il])
    _unpack_face!(fields, handler.recv_face[ih], handler.recv_bufs[ih])

    return nothing
end

# TODO: Test pack/unpack kernels
# ── GPU gather kernel: pack field values into a contiguous send buffer ────────
@kernel function _gather_kernel!(buf, @Const(slab), @Const(field), offset)
    i = @index(Global)
    @inbounds buf[offset + i] = field[slab[i]]
end

# ── GPU scatter kernel: unpack a recv buffer back into field positions ─────────
@kernel function _scatter_kernel!(field, @Const(slab), @Const(buf), offset)
    i = @index(Global)
    @inbounds field[slab[i]] = buf[offset + i]
end

# ── Pack: replaces scalar _pack_face! loop [7] ────────────────────────────────
function _pack_face!(buf::AbstractVector, fb::FaceBuffer, fields::AbstractVector)
    backend = get_backend(buf)
    offset  = 0
    for (field, slab, cell_len) in zip(fields, fb.slabs, fb.cell_lens)
        if cell_len > 0
            _gather_kernel!(backend)(buf, slab, field, offset;
                                     ndrange = cell_len)
        end
        offset += cell_len
    end
    KernelAbstractions.synchronize(backend)
    return nothing
end

# ── Unpack: replaces scalar _unpack_face! loop [7] ───────────────────────────
function _unpack_face!(fields::AbstractVector, fb::FaceBuffer, buf::AbstractVector)
    backend = get_backend(buf)
    offset  = 0
    for (field, slab, cell_len) in zip(fields, fb.slabs, fb.cell_lens)
        if cell_len > 0
            _scatter_kernel!(backend)(field, slab, buf, offset;
                                      ndrange = cell_len)
        end
        offset += cell_len
    end
    KernelAbstractions.synchronize(backend)
    return nothing
end

# function _pack_face!(buf::Vector, fb::FaceBuffer, fields::AbstractVector)
#     offset = 0
#     for (field, slab, cell_len) in zip(fields, fb.slabs, fb.cell_lens)
#         for k in 1:cell_len
#             buf[offset + k] = field[slab[k]]
#         end
#         offset += cell_len
#     end
#     return nothing
# end

# function _unpack_face!(fields::AbstractVector, fb::FaceBuffer, buf::Vector)
#     offset = 0
#     for (field, slab, cell_len) in zip(fields, fb.slabs, fb.cell_lens)
#         for k in 1:cell_len
#             field[slab[k]] = buf[offset + k]
#         end
#         offset += cell_len
#     end
#     return nothing
# end

function close!(handler::ExchangeHandler{N,FT}) where {N,FT}
    for i in faces(handler)
        MPI.free(handler.send_reqs[i])
        MPI.free(handler.recv_reqs[i])
    end
    return nothing
end