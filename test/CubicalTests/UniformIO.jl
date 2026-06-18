using Test

include(joinpath(@__DIR__, "..", "..", "src", "CubicalCode", "UniformMPI.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "CubicalCode", "UniformIO.jl"))

# ── Shared test meshes ────────────────────────────────────────────────────────

const m2 = PseudoCubicalMesh2D(10, 8)
const m3 = PseudoCubicalMesh3D(10, 8, 6)

@testset "datum_dims" begin

    @testset "Vert" begin
        @test datum_dims(Datum{Vert,2}(), m2) == [(10, 8)]
        @test datum_dims(Datum{Vert,3}(), m3) == [(10, 8, 6)]
    end

    @testset "Boid" begin
        @test datum_dims(Datum{Boid,3}(), m3) == [(9, 7, 5)]
    end

    @testset "Quad 2D" begin
        @test datum_dims(Datum{Quad,2}(), m2) == [(9, 7)]
    end

    @testset "Quad 3D" begin
        @test datum_dims(Datum{Quad,3}(), m3) == [(9, 7, 6), (9, 8, 5), (10, 7, 5)]
    end

    @testset "Edge 2D" begin
        @test datum_dims(Datum{Edge,2}(), m2) == [(9, 8), (10, 7)]
    end

    @testset "Edge 3D" begin
        @test datum_dims(Datum{Edge,3}(), m3) == [(9, 8, 6), (10, 7, 6), (10, 8, 5)]
    end
end

@testset "mesh_count" begin
    @test mesh_count(Datum{Vert,2}(), m2) == 10 * 8
    @test mesh_count(Datum{Vert,3}(), m3) == 10 * 8 * 6
    @test mesh_count(Datum{Boid,3}(), m3) == 9 * 7 * 5
    @test mesh_count(Datum{Quad,2}(), m2) == 9 * 7
    @test mesh_count(Datum{Quad,3}(), m3) == 9*7*6 + 9*8*5 + 10*7*5
    @test mesh_count(Datum{Edge,2}(), m2) == 9*8 + 10*7
    @test mesh_count(Datum{Edge,3}(), m3) == 9*8*6 + 10*7*6 + 10*8*5

    for datum in [Datum{Vert,2}(), Datum{Quad,2}(), Datum{Edge,2}()]
        @test mesh_count(datum, m2) == sum(prod(d) for d in datum_dims(datum, m2))
    end
    for datum in [Datum{Vert,3}(), Datum{Boid,3}(), Datum{Quad,3}(), Datum{Edge,3}()]
        @test mesh_count(datum, m3) == sum(prod(d) for d in datum_dims(datum, m3))
    end
end

@testset "tile_buffer" begin
    function check_bufs(datum, mesh)
        bufs  = tile_buffer(datum, mesh)
        ddims = datum_dims(datum, mesh)
        @test length(bufs) == length(ddims)
        for (buf, dims) in zip(bufs, ddims)
            @test size(buf) == dims
            @test eltype(buf) == datum.entrytype
        end
    end

    for datum in [Datum{Vert,2}(), Datum{Quad,2}(), Datum{Edge,2}()]
        check_bufs(datum, m2)
    end
    for datum in [Datum{Vert,3}(), Datum{Boid,3}(), Datum{Quad,3}(), Datum{Edge,3}()]
        check_bufs(datum, m3)
    end
end

# ── _build_gatherv_counts ─────────────────────────────────────────────────────

@testset "_build_gatherv_counts" begin
    wcs = [
        OutputWorkerCache((5, 4), (0, 0)),
        OutputWorkerCache((5, 6), (0, 3)),
        OutputWorkerCache((7, 4), (4, 0)),
        OutputWorkerCache((7, 6), (4, 3)),
    ]
    datum = Datum{Quad,2}()
    counts, displs = _build_gatherv_counts(datum, wcs)

    @test counts == Cint[4*3, 4*5, 6*3, 6*5]
    @test displs == Cint[0, 12, 32, 50]
    @test length(counts) == length(wcs)
    @test displs[1] == 0
    @test all(displs[i] == sum(counts[1:i-1]) for i in 2:length(wcs))
end

# ── _scatter_worker_chunk! ────────────────────────────────────────────────────

@testset "_scatter_worker_chunk!" begin
    wc     = OutputWorkerCache((5, 4), (0, 0))
    datum  = Datum{Quad,2}()
    om_dims = (9, 7)
    tbuf   = zeros(Float64, om_dims...)
    recv   = Float64.(collect(1:mesh_count(datum, wc.mesh)))

    _scatter_worker_chunk!(tbuf, recv, wc, (0, 0), 0, datum)

    @test tbuf[1:4, 1:3] == reshape(recv, 4, 3)
    @test all(tbuf[5:end, :] .== 0)
    @test all(tbuf[:, 4:end] .== 0)
end

# ── scatter_to_tile! ──────────────────────────────────────────────────────────

@testset "scatter_to_tile!" begin
    wcs = [
        OutputWorkerCache((5, 4), (0, 0)),
        OutputWorkerCache((5, 4), (0, 3)),
        OutputWorkerCache((5, 4), (4, 0)),
        OutputWorkerCache((5, 4), (4, 3)),
    ]
    datum   = Datum{Quad,2}()
    cache = OutputCache{2}(length(wcs), wcs, [(0,0),(0,3),(4,0),(4,3)], (2,2), (9,7), (0,0), PseudoCubicalMesh(9, 7))
    handler = DataHandler(DataStream(datum), cache)

    recv = handler.gatherv_caches[1].recv_buffer
    for (i, wc) in enumerate(wcs)
        n = mesh_count(datum, wc.mesh)
        recv[(i-1)*n+1 : i*n] .= Float64(i)
    end

    scatter_to_tile!(datum, handler.gatherv_caches[1], handler.tile_buffers[1], handler)

    tbuf = handler.tile_buffers[1]
    @test only(tbuf)[1:4, 1:3] == fill(1.0, 4, 3)
    @test only(tbuf)[1:4, 4:6] == fill(2.0, 4, 3)
    @test only(tbuf)[5:8, 1:3] == fill(3.0, 4, 3)
    @test only(tbuf)[5:8, 4:6] == fill(4.0, 4, 3)
    @test only(tbuf) == [
        1.0 1.0 1.0 2.0 2.0 2.0;
        1.0 1.0 1.0 2.0 2.0 2.0;
        1.0 1.0 1.0 2.0 2.0 2.0;
        1.0 1.0 1.0 2.0 2.0 2.0;
        3.0 3.0 3.0 4.0 4.0 4.0;
        3.0 3.0 3.0 4.0 4.0 4.0;
        3.0 3.0 3.0 4.0 4.0 4.0;
        3.0 3.0 3.0 4.0 4.0 4.0;
    ]
end

@testset "scatter_to_tile! unequal worker sizes" begin
    wcs = [
        OutputWorkerCache((6, 5), (0, 0)),
        OutputWorkerCache((6, 4), (0, 4)),
        OutputWorkerCache((4, 5), (5, 0)),
        OutputWorkerCache((4, 4), (5, 4)),
    ]
    datum   = Datum{Quad,2}()
    cache = OutputCache{2}(length(wcs), wcs, [(0,0),(0,4),(5,0),(5,4)],
                           (2,2), (9,8), (0,0), PseudoCubicalMesh(9, 8))
    handler = DataHandler(DataStream(datum), cache)

    recv       = handler.gatherv_caches[1].recv_buffer
    src_starts = [0; cumsum([mesh_count(datum, wc.mesh) for wc in wcs])[1:end-1]]
    for (i, (wc, src_start)) in enumerate(zip(wcs, src_starts))
        n = mesh_count(datum, wc.mesh)
        recv[src_start+1 : src_start+n] .= Float64(i)
    end

    scatter_to_tile!(datum, handler.gatherv_caches[1], handler.tile_buffers[1], handler)

    tbuf = only(handler.tile_buffers[1])
    @test size(tbuf) == (8, 7)
    @test tbuf[1:5, 1:4] == fill(1.0, 5, 4)
    @test tbuf[1:5, 5:7] == fill(2.0, 5, 3)
    @test tbuf[6:8, 1:4] == fill(3.0, 3, 4)
    @test tbuf[6:8, 5:7] == fill(4.0, 3, 3)
    @test tbuf == [
        1.0 1.0 1.0 1.0 2.0 2.0 2.0
        1.0 1.0 1.0 1.0 2.0 2.0 2.0
        1.0 1.0 1.0 1.0 2.0 2.0 2.0
        1.0 1.0 1.0 1.0 2.0 2.0 2.0
        1.0 1.0 1.0 1.0 2.0 2.0 2.0
        3.0 3.0 3.0 3.0 4.0 4.0 4.0
        3.0 3.0 3.0 3.0 4.0 4.0 4.0
        3.0 3.0 3.0 3.0 4.0 4.0 4.0
    ]
end

# ── _hyperslab_ranges ─────────────────────────────────────────────────────────

@testset "_hyperslab_ranges" begin
    wcs  = [OutputWorkerCache((9, 7), (0, 0))]
    om   = PseudoCubicalMesh(9, 7)
    cache = OutputCache{2}(1, wcs, [(0,0)], (1,1), (9,7), (3, 5), om)

    ranges = _hyperslab_ranges(cache, (8, 6))
    @test ranges == (4:11, 6:11)

    ranges0 = _hyperslab_ranges(cache, (8, 6))
    @test ranges0 == (4:11, 6:11)
end