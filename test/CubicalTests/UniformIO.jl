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
    @test mesh_count(Datum{Quad,3}(), m3) == 9 * 7 * 6 + 9 * 8 * 5 + 10 * 7 * 5
    @test mesh_count(Datum{Edge,2}(), m2) == 9 * 8 + 10 * 7
    @test mesh_count(Datum{Edge,3}(), m3) == 9 * 8 * 6 + 10 * 7 * 6 + 10 * 8 * 5

    for datum in [Datum{Vert,2}(), Datum{Quad,2}(), Datum{Edge,2}()]
        @test mesh_count(datum, m2) == sum(prod(d) for d in datum_dims(datum, m2))
    end
    for datum in [Datum{Vert,3}(), Datum{Boid,3}(), Datum{Quad,3}(), Datum{Edge,3}()]
        @test mesh_count(datum, m3) == sum(prod(d) for d in datum_dims(datum, m3))
    end
end

@testset "tile_buffer" begin
    function check_bufs(datum, mesh)
        bufs = tile_buffer(datum, mesh)
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
    wcs = [OutputWorkerCache((5, 4), (0, 0)), OutputWorkerCache((5, 6), (0, 3)), OutputWorkerCache((7, 4), (4, 0)), OutputWorkerCache((7, 6), (4, 3))]
    datum = Datum{Quad,2}()
    counts, displs = _build_gatherv_counts(datum, wcs)

    @test counts == Cint[4 * 3, 4 * 5, 6 * 3, 6 * 5]
    @test displs == Cint[0, 12, 32, 50]
    @test length(counts) == length(wcs)
    @test displs[1] == 0
    @test all(displs[i] == sum(counts[1:(i - 1)]) for i in 2:length(wcs))
end

# ── _scatter_worker_chunk! ────────────────────────────────────────────────────

# TODO: Update this test
# @testset "_scatter_worker_chunk!" begin
#     wc = OutputWorkerCache((5, 4), (0, 0))
#     datum = Datum{Quad,2}()
#     om_dims = (9, 7)
#     tbuf = zeros(Float64, om_dims...)
#     recv = Float64.(collect(1:mesh_count(datum, wc.mesh)))

#     _scatter_worker_chunk!(tbuf, recv, wc, (0, 0), 0, datum)

#     @test tbuf[1:4, 1:3] == reshape(recv, 4, 3)
#     @test all(tbuf[5:end, :] .== 0)
#     @test all(tbuf[:, 4:end] .== 0)
# end

# ── scatter_to_tile! ──────────────────────────────────────────────────────────

@testset "scatter_to_tile!" begin
    wcs = [OutputWorkerCache((5, 4), (0, 0)), OutputWorkerCache((5, 4), (0, 3)), OutputWorkerCache((5, 4), (4, 0)), OutputWorkerCache((5, 4), (4, 3))]
    datum = Datum{Quad,2}()
    cache = OutputCache{2}(length(wcs), wcs, [(0, 0), (0, 3), (4, 0), (4, 3)], (2, 2), (9, 7), (0, 0), PseudoCubicalMesh(9, 7))
    handler = DataHandler(DataStream(datum), cache)

    recv = handler.gatherv_caches[1].recv_buffer
    for (i, wc) in enumerate(wcs)
        n = mesh_count(datum, wc.mesh)
        recv[((i - 1) * n + 1):(i * n)] .= Float64(i)
    end

    scatter_to_tile!(datum, handler.gatherv_caches[1], handler.tile_buffers[1], handler)

    tbuf = handler.tile_buffers[1]
    @test only(tbuf)[1:4, 1:3] == fill(1.0, 4, 3)
    @test only(tbuf)[1:4, 4:6] == fill(2.0, 4, 3)
    @test only(tbuf)[5:8, 1:3] == fill(3.0, 4, 3)
    @test only(tbuf)[5:8, 4:6] == fill(4.0, 4, 3)
    @test only(tbuf) == [
        1.0 1.0 1.0 2.0 2.0 2.0
        1.0 1.0 1.0 2.0 2.0 2.0
        1.0 1.0 1.0 2.0 2.0 2.0
        1.0 1.0 1.0 2.0 2.0 2.0
        3.0 3.0 3.0 4.0 4.0 4.0
        3.0 3.0 3.0 4.0 4.0 4.0
        3.0 3.0 3.0 4.0 4.0 4.0
        3.0 3.0 3.0 4.0 4.0 4.0
    ]
end

@testset "scatter_to_tile! unequal worker sizes" begin
    wcs = [OutputWorkerCache((6, 5), (0, 0)), OutputWorkerCache((6, 4), (0, 4)), OutputWorkerCache((4, 5), (5, 0)), OutputWorkerCache((4, 4), (5, 4))]
    datum = Datum{Quad,2}()
    cache = OutputCache{2}(length(wcs), wcs, [(0, 0), (0, 4), (5, 0), (5, 4)], (2, 2), (9, 8), (0, 0), PseudoCubicalMesh(9, 8))
    handler = DataHandler(DataStream(datum), cache)

    recv = handler.gatherv_caches[1].recv_buffer
    src_starts = [0; cumsum([mesh_count(datum, wc.mesh) for wc in wcs])[1:(end - 1)]]
    for (i, (wc, src_start)) in enumerate(zip(wcs, src_starts))
        n = mesh_count(datum, wc.mesh)
        recv[(src_start + 1):(src_start + n)] .= Float64(i)
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
    wcs = [OutputWorkerCache((9, 7), (0, 0))]
    om = PseudoCubicalMesh(9, 7)
    cache = OutputCache{2}(1, wcs, [(0, 0)], (1, 1), (9, 7), (3, 5), om)

    ranges = _hyperslab_ranges(cache, (8, 6))
    @test ranges == (4:11, 6:11)

    ranges0 = _hyperslab_ranges(cache, (8, 6))
    @test ranges0 == (4:11, 6:11)
end

function simulate_axis_exchange!(fa, fb, g::GhostRegion, axis::Int)
    low_recv = recv_slab(g, 2 * axis - 1)
    high_recv = recv_slab(g, 2 * axis)
    low_send = send_slab(g, 2 * axis - 1)
    high_send = send_slab(g, 2 * axis)

    fa[low_recv] .= fb[high_send]
    fa[high_recv] .= fb[low_send]
    return nothing
end

@testset "GhostRegion Quad 2D" begin
    s = UniformCubicalComplex2D(7, 7, 1.0, 1.0; halo_x = 1, halo_y = 1)
    g = GhostRegion(Datum{Quad,2}, s)

    @testset "struct shape" begin
        @test nfaces(g) == 4
        @test length(send_slab(g, 1)) == hxq(s) * nyqr(s)
        @test length(recv_slab(g, 1)) == hxq(s) * nyqr(s)
        @test length(send_slab(g, 2)) == hxq(s) * nyqr(s)
        @test length(recv_slab(g, 2)) == hxq(s) * nyqr(s)
        @test length(send_slab(g, 3)) == hyq(s) * nxq(s)
        @test length(recv_slab(g, 3)) == hyq(s) * nxq(s)
        @test length(send_slab(g, 4)) == hyq(s) * nxq(s)
        @test length(recv_slab(g, 4)) == hyq(s) * nxq(s)
    end

    @testset "no overlap within same axis" begin
        @test isempty(intersect(Set(send_slab(g, 1)), Set(send_slab(g, 2))))
        @test isempty(intersect(Set(recv_slab(g, 1)), Set(recv_slab(g, 2))))
        @test isempty(intersect(Set(send_slab(g, 3)), Set(send_slab(g, 4))))
        @test isempty(intersect(Set(recv_slab(g, 3)), Set(recv_slab(g, 4))))
    end

    @testset "all indices in bounds" begin
        for i in 1:4
            @test all(idx -> 1 <= idx <= nquads(s), send_slab(g, i))
            @test all(idx -> 1 <= idx <= nquads(s), recv_slab(g, i))
        end
    end

    @testset "x-axis exchange" begin
        fa = zeros(Int64, nquads(s))
        fb = map(1:nquads(s)) do q
            x, y = quad_to_coord(s, q)
            return 100 * x + y
        end

        simulate_axis_exchange!(fa, fb, g, 1)

        # Recv slabs match fb's send slabs
        for idx in eachindex(recv_slab(g, 1))
            @test fa[recv_slab(g, 1)[idx]] == fb[send_slab(g, 2)[idx]]
        end
        for idx in eachindex(recv_slab(g, 2))
            @test fa[recv_slab(g, 2)[idx]] == fb[send_slab(g, 1)[idx]]
        end

        # Interior quads untouched
        for y in (hyq(s) + 1):(hyq(s) + nyqr(s)), x in (hxq(s) + 1):(hxq(s) + nxqr(s))
            @test fa[coord_to_quad(s, x, y)] == 0.0
        end
    end

    @testset "y-axis exchange fills corners via full-x transverse" begin
        fa = zeros(Int64, nquads(s))
        fb = map(1:nquads(s)) do q
            x, y = quad_to_coord(s, q)
            return 100 * x + y
        end

        simulate_axis_exchange!(fa, fb, g, 1)
        simulate_axis_exchange!(fa, fb, g, 2)

        # y-axis recv slabs populated
        for idx in eachindex(recv_slab(g, 3))
            @test fa[recv_slab(g, 3)[idx]] == fb[send_slab(g, 4)[idx]]
        end
        for idx in eachindex(recv_slab(g, 4))
            @test fa[recv_slab(g, 4)[idx]] == fb[send_slab(g, 3)[idx]]
        end

        # Interior untouched
        for y in (hyq(s) + 1):(hyq(s) + nyqr(s)), x in (hxq(s) + 1):(hxq(s) + nxqr(s))
            @test fa[coord_to_quad(s, x, y)] == 0.0
        end

        # Corners were zero and must now be nonzero (y-pass used x-halo columns)
        @test fa[coord_to_quad(s, 1, 1)] == 107
        @test fa[coord_to_quad(s, nxq(s), 1)] == 807
        @test fa[coord_to_quad(s, 1, nyq(s))] == 102
        @test fa[coord_to_quad(s, nxq(s), nyq(s))] == 802
    end

    @testset "zero halo returns empty slabs" begin
        s0 = UniformCubicalComplex2D(4, 4, 1.0, 1.0; halo_x = 0, halo_y = 0)
        g0 = GhostRegion(Datum{Quad,2}, s0)
        for i in 1:4
            @test isempty(send_slab(g0, i))
            @test isempty(recv_slab(g0, i))
        end
    end
end

# ── 3D Tests ──────────────────────────────────────────────────────────────────

@testset "GhostRegion Boid 3D" begin
    s = UniformCubicalComplex3D(5, 5, 5, 1.0, 1.0, 1.0; halo_x = 1, halo_y = 1, halo_z = 1)
    g = GhostRegion(Datum{Boid,3}, s)

    @testset "struct shape" begin
        @test nfaces(g) == 6
        @test length(send_slab(g, 1)) == hxb(s) * nybr(s) * nzbr(s)
        @test length(recv_slab(g, 1)) == hxb(s) * nybr(s) * nzbr(s)
        @test length(send_slab(g, 2)) == hxb(s) * nybr(s) * nzbr(s)
        @test length(recv_slab(g, 2)) == hxb(s) * nybr(s) * nzbr(s)
        @test length(send_slab(g, 3)) == hyb(s) * nxb(s) * nzbr(s)
        @test length(recv_slab(g, 3)) == hyb(s) * nxb(s) * nzbr(s)
        @test length(send_slab(g, 4)) == hyb(s) * nxb(s) * nzbr(s)
        @test length(recv_slab(g, 4)) == hyb(s) * nxb(s) * nzbr(s)
        @test length(send_slab(g, 5)) == hzb(s) * nxb(s) * nyb(s)
        @test length(recv_slab(g, 5)) == hzb(s) * nxb(s) * nyb(s)
        @test length(send_slab(g, 6)) == hzb(s) * nxb(s) * nyb(s)
        @test length(recv_slab(g, 6)) == hzb(s) * nxb(s) * nyb(s)
    end

    @testset "no overlap within same axis" begin
        @test isempty(intersect(Set(send_slab(g, 1)), Set(send_slab(g, 2))))
        @test isempty(intersect(Set(recv_slab(g, 1)), Set(recv_slab(g, 2))))
        @test isempty(intersect(Set(send_slab(g, 3)), Set(send_slab(g, 4))))
        @test isempty(intersect(Set(recv_slab(g, 3)), Set(recv_slab(g, 4))))
        @test isempty(intersect(Set(send_slab(g, 5)), Set(send_slab(g, 6))))
        @test isempty(intersect(Set(recv_slab(g, 5)), Set(recv_slab(g, 6))))
    end

    @testset "all indices in bounds" begin
        for i in 1:6
            @test all(idx -> 1 <= idx <= nboids(s), send_slab(g, i))
            @test all(idx -> 1 <= idx <= nboids(s), recv_slab(g, i))
        end
    end

    @testset "x-axis exchange" begin
        fa = map(1:nboids(s)) do b
            x, y, z = boid_to_coord(s, b)
            return 10000 * x + 100 * y + z
        end
        fb = copy(fa)
        simulate_axis_exchange!(fa, fb, g, 1)

        # println(repr("text/plain", reshape(fa, nxb(s), nyb(s), nzb(s))))
        # println(repr("text/plain", reshape(fb, nxb(s), nyb(s), nzb(s))))

        for idx in eachindex(recv_slab(g, 2))
            @test fa[recv_slab(g, 2)[idx]] == fb[send_slab(g, 1)[idx]]
        end
        for idx in eachindex(recv_slab(g, 1))
            @test fa[recv_slab(g, 1)[idx]] == fb[send_slab(g, 2)[idx]]
        end
    end

    @testset "staged exchange fills edges and corners" begin
        # Zero-initialize; only fill real interior boids
        fa = zeros(Float64, nboids(s))
        fb = map(1:nboids(s)) do b
            x, y, z = boid_to_coord(s, b)
            return 10000 * x + 100 * y + z
        end

        simulate_axis_exchange!(fa, fb, g, 1)
        simulate_axis_exchange!(fa, fb, g, 2)
        simulate_axis_exchange!(fa, fb, g, 3)

        # Interior untouched
        for z in (hzb(s) + 1):(hzb(s) + nzbr(s)), y in (hyb(s) + 1):(hyb(s) + nybr(s)), x in (hxb(s) + 1):(hxb(s) + nxbr(s))
            @test fa[coord_to_boid(s, x, y, z)] == 0.0
        end

        # z-axis recv slabs consistent with what was sent
        for idx in eachindex(recv_slab(g, 5))
            @test fa[recv_slab(g, 5)[idx]] == fb[send_slab(g, 6)[idx]]
        end
        for idx in eachindex(recv_slab(g, 6))
            @test fa[recv_slab(g, 6)[idx]] == fb[send_slab(g, 5)[idx]]
        end

        @test fa[coord_to_boid(s, 1, 1, 1)] == 10000.0 + 100.0 + 5.0
        @test fa[coord_to_boid(s, nxb(s), 1, 1)] == 60000.0 + 100.0 + 5.0
        @test fa[coord_to_boid(s, 1, nyb(s), 1)] == 10000.0 + 600.0 + 5.0
        @test fa[coord_to_boid(s, nxb(s), nyb(s), 1)] == 60000.0 + 600.0 + 5.0
        @test fa[coord_to_boid(s, 1, 1, nzb(s))] == 10000.0 + 100.0 + 2.0
        @test fa[coord_to_boid(s, nxb(s), 1, nzb(s))] == 60000.0 + 100.0 + 2.0
        @test fa[coord_to_boid(s, 1, nyb(s), nzb(s))] == 10000.0 + 600.0 + 2.0
        @test fa[coord_to_boid(s, nxb(s), nyb(s), nzb(s))] == 60000.0 + 600.0 + 2.0
    end

    @testset "zero halo returns empty slabs" begin
        s0 = UniformCubicalComplex3D(4, 4, 4, 1.0, 1.0, 1.0)
        g0 = GhostRegion(Datum{Boid,3}, s0)
        for i in 1:6
            @test isempty(send_slab(g0, i))
            @test isempty(recv_slab(g0, i))
        end
    end
end