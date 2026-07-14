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
        @test length(send_slab(g, 1)) == nyqr(s)
        @test length(recv_slab(g, 1)) == nyqr(s)
        @test length(send_slab(g, 2)) == nyqr(s)
        @test length(recv_slab(g, 2)) == nyqr(s)
        @test length(send_slab(g, 3)) == nxq(s)
        @test length(recv_slab(g, 3)) == nxq(s)
        @test length(send_slab(g, 4)) == nxq(s)
        @test length(recv_slab(g, 4)) == nxq(s)
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
        for y in (2):(1 + nyqr(s)), x in (2):(1 + nxqr(s))
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
        for y in (2):(1 + nyqr(s)), x in (2):(1 + nxqr(s))
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

@testset "GhostRegion Edge 2D" begin
    _hx = _hy = 1
    s = UniformCubicalComplex2D(5, 5, 1.0, 1.0; halo_x = _hx, halo_y = _hy)
    g = GhostRegion(Datum{Edge,2}, s)


    @testset "struct shape" begin
        @test length(send_slab(g, WEST)) == _hx * nyqr(s) + (_hx + 1) * nyqr(s)
        @test length(recv_slab(g, EAST)) == _hx * nyqr(s) + (_hx + 1) * nyqr(s)

        @test length(send_slab(g, EAST)) == _hx * nyqr(s) + _hx * nyqr(s)
        @test length(recv_slab(g, WEST)) == _hx * nyqr(s) + _hx * nyqr(s)

        # Include x-halo as well
        @test length(send_slab(g, SOUTH)) == _hy * nx(s) + (_hy + 1) * nxe(s)
        @test length(recv_slab(g, NORTH)) == _hy * nx(s) + (_hy + 1) * nxe(s)

        @test length(send_slab(g, NORTH)) == _hy * nx(s) + _hy * nxe(s)
        @test length(recv_slab(g, SOUTH)) == _hy * nx(s) + _hy * nxe(s)
    end

    @testset "no overlap within same axis" begin
        @test isempty(intersect(Set(send_slab(g, WEST)), Set(send_slab(g, EAST))))
        @test isempty(intersect(Set(recv_slab(g, WEST)), Set(recv_slab(g, EAST))))
        @test isempty(intersect(Set(send_slab(g, SOUTH)), Set(send_slab(g, NORTH))))
        @test isempty(intersect(Set(recv_slab(g, SOUTH)), Set(recv_slab(g, NORTH))))
    end

    @testset "all indices in bounds" begin
        for i in 1:4
            @test all(idx -> 1 <= idx <= ne(s), send_slab(g, i))
            @test all(idx -> 1 <= idx <= ne(s), recv_slab(g, i))
        end
    end

    @testset "x-axis exchange" begin
        fa = zeros(Int64, ne(s))
        fb = map(1:ne(s)) do e
            x, y, align = edge_to_coord(s, e)
            return 1000 * x + y
        end

        simulate_axis_exchange!(fa, fb, g, 1)

        @test all(fa[recv_slab(g, WEST)] .== fb[send_slab(g, EAST)])
        @test all(fa[recv_slab(g, EAST)] .== fb[send_slab(g, WEST)])

        # Interior x-edges untouched
        for y in (_hy + 1):(_hy + nyqr(s)), x in (_hx + 1):(nxe(s) - _hx)
            @test fa[coord_to_edge(s, x, y, X_ALIGN)] == 0
        end
        # Interior y-edges untouched (excluding the shared boundary column now in send slab)
        for y in (_hy + 1):(_hy + nyqr(s)), x in (_hx + 1):(nx(s) - _hx - 1)
            @test fa[coord_to_edge(s, x, y, Y_ALIGN)] == 0
        end

        xe = reshape(xedges(s, fa), nxe(s), ny(s))
        ye = reshape(yedges(s, fa), nx(s), nye(s))

        @test xe[1, :] == [0; [1000 * (nxe(s) - _hx) + y for y in (_hy + 1):(_hy + nyqr(s))]; 0; 0]
        @test xe[nxe(s), :] == [0; [1000 * (_hx + 1) + y for y in (_hy + 1):(_hy + nyqr(s))]; 0; 0]
        @test ye[1, :] == [0; [1000 * (nx(s) - 2_hx) + y for y in (_hy + 1):(_hy + nyqr(s))]; 0]
        @test ye[nx(s) - _hx, :] == [0; [1000 * (_hx + 1) + y for y in (_hy + 1):(_hy + nyqr(s))]; 0]
        @test ye[nx(s), :] == [0; [1000 * (_hx + 2) + y for y in (_hy + 1):(_hy + nyqr(s))]; 0]
    end

    @testset "y-axis exchange" begin
        fa = zeros(Int64, ne(s))
        fb = map(1:ne(s)) do e
            x, y, align = edge_to_coord(s, e)
            return 1000 * x + y
        end

        simulate_axis_exchange!(fa, fb, g, 2)

        @test all(fa[recv_slab(g, SOUTH)] .== fb[send_slab(g, NORTH)])
        @test all(fa[recv_slab(g, NORTH)] .== fb[send_slab(g, SOUTH)])

        # Interior untouched
        for y in (_hy + 1):(_hy + nyqr(s)), x in (_hx + 1):(nxe(s) - _hx)
            @test fa[coord_to_edge(s, x, y, X_ALIGN)] == 0
        end
        for y in (_hy + 1):(_hy + nyqr(s)), x in (_hx + 1):(nx(s) - _hx - 1)
            @test fa[coord_to_edge(s, x, y, Y_ALIGN)] == 0
        end

        xe = reshape(xedges(s, fa), nxe(s), ny(s))
        ye = reshape(yedges(s, fa), nx(s), nye(s))

        @test xe[:, 1] == [1000x + (ny(s) - 2_hy) for x in 1:nxe(s)]
        @test xe[:, ny(s) - _hy] == [1000x + (_hy + 1) for x in 1:nxe(s)]
        @test xe[:, ny(s)] == [1000x + (_hy + 2) for x in 1:nxe(s)]
        @test ye[:, 1] == [1000x + (nye(s) - 2_hy + 1) for x in 1:nx(s)]
        @test ye[:, nye(s)] == [1000x + (_hy + 1) for x in 1:nx(s)]
    end

    @testset "zero halo returns boundary shared-edge slabs" begin
        s0 = UniformCubicalComplex2D(4, 4, 1.0, 1.0; halo_x = 0, halo_y = 0)
        g0 = GhostRegion(Datum{Edge,2}, s0)

        # recv_low slabs are empty — no halo depth to receive into
        @test isempty(recv_slab(g0, 1))   # WEST
        @test isempty(recv_slab(g0, 3))   # SOUTH

        # WEST send: left boundary y-edges
        expected_west_send = Int32[coord_to_edge(s0, 1, b, Y_ALIGN) for b in 1:nyqr(s0)]
        @test send_slab(g0, 1) == expected_west_send

        # EAST recv: right boundary y-edges
        expected_east_recv = Int32[coord_to_edge(s0, nx(s0), b, Y_ALIGN) for b in 1:nyqr(s0)]
        @test recv_slab(g0, 2) == expected_east_recv

        # SOUTH send: bottom boundary x-edges
        expected_south_send = Int32[coord_to_edge(s0, b, 1, X_ALIGN) for b in 1:nxe(s0)]
        @test send_slab(g0, 3) == expected_south_send

        # NORTH recv: top boundary x-edges
        expected_north_recv = Int32[coord_to_edge(s0, b, ny(s0), X_ALIGN) for b in 1:nxe(s0)]
        @test recv_slab(g0, 4) == expected_north_recv

        # EAST and NORTH send slabs are empty — no real layer beyond the boundary to send
        @test isempty(send_slab(g0, 2))   # EAST
        @test isempty(send_slab(g0, 4))   # NORTH
    end
end

@testset "GhostRegion Edge 2D pack/unpack roundtrip" begin
    s = UniformCubicalComplex2D(5, 5, 1.0, 1.0; halo_x = 1, halo_y = 1)
    g = GhostRegion(Datum{Edge,2}, s)

    d_edges = Datum{Edge,2}("edges", "fields", Float64)
    stream = DataStream(Datum[d_edges])
    fields = [map(1:ne(s)) do e
        x, y, align = edge_to_coord(s, e)
        return Float64(1000 * x + y)
    end]

    @testset "WEST/EAST roundtrip" begin
        fb_send_low = FaceBuffer([g.send[Int(WEST)]], [length(g.send[Int(WEST)])])
        fb_recv_high = FaceBuffer([g.recv[Int(EAST)]], [length(g.recv[Int(EAST)])])
        fb_send_high = FaceBuffer([g.send[Int(EAST)]], [length(g.send[Int(EAST)])])
        fb_recv_low = FaceBuffer([g.recv[Int(WEST)]], [length(g.recv[Int(WEST)])])

        buf_low = zeros(Float64, sum(fb_send_low.cell_lens))
        buf_high = zeros(Float64, sum(fb_send_high.cell_lens))

        fa = zeros(Float64, ne(s))

        _pack_face!(buf_low, fb_send_low, fields)
        _pack_face!(buf_high, fb_send_high, fields)

        @test buf_low[1:4] == Float64[2002, 2003, 2004, 2005] # xedges
        @test buf_low[5:end] == Float64[2002, 3002, 2003, 3003, 2004, 3004, 2005, 3005] # yedges

        @test buf_high[1:4] == Float64[5002, 5003, 5004, 5005] # xedges
        @test buf_high[5:end] == Float64[5002, 5003, 5004, 5005] # yedges

        _unpack_face!([fa], fb_recv_low, buf_high)
        _unpack_face!([fa], fb_recv_high, buf_low)

        for idx in eachindex(recv_slab(g, Int(WEST)))
            @test fa[recv_slab(g, Int(WEST))[idx]] == fields[1][send_slab(g, Int(EAST))[idx]]
        end
        for idx in eachindex(recv_slab(g, Int(EAST)))
            @test fa[recv_slab(g, Int(EAST))[idx]] == fields[1][send_slab(g, Int(WEST))[idx]]
        end
    end

    @testset "SOUTH/NORTH roundtrip" begin
        fb_send_low = FaceBuffer([g.send[Int(SOUTH)]], [length(g.send[Int(SOUTH)])])
        fb_recv_high = FaceBuffer([g.recv[Int(NORTH)]], [length(g.recv[Int(NORTH)])])
        fb_send_high = FaceBuffer([g.send[Int(NORTH)]], [length(g.send[Int(NORTH)])])
        fb_recv_low = FaceBuffer([g.recv[Int(SOUTH)]], [length(g.recv[Int(SOUTH)])])

        buf_low = zeros(Float64, sum(fb_send_low.cell_lens))
        buf_high = zeros(Float64, sum(fb_send_high.cell_lens))

        fa = zeros(Float64, ne(s))

        _pack_face!(buf_low, fb_send_low, fields)
        _pack_face!(buf_high, fb_send_high, fields)

        @test buf_low[1:7] == Float64[1002, 2002, 3002, 4002, 5002, 6002, 7002] # xedges
        @test buf_low[8:end] == Float64[1002, 1003, 2002, 2003, 3002, 3003, 4002, 4003, 5002, 5003, 6002, 6003] # yedges

        @test buf_high[1:7] == Float64[1005, 2005, 3005, 4005, 5005, 6005, 7005] # xedges
        @test buf_high[8:end] == Float64[1005, 2005, 3005, 4005, 5005, 6005] # yedges

        _unpack_face!([fa], fb_recv_low, buf_high)
        _unpack_face!([fa], fb_recv_high, buf_low)

        for idx in eachindex(recv_slab(g, Int(SOUTH)))
            @test fa[recv_slab(g, Int(SOUTH))[idx]] == fields[1][send_slab(g, Int(NORTH))[idx]]
        end
        for idx in eachindex(recv_slab(g, Int(NORTH)))
            @test fa[recv_slab(g, Int(NORTH))[idx]] == fields[1][send_slab(g, Int(SOUTH))[idx]]
        end
    end
end

# ── 3D Tests ──────────────────────────────────────────────────────────────────

@testset "GhostRegion Boid 3D" begin
    _hx = _hy = _hz = 1
    s = UniformCubicalComplex3D(5, 5, 5, 1.0, 1.0, 1.0; halo_x = _hx, halo_y = _hy, halo_z = _hz)
    g = GhostRegion(Datum{Boid,3}, s)

    @testset "struct shape" begin
        @test length(send_slab(g, 1)) == _hx * nybr(s) * nzbr(s)
        @test length(recv_slab(g, 1)) == _hx * nybr(s) * nzbr(s)
        @test length(send_slab(g, 2)) == _hx * nybr(s) * nzbr(s)
        @test length(recv_slab(g, 2)) == _hx * nybr(s) * nzbr(s)
        @test length(send_slab(g, 3)) == _hy * nxb(s) * nzbr(s)
        @test length(recv_slab(g, 3)) == _hy * nxb(s) * nzbr(s)
        @test length(send_slab(g, 4)) == _hy * nxb(s) * nzbr(s)
        @test length(recv_slab(g, 4)) == _hy * nxb(s) * nzbr(s)
        @test length(send_slab(g, 5)) == _hz * nxb(s) * nyb(s)
        @test length(recv_slab(g, 5)) == _hz * nxb(s) * nyb(s)
        @test length(send_slab(g, 6)) == _hz * nxb(s) * nyb(s)
        @test length(recv_slab(g, 6)) == _hz * nxb(s) * nyb(s)
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
        for z in (_hz + 1):(_hz + nzbr(s)), y in (_hy + 1):(_hy + nybr(s)), x in (_hx + 1):(_hx + nxbr(s))
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