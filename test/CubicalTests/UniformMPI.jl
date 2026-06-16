using Test

include(joinpath(@__DIR__, "..", "..", "src", "CubicalCode", "UniformMPI.jl"))

@testset "Tile Offset" begin
    base, rem = 2, 0
    @test all(tile_offset(out, base, rem) == out*2 for out in 0:2) # nout rem, constant offset

    base, rem = 14, 2
    @test [tile_offset(out, base, rem) for out in 0:6] == [0, 15, 30, 44, 58, 72, 86]

    base, rem = 111, 1
    @test tile_offset(0, base, rem) == 0
    @test tile_offset(1, base, rem) == 112
    @test tile_offset(8, base, rem) == 8*111 + 1

    base, rem = 2, 3
    @test [tile_offset(out, base, rem) for out in 0:3] == [0, 3, 6, 9]

    base, rem = 1, 6
    @test [tile_offset(out, base, rem) for out in 0:6] == [0, 2, 4, 6, 8, 10, 12]

    base, rem = 0, 2
    @test [tile_offset(out, base, rem) for out in 0:4] == [0, 1, 2, 2, 2]

    for (nwork, nout) in [(6,3), (7,3), (100,7), (11,4), (1,4), (2,5), (3,7), (0,3)]
        base, rem = nwork ÷ nout, nwork % nout
        offsets = [tile_offset(out, base, rem) for out in 0:nout]
        @test all(offsets[i] <= offsets[i+1] for i in 1:nout)
        @test offsets[end] == nwork
    end
end

@testset "Tile Size" begin
    base, rem = 14, 2
    @test tile_size(0, base, rem) == 15
    @test tile_size(1, base, rem) == 15
    @test tile_size(2, base, rem) == 14

    base, rem = 2, 3
    @test [tile_size(out, base, rem) for out in 0:3] == [3, 3, 3, 2]

    base, rem = 12, 0
    @test all(tile_size(out, base, rem) == 12 for out in 0:11)

    base, rem = 0, 2 # More outputs than workers
    @test [tile_size(out, base, rem) for out in 0:4] == [1, 1, 0, 0, 0]
end

@testset "Tile Coverage" begin
    for (nwork, nout) in [(6,3), (7,3), (5,3), (8,3), (100,7), (1000,9),
                     (11,4), (13,7), (144,12), (1,1), (1,5), (99,1),
                     (17,17), (1024,32), (2,5), (3,7), (0,3)]
        base, rem = nwork ÷ nout, nwork % nout
        @test sum(tile_size(out, base, rem) for out in 0:nout-1) == nwork
    end
end

@testset "Owning Output" begin
    base, rem, n = 2, 0, 3
    @test [owning_output_coord(w, base, rem, n) for w in 0:5] == [0,0,1,1,2,2]

    base, rem, n = 14, 2, 7
    @test owning_output_coord(0,  base, rem, n) == 0
    @test owning_output_coord(14, base, rem, n) == 0
    @test owning_output_coord(15, base, rem, n) == 1
    @test owning_output_coord(99, base, rem, n) == 6

    base, rem, n = 2, 3, 4
    @test [owning_output_coord(w, base, rem, n) for w in 0:10] == [0,0,0,1,1,1,2,2,2,3,3]

    base, rem, n = 1, 0, 17
    @test all(owning_output_coord(w, base, rem, n) == w for w in 0:16)

    # Tests that worker mapping to output is really in output's worker tile
    for (nwork, nout) in [(6,3), (7,3), (100,7), (1000,9), (11,4), (13,7),
                          (17,17), (1024,32), (99,1), (1,5)]

        base, rem = nwork ÷ nout, nwork % nout
        for w in 0:nwork-1
            out  = owning_output_coord(w, base, rem, nout)
            lo = tile_offset(out, base, rem)
            hi = lo + tile_size(out, base, rem) - 1
            @test lo <= w <= hi
        end
    end

end

@testset "Output Coordinate to Index" begin
    @test [o_coord_to_idx((ox,oy), (2,2)) for ox in 0:1 for oy in 0:1] == [0,1,2,3]
    @test [o_coord_to_idx((ox,oy), (3,2)) for ox in 0:2 for oy in 0:1] == collect(0:5)

    @test o_coord_to_idx((0,0,0), (2,2,2)) == 0
    @test o_coord_to_idx((0,0,1), (2,2,2)) == 1
    @test o_coord_to_idx((1,0,0), (2,2,2)) == 4
    @test o_coord_to_idx((1,2,3), (2,3,4)) == 23

    # Roundtrip
    for dims in [(2,2), (3,2), (4,3), (2,3,4)]
        coords = vec(collect(Iterators.product(map(d -> 0:d-1, dims)...)))
        @test sort([o_coord_to_idx(c, dims) for c in coords]) == collect(0:prod(dims)-1)
    end
end

# TODO: This doesn't check if there are fewer mesh points than workers
@testset "Worker Mesh Sizes and Offsets" begin
    # Even division
    m_dim, w_dim = 31, 3
    @test [worker_mesh_offset(c, m_dim, w_dim) for c in 0:2] == [0, 10, 20]
    @test all(worker_mesh_size(c, m_dim, w_dim) == 11 for c in 0:2)

    # Uneven division
    m_dim, w_dim = 101, 7
    @test [worker_mesh_offset(c, m_dim, w_dim) for c in 0:6] == [0, 15, 30, 44, 58, 72, 86]
    @test [worker_mesh_size(c, m_dim, w_dim) for c in 0:6] == [16, 16, 15, 15, 15, 15, 15]

    # Large offset boundary verification
    @test worker_mesh_offset(8, 890, 8) == 889
end

@testset "Worker Mesh Size" begin
    # Even division
    m_dim, w_dim = 31, 5
    @test all(worker_mesh_size(c, m_dim, w_dim) == 7 for c in 0:4)

    # Uneven division
    m_dim, w_dim = 33, 5
    @test [worker_mesh_size(c, m_dim, w_dim) for c in 0:4] == [8, 8, 7, 7, 7]

    # One worker only
    m_dim, w_dim = 3, 1
    @test [worker_mesh_size(c, m_dim, w_dim) for c in 0:0] == [m_dim]

    # Same workers as points
    m_dim, w_dim = 12, 12
    @test all(worker_mesh_size(c, m_dim, w_dim) == 2 for c in 0:4)

    # Conservation of Vertices
    test_cases = [
        (31, 5),   # Even division
        (33, 5),   # Uneven division
        (100, 7),  # Larger uneven
        (12, 12),  # Worker count == cell count
        (13, 1),   # 1 Worker
    ]
    for (m_dim, w_dim) in test_cases
        num_dual_cells = m_dim - 1
        @test sum(worker_mesh_size(c, m_dim, w_dim) - 1 for c in 0:w_dim-1) == num_dual_cells
    end
end

@testset "OutputWorkerCache Serialization" begin
    # Test for N=2 (2D case)
    let N = 2
        original_cache = OutputWorkerCache{N}((10, 20), (5, 8))
        
        # Serialize the cache
        serialized_buf = serialize(original_cache)
        
        # Check buffer type and length
        @test serialized_buf isa AbstractVector{Int32}
        @test length(serialized_buf) == 2 * N
        @test serialized_buf == Int32[10, 20, 5, 8]
        
        # Deserialize back
        roundtrip_cache = deserialize(OutputWorkerCache{N}, serialized_buf)
        
        # Verify equality
        @test roundtrip_cache.lm_dims == original_cache.lm_dims
        @test roundtrip_cache.lm_gm_offsets == original_cache.lm_gm_offsets
        @test roundtrip_cache == original_cache
    end

    # Test for N=3 (3D case)
    let N = 3
        original_cache = OutputWorkerCache{N}((12, 12, 12), (11, 23, 35))
        
        serialized_buf = serialize(original_cache)
        
        @test length(serialized_buf) == 2 * N
        @test serialized_buf == Int32[12, 12, 12, 11, 23, 35]
        
        roundtrip_cache = deserialize(OutputWorkerCache{N}, serialized_buf)
        
        @test roundtrip_cache == original_cache
    end
end

@testset "Output Mesh Dimensions" begin
    # Test a 2x2 worker tile in 2D (Y-major ordering)
    let N = 2
        worker_caches = [
            OutputWorkerCache{N}((10, 20), (0,0)),   # W(0,0)
            OutputWorkerCache{N}((10, 22), (0,20)),  # W(0,1)
            OutputWorkerCache{N}((11, 20), (10,0)),  # W(1,0)
            OutputWorkerCache{N}((11, 22), (10,20))  # W(1,1)
        ]
        lt_dims = (2, 2)
        om_dims = output_mesh_dimensions(worker_caches, lt_dims)
        @test om_dims == (10 + 11 - 1, 20 + 22 - 1)
    end
    
    # Test a 2x1x3 worker tile in 3D (Z-Y-X major ordering)
    let N = 3
        worker_caches = [
            OutputWorkerCache{N}((10, 5, 8), (0,0,0)),    # W(0,0,0)
            OutputWorkerCache{N}((10, 5, 9), (0,0,8)),    # W(0,0,1)
            OutputWorkerCache{N}((10, 5, 7), (0,0,17)),   # W(0,0,2)
            OutputWorkerCache{N}((12, 5, 8), (10,0,0)),   # W(1,0,0)
            OutputWorkerCache{N}((12, 5, 9), (10,0,8)),   # W(1,0,1)
            OutputWorkerCache{N}((12, 5, 7), (10,0,17))   # W(1,0,2)
        ]
        lt_dims = (2, 1, 3)
        om_dims = output_mesh_dimensions(worker_caches, lt_dims)
        @test om_dims == (10 + 12 - 1, 5, 8 + 9 + 7 - 2)
    end

    # Edge Case: Tile is a single worker
    let N = 2
        worker_caches = [OutputWorkerCache{N}((15, 25), (0,0))]
        lt_dims = (1, 1)
        om_dims = output_mesh_dimensions(worker_caches, lt_dims)
        @test om_dims == (15, 25)
    end

    # Edge Case: Long, thin tile (4x1)
    let N = 2
        worker_caches = [
            OutputWorkerCache{N}((10, 50), (0,0)),
            OutputWorkerCache{N}((11, 50), (10,0)),
            OutputWorkerCache{N}((9, 50), (21,0)),
            OutputWorkerCache{N}((12, 50), (30,0))
        ]
        lt_dims = (4, 1)
        om_dims = output_mesh_dimensions(worker_caches, lt_dims)
        @test om_dims == (10 + 11 + 9 + 12 - 3, 50)
    end
end

@testset "Mesh Offsets" begin
    # 2D Case
    let N = 2
        worker_caches = [
            OutputWorkerCache{N}((10, 20), (15, 30)),   # W(0,0) - Origin of output mesh
            OutputWorkerCache{N}((10, 22), (15, 50)),   # W(0,1)
            OutputWorkerCache{N}((11, 20), (25, 30)),   # W(1,0)
            OutputWorkerCache{N}((11, 22), (25, 50))    # W(1,1)
        ]
        
        # Test output_mesh_offsets (should be the global offset of the first cache)
        om_gm_offsets = output_mesh_offsets(worker_caches)
        @test om_gm_offsets == (15, 30)
        
        # Test local_mesh_worker_offsets (should be relative to (15, 30))
        lm_offsets = local_mesh_worker_offsets(worker_caches)
        @test length(lm_offsets) == 4
        @test lm_offsets[1] == (0, 0)
        @test lm_offsets[2] == (0, 20)
        @test lm_offsets[3] == (10, 0)
        @test lm_offsets[4] == (10, 20)
    end

    # 3D Case
    let N = 3
        worker_caches = [
            OutputWorkerCache{N}((5, 5, 5), (100, 200, 300)),  # W(0,0,0) - Origin
            OutputWorkerCache{N}((5, 5, 8), (100, 200, 305)),  # W(0,0,1)
            OutputWorkerCache{N}((5, 7, 5), (100, 205, 300)),  # W(0,1,0)
            OutputWorkerCache{N}((6, 5, 5), (105, 200, 300)),  # W(1,0,0)
        ]

        # Test output_mesh_offsets
        @test output_mesh_offsets(worker_caches) == (100, 200, 300)

        # Test local_mesh_worker_offsets
        lm_offsets = local_mesh_worker_offsets(worker_caches)
        @test lm_offsets == [(0, 0, 0), (0, 0, 5), (0, 5, 0), (5, 0, 0)]
    end
    
    # Edge Case: Only 1 worker
    let N = 2
        worker_caches = [OutputWorkerCache{N}((15, 25), (7, 14))]
        
        @test output_mesh_offsets(worker_caches) == (7, 14)
        @test local_mesh_worker_offsets(worker_caches) == [(0, 0)]
    end
end

@testset "OutputCache Creation from test_arr" begin
    # Test a 2D case where this output process manages a 2x2 tile of workers
    let N = 2
        cart_coords = (0, 0) # This output process is at coords (0,0)
        base_tiles = (2, 2) # Mock base sizes for worker distribution
        rems = (0, 0) # Mock remainders
        nworkers = 4

        # 2. Create and serialize mock worker caches
        wc_list = [
            OutputWorkerCache{N}((10, 20), (0, 0)),    # W(0,0)
            OutputWorkerCache{N}((10, 22), (0, 19)),   # W(0,1)
            OutputWorkerCache{N}((11, 20), (9, 0)),    # W(1,0)
            OutputWorkerCache{N}((11, 22), (9, 19))    # W(1,1)
        ]
        test_arr = vcat(serialize.(wc_list)...)

        # 3. Call constructor with test_arr
        cache = OutputCache{N}(nothing, cart_coords, base_tiles, rems, test_arr=test_arr)

        # 4. Validate fields
        @test cache isa OutputCache{N}
        @test cache.nworkers == nworkers
        @test length(cache.worker_caches) == nworkers
        @test cache.worker_caches == wc_list
        @test cache.lt_dims == (2, 2)
        @test cache.om_dims == (10 + 11 - 1, 20 + 22 - 1)
        @test cache.om_gm_offsets == (0, 0)
        @test cache.lm_om_offsets[4] == (9, 19)
    end

    # Test a 3D case with a 1x1x2 worker tile
    let N = 3
        cart_coords = (1, 2, 0)
        base_tiles = (1, 1, 2)
        rems = (0, 0, 0)
        nworkers = 2

        wc_list = [
            OutputWorkerCache{N}((5, 6, 7), (10, 20, 30)),
            OutputWorkerCache{N}((5, 6, 8), (10, 20, 36))
        ]
        test_arr = vcat(serialize.(wc_list)...)
        
        cache = OutputCache{N}(nothing, cart_coords, base_tiles, rems, test_arr=test_arr)

        @test cache.nworkers == nworkers
        @test cache.worker_caches == wc_list
        @test cache.lt_dims == (1, 1, 2)
        @test cache.om_dims == (5, 6, 7 + 8 - 1)
        @test cache.om_gm_offsets == (10, 20, 30)
    end
end
@testset "Integration: Global Mesh -> Workers -> Outputs" begin
    # ==========================================
    # Case 1: Even Division
    # ==========================================
    let m_dim = 33, w_dim = 8, o_dim = 4
        # 33 vertices (32 cells) / 8 workers = 4 cells per worker (5 vertices)
        # 8 workers / 4 outputs = 2 workers per output process
        
        w_base, w_rem = w_dim ÷ o_dim, w_dim % o_dim
        
        # Test Output Process 1 (the 2nd output process)
        out_id = 1 
        
        # 1. Output's worker coverage
        num_workers = tile_size(out_id, w_base, w_rem)
        first_worker = tile_offset(out_id, w_base, w_rem)
        @test num_workers == 2
        @test first_worker == 2 # This output owns workers 2 and 3
        
        # 2. Worker mesh properties for this output
        worker_sizes = [worker_mesh_size(w, m_dim, w_dim) for w in first_worker : first_worker + num_workers - 1]
        worker_offsets = [worker_mesh_offset(w, m_dim, w_dim) for w in first_worker : first_worker + num_workers - 1]
        
        @test worker_sizes == [5, 5]    # Each worker gets 5 vertices
        @test worker_offsets == [8, 12] # Offset by 8 cells (2 workers * 4 cells) and 12 cells
        
        # 3. Output mesh reconstruction
        # Total output mesh size is sum of worker cells + 1 vertex
        out_mesh_size = sum(sz - 1 for sz in worker_sizes) + 1
        @test out_mesh_size == 9 # 8 cells + 1 vertex
    end

    # ==========================================
    # Case 2: Uneven Division
    # ==========================================
    let m_dim = 35, w_dim = 5, o_dim = 3
        # 35 vertices (34 cells) / 5 workers: base=6, rem=4
        # Worker cell distribution: 7, 7, 7, 7, 6 (Vertices: 8, 8, 8, 8, 7)
        # 5 workers / 3 outputs: base=1, rem=2
        # Output worker counts: 2, 2, 1
        
        w_base, w_rem = w_dim ÷ o_dim, w_dim % o_dim

        # --- Test Output Process 0 (The 1st output) ---
        out_id_0 = 0
        num_workers_0 = tile_size(out_id_0, w_base, w_rem)
        first_worker_0 = tile_offset(out_id_0, w_base, w_rem)
        
        @test num_workers_0 == 2 # Owns workers 0 and 1
        @test first_worker_0 == 0
        
        worker_sizes_0 = [worker_mesh_size(w, m_dim, w_dim) for w in first_worker_0 : first_worker_0 + num_workers_0 - 1]
        worker_offsets_0 = [worker_mesh_offset(w, m_dim, w_dim) for w in first_worker_0 : first_worker_0 + num_workers_0 - 1]
        
        @test worker_sizes_0 == [8, 8]
        @test worker_offsets_0 == [0, 7]
        @test sum(sz - 1 for sz in worker_sizes_0) + 1 == 15 # 15 vertices for Output 0

        # --- Test Output Process 2 (The 3rd output) ---
        out_id_2 = 2
        num_workers_2 = tile_size(out_id_2, w_base, w_rem)
        first_worker_2 = tile_offset(out_id_2, w_base, w_rem)
        
        @test num_workers_2 == 1 # Owns only worker 4
        @test first_worker_2 == 4
        
        worker_sizes_2 = [worker_mesh_size(w, m_dim, w_dim) for w in first_worker_2 : first_worker_2 + num_workers_2 - 1]
        worker_offsets_2 = [worker_mesh_offset(w, m_dim, w_dim) for w in first_worker_2 : first_worker_2 + num_workers_2 - 1]
        
        @test worker_sizes_2 == [7] # Last worker gets the remaining 6 cells (7 vertices)
        @test worker_offsets_2 == [28] # 4 previous workers * 7 cells = 28
        @test sum(sz - 1 for sz in worker_sizes_2) + 1 == 7
        
        # --- Final Sanity Check ---
        # The sum of all output cells should exactly equal global cells (m_dim - 1)
        # Output 0 cells: 14. Output 1 cells (workers 2,3): 14. Output 2 cells: 6.
        # Total: 14 + 14 + 6 = 34 cells.
        @test 14 + 14 + 6 == m_dim - 1
    end
end
