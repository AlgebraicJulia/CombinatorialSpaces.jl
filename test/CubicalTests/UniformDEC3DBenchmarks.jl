using BenchmarkTools
using KernelAbstractions

# Include your source files (adjust paths as necessary)
include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMesh3D.jl")
include("../../src/CubicalCode/UniformKernelDEC3D.jl")

function run_benchmarks()
    println("Setting up benchmark domain...")
    
    # Use a moderately large mesh to get meaningful kernel timings
    nx, ny, nz = 50, 50, 50 
    s = UniformCubicalComplex3D(nx, ny, nz, 0.1, 0.1, 0.1)
    FT = Float64
    
    # Assuming CPU backend for the benchmark, but you can swap this for CUDA/AMDGPU etc.
    backend = CPU() 

    # --- Allocate Primal Forms ---
    # f0: vertices, f1: edges, f2: quads (faces), f3: boids (cubes)
    f0 = KernelAbstractions.ones(backend, FT, nv(s))
    f1 = KernelAbstractions.ones(backend, FT, ne(s))
    f2 = KernelAbstractions.ones(backend, FT, nquads(s))
    f3 = KernelAbstractions.ones(backend, FT, nboids(s))

    # --- Allocate Dual Forms ---
    # d0: boid centers, d1: quad centers, d2: edge centers, d3: vertices
    d0 = KernelAbstractions.ones(backend, FT, nboids(s))
    d1 = KernelAbstractions.ones(backend, FT, nquads(s))
    d2 = KernelAbstractions.ones(backend, FT, ne(s))
    d3 = KernelAbstractions.ones(backend, FT, nv(s))

    # --- Allocate Vector Field Components (for flat_dp) ---
    X = KernelAbstractions.ones(backend, FT, nboids(s))
    Y = KernelAbstractions.ones(backend, FT, nboids(s))
    Z = KernelAbstractions.ones(backend, FT, nboids(s))

    suite = BenchmarkGroup()

    # --- 1. Exterior Derivatives (Primal) ---
    suite["Exterior Derivative", "d0 (0-form -> 1-form)"] = @benchmarkable exterior_derivative(Val(0), $s, $f0)
    suite["Exterior Derivative", "d1 (1-form -> 2-form)"] = @benchmarkable exterior_derivative(Val(1), $s, $f1)
    suite["Exterior Derivative", "d2 (2-form -> 3-form)"] = @benchmarkable exterior_derivative(Val(2), $s, $f2)

    # --- 2. Dual Exterior Derivatives ---
    suite["Dual Derivative", "dd0 (dual 0-form -> dual 1-form)"] = @benchmarkable dual_derivative(Val(0), $s, $d0)
    suite["Dual Derivative", "dd1 (dual 1-form -> dual 2-form)"] = @benchmarkable dual_derivative(Val(1), $s, $d1)
    suite["Dual Derivative", "dd2 (dual 2-form -> dual 3-form)"] = @benchmarkable dual_derivative(Val(2), $s, $d2)

    # --- 3. Hodge Stars (Primal -> Dual) ---
    suite["Hodge Star", "⋆0 (0-form)"] = @benchmarkable hodge_star(Val(0), $s, $f0)
    suite["Hodge Star", "⋆1 (1-form)"] = @benchmarkable hodge_star(Val(1), $s, $f1)
    suite["Hodge Star", "⋆2 (2-form)"] = @benchmarkable hodge_star(Val(2), $s, $f2)
    suite["Hodge Star", "⋆3 (3-form)"] = @benchmarkable hodge_star(Val(3), $s, $f3)

    # --- 4. Inverse Hodge Stars (Dual -> Primal) ---
    suite["Inverse Hodge Star", "⋆⁻¹0 (dual 0-form)"] = @benchmarkable inv_hodge_star(Val(0), $s, $d0)
    suite["Inverse Hodge Star", "⋆⁻¹1 (dual 1-form)"] = @benchmarkable inv_hodge_star(Val(1), $s, $d1)
    suite["Inverse Hodge Star", "⋆⁻¹2 (dual 2-form)"] = @benchmarkable inv_hodge_star(Val(2), $s, $d2)
    suite["Inverse Hodge Star", "⋆⁻¹3 (dual 3-form)"] = @benchmarkable inv_hodge_star(Val(3), $s, $d3)

    # --- 5. Wedge Products ---
    suite["Wedge Product", "∧ (1-form, 1-form)"] = @benchmarkable wedge_product(Val(1), Val(1), $s, $f1, $f1)
    suite["Wedge Product", "∧ (1-form, 2-form)"] = @benchmarkable wedge_product(Val(1), Val(2), $s, $f1, $f2)
    suite["Wedge Product", "∧_dd (dual 0-form, dual 1-form)"] = @benchmarkable wedge_product_dd(Val(0), Val(1), $s, $d0, $d1)

    # --- 6. Sharp and Flat Operators ---
    suite["Vector Ops", "Sharp DD (dual 1-form -> vector field)"] = @benchmarkable sharp_dd($s, $d1)
    suite["Vector Ops", "Flat DP (vector field -> primal 1-form)"] = @benchmarkable flat_dp($s, $X, $Y, $Z)

    println("Tuning benchmarks (this will take a couple of minutes)...")
    tune!(suite)

    println("Running benchmarks...")
    results = run(suite, verbose=true)

    output_file = "benchmark_results.txt"
    println("Writing results to $output_file...")
    
    open(output_file, "w") do io
        println(io, "=== 3D Cubical DEC Benchmark Results (Allocating) ===")
        println(io, "Mesh Size: $(nx)x$(ny)x$(nz)")
        println(io, "Data Type: $FT")
        println(io, "=====================================================\n")
        show(io, MIME("text/plain"), results)
    end
    
    println("Benchmarking complete! Check $output_file for the results.")
end

run_benchmarks()
