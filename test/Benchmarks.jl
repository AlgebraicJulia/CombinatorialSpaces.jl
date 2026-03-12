module Benchmarks

using Catlab
using Catlab.Graphics
using CombinatorialSpaces
using LinearAlgebra
using BenchmarkTools
using Printf
using Random

@info "Beginning DEC Operator Benchmarks"
begin
    mesh_size = 5
    float_type::DataType = Float64
    primal_earth = loadmesh(Icosphere(mesh_size))
    orient!(primal_earth);
    earth = EmbeddedDeltaDualComplex2D{Bool,float_type,Point3d}(primal_earth);
    subdivide_duals!(earth, Barycenter());
end

begin
    println("Mesh: " * "Blender Ico Sphere, $(mesh_size) Subdivisions")
    println("")

    println("Number of primal vertices: ", nv(earth))
    println("Number of primal edges: ", ne(earth))
    println("Number of primal triangles: ", ntriangles(earth))
    println("")

    println("Number of dual vertices: ", nparts(earth, :DualV))
    println("Number of dual edges: ", nparts(earth, :DualE))
    println("Number of dual triangles: ", nparts(earth, :DualTri))
    println("----------------------------------------------------------------")
end

dec_op_suite = BenchmarkGroup()

begin
    dec_op_suite["Exterior Derivative"]["New Form-0"] = @benchmarkable dec_differential(0, $earth)
    dec_op_suite["Exterior Derivative"]["Old Form-0"] = @benchmarkable d(0, $earth)

    dec_op_suite["Exterior Derivative"]["New Form-1"] = @benchmarkable dec_differential(1, $earth)
    dec_op_suite["Exterior Derivative"]["Old Form-1"] = @benchmarkable d(1, $earth)
end

begin
    dec_op_suite["Boundary"]["New Form-1"] = @benchmarkable dec_boundary(1, $earth)
    dec_op_suite["Boundary"]["Old Form-1"] = @benchmarkable ∂(1, $earth)

    dec_op_suite["Boundary"]["New Form-2"] = @benchmarkable dec_boundary(2, $earth)
    dec_op_suite["Boundary"]["Old Form-2"] = @benchmarkable ∂(2, $earth)
end

begin
    dec_op_suite["Dual Derivative"]["New Dual-Form-0"] = @benchmarkable dec_dual_derivative(0, $earth)
    dec_op_suite["Dual Derivative"]["Old Dual-Form-0"] = @benchmarkable dual_derivative(0, $earth)

    dec_op_suite["Dual Derivative"]["New Dual-Form-1"] = @benchmarkable dec_dual_derivative(1, $earth)
    dec_op_suite["Dual Derivative"]["Old Dual-Form-1"] = @benchmarkable dual_derivative(1, $earth)
end

begin
    dec_op_suite["Diagonal Hodge"]["New Form-0"] = @benchmarkable dec_hodge_star(0, $earth, DiagonalHodge())
    dec_op_suite["Diagonal Hodge"]["Old Form-0"] = @benchmarkable hodge_star(0, $earth, DiagonalHodge())

    dec_op_suite["Diagonal Hodge"]["New Form-1"] = @benchmarkable dec_hodge_star(1, $earth, DiagonalHodge())
    dec_op_suite["Diagonal Hodge"]["Old Form-1"] = @benchmarkable hodge_star(1, $earth, DiagonalHodge())

    dec_op_suite["Diagonal Hodge"]["New Form-2"] = @benchmarkable dec_hodge_star(2, $earth, DiagonalHodge())
    dec_op_suite["Diagonal Hodge"]["Old Form-2"] = @benchmarkable hodge_star(2, $earth, DiagonalHodge())

end

begin
    dec_op_suite["Geometric Hodge"]["New Form-1"] = @benchmarkable dec_hodge_star(1, $earth, GeometricHodge())
    dec_op_suite["Geometric Hodge"]["Old Form-1"] = @benchmarkable hodge_star(1, $earth, GeometricHodge())
end

begin
    dec_op_suite["Inverse Diagonal Hodge"]["New Form-0"] = @benchmarkable dec_inv_hodge_star(0, $earth, DiagonalHodge())
    dec_op_suite["Inverse Diagonal Hodge"]["Old Form-0"] = @benchmarkable inv_hodge_star(0, $earth, DiagonalHodge())

    dec_op_suite["Inverse Diagonal Hodge"]["New Form-1"] = @benchmarkable dec_inv_hodge_star(1, $earth, DiagonalHodge())
    dec_op_suite["Inverse Diagonal Hodge"]["Old Form-1"] = @benchmarkable inv_hodge_star(1, $earth, DiagonalHodge())

    dec_op_suite["Inverse Diagonal Hodge"]["New Form-2"] = @benchmarkable dec_inv_hodge_star(2, $earth, DiagonalHodge())
    dec_op_suite["Inverse Diagonal Hodge"]["Old Form-2"] = @benchmarkable inv_hodge_star(2, $earth, DiagonalHodge())
end

begin
    Random.seed!(7331)
    V_1 = rand(float_type, nv(earth))
    E_1, E_2 = rand(float_type, ne(earth)), rand(float_type, ne(earth))
    T_2 = rand(float_type, ntriangles(earth))

    dec_op_suite["Wedge Product"] = BenchmarkGroup()

    dec_op_suite["Wedge Product"]["New Form-0, Form-1"] = @benchmarkable dec_wedge_product(Tuple{0,1}, $earth)($V_1, $E_1)
    dec_op_suite["Wedge Product"]["Old Form-0, Form-1"] = @benchmarkable wedge_product(Tuple{0,1}, $earth, $V_1, $E_1)

    dec_op_suite["Wedge Product"]["New Form-1, Form-1"] = @benchmarkable dec_wedge_product(Tuple{1,1}, $earth)($E_1, $E_2)
    dec_op_suite["Wedge Product"]["Old Form-1, Form-1"] = @benchmarkable wedge_product(Tuple{1,1}, $earth, $E_1, $E_2)

    dec_op_suite["Wedge Product"]["New Form-0, Form-2"] = @benchmarkable dec_wedge_product(Tuple{0,2}, $earth)($V_1, $T_2)
    dec_op_suite["Wedge Product"]["Old Form-0, Form-2"] = @benchmarkable wedge_product(Tuple{0,2}, $earth, $V_1, $T_2)
end

begin
    Random.seed!(7331)
    V_1 = rand(float_type, nv(earth))
    E_1, E_2 = rand(float_type, ne(earth)), rand(float_type, ne(earth))
    T_2 = rand(float_type, ntriangles(earth))

    dec_op_suite["Wedge Product Computation"] = BenchmarkGroup()

    wdg01 = dec_wedge_product(Tuple{0,1}, earth)
    wdg11 = dec_wedge_product(Tuple{1,1}, earth)
    wdg02 = dec_wedge_product(Tuple{0,2}, earth)

    dec_op_suite["Wedge Product Computation"]["New Form-0, Form-1"] = @benchmarkable wdg01($V_1, $E_1)
    dec_op_suite["Wedge Product Computation"]["New Form-1, Form-1"] = @benchmarkable wdg11($E_1, $E_2)
    dec_op_suite["Wedge Product Computation"]["New Form-0, Form-2"] = @benchmarkable wdg02($V_1, $T_2)
end

# tune!(dec_op_suite)

@info "Running DEC Operator Benchmarks"

dec_op_results = run(dec_op_suite, verbose = true, seconds = 1)

for op in sort(Base.collect(keys(dec_op_results)))
    test = median(dec_op_results[op])

    println("Operator: $op")
    for k in sort(Base.collect(keys(test)))
        t = test[k].time / 1e6
        m = test[k].memory / 1e6
        println("Variant: $k, [$t ms, $m MB]")
    end
    println("----------------------------------------------------------------")
end

@info "Beginning Dual Mesh Generation Benchmarks"
begin
    mesh_size = 100
    grid_spacings = [1.0, 0.8]#, 0.5, 0.4, 0.25, 0.2]
    point_type = Point2D
    benchmark_dual_meshes = map(gs -> triangulated_grid(mesh_size, mesh_size, gs, gs, point_type), grid_spacings);
end;
@info "Generated Primal Meshes"

function create_dual_mesh(s, point_type, center_type)
    sd = EmbeddedDeltaDualComplex2D{Bool,Float64, point_type}(s)
    subdivide_duals!(sd, center_type)
    sd;
end

dual_mesh_suite = BenchmarkGroup()
filler_name = "Grid Spacing"
begin
    for (i, grid_spacing) in enumerate(grid_spacings)
        dual_mesh_suite["Barycenter"][filler_name][grid_spacing] = @benchmarkable create_dual_mesh($(benchmark_dual_meshes[i]), $point_type, $(Barycenter())) gcsample=true seconds=10
        dual_mesh_suite["Circumcenter"][filler_name][grid_spacing] = @benchmarkable create_dual_mesh($(benchmark_dual_meshes[i]), $point_type, $(Circumcenter())) gcsample=true seconds=10
    end
end

@info "Running Dual Mesh Benchmarks"

dual_mesh_results = run(dual_mesh_suite, verbose = true)

for center in sort(Base.collect(keys(dual_mesh_results)))
    trial = median(dual_mesh_results[center][filler_name])

    println("Center: $center")
    for k in sort(Base.collect(keys(trial)), rev=true)
        t = trial[k].time / 1e9
        m = trial[k].memory / 1e9
        println("$k, [$t s, $m GB]")
    end
    println("----------------------------------------------------------------")
end

# ── Circumcenter performance: before/after comparison on Icosphere(8) ────────
#
# The "before" (old) implementation uses the generic Cayley-Menger matrix
# inversion for every simplex center computation.  The "after" (new)
# implementation uses closed-form formulas derived from the Gram matrix:
#   • edge        → midpoint (trivial)
#   • triangle    → 2×2 Cramer's rule  (avoids 4×4 Cayley-Menger inversion)
#   • tetrahedron → 3×3 Gram-matrix \  (avoids 5×5 Cayley-Menger inversion)
# ─────────────────────────────────────────────────────────────────────────────

using StaticArrays: SVector, StaticVector, MVector, SMatrix
using CombinatorialSpaces.SimplicialSets: cayley_menger

# Reference implementation identical to the pre-optimization code.
function geometric_center_cayley_menger(points::StaticVector{N}, ::Circumcenter) where N
    CM = cayley_menger(points...)
    inv_CM = inv(CM)
    barycentric_coords = SVector(ntuple(i -> inv_CM[1, i+1], Val(N)))
    mapreduce(*, +, barycentric_coords, points)
end

@info "Building Icosphere(8) primal mesh for circumcenter benchmark"
primal_ico8 = loadmesh(Icosphere(8))
orient!(primal_ico8)
@info "  nv=$(nv(primal_ico8))  ne=$(ne(primal_ico8))  ntri=$(ntriangles(primal_ico8))"

# Build a dual complex and pre-populate primal points so we can collect
# representative triangle vertex triples for the micro-benchmark.
let _sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}(primal_ico8)
    subdivide_duals!(_sd, Barycenter())
    global tri_pts_ico8 = map(triangles(_sd)) do t
        p1, p2, p3 = triangle_vertices(_sd, t)
        MVector{3,Point3d}(_sd[p1,:point], _sd[p2,:point], _sd[p3,:point])
    end
end
@info "  Collected $(length(tri_pts_ico8)) triangle point triples"

# --- micro-benchmark: geometric_center per triangle ---
circumcenter_suite = BenchmarkGroup()

circumcenter_suite["new (Gram-matrix)"]["triangle"] = @benchmarkable begin
    for pts in $tri_pts_ico8
        geometric_center(pts, Circumcenter())
    end
end seconds=5 evals=3

circumcenter_suite["old (Cayley-Menger)"]["triangle"] = @benchmarkable begin
    for pts in $tri_pts_ico8
        geometric_center_cayley_menger(pts, Circumcenter())
    end
end seconds=5 evals=3

# --- macro-benchmark: full subdivide_duals! on Icosphere(8) ---
function dualize_ico8(primal, center)
    sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}(primal)
    subdivide_duals!(sd, center)
    sd
end

circumcenter_suite["new (Gram-matrix)"]["subdivide_duals! Icosphere(8)"] =
    @benchmarkable dualize_ico8($primal_ico8, Circumcenter()) seconds=10 evals=1

circumcenter_suite["old (Cayley-Menger)"]["subdivide_duals! Icosphere(8)"] =
    @benchmarkable begin
        sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}($primal_ico8)
        # Temporarily rebind geometric_center to the old Cayley-Menger path by
        # running it manually inside the loop (mirrors subdivide_duals_2d! logic).
        for v in vertices(sd)
            sd[v, :dual_point] = sd[v, :point]
        end
        pt_arr = MVector{2,Point3d}(undef)
        for e in edges(sd)
            p1, p2 = edge_vertices(sd, e)
            pt_arr[1] = sd[p1,:point]; pt_arr[2] = sd[p2,:point]
            sd[sd[e,:edge_center],:dual_point] =
                geometric_center_cayley_menger(SVector{2,Point3d}(pt_arr), Circumcenter())
        end
        pt_arr3 = MVector{3,Point3d}(undef)
        for t in triangles(sd)
            p1,p2,p3 = triangle_vertices(sd, t)
            pt_arr3[1]=sd[p1,:point]; pt_arr3[2]=sd[p2,:point]; pt_arr3[3]=sd[p3,:point]
            sd[sd[t,:tri_center],:dual_point] =
                geometric_center_cayley_menger(SVector{3,Point3d}(pt_arr3), Circumcenter())
        end
        sd
    end seconds=10 evals=1

@info "Running Circumcenter before/after benchmarks on Icosphere(8)"
cc_results = run(circumcenter_suite, verbose=true)

println("\n════════════════════════════════════════════════════════════════")
println("  Circumcenter performance: before vs after  [Icosphere(8)]")
println("════════════════════════════════════════════════════════════════")
for variant in ["triangle", "subdivide_duals! Icosphere(8)"]
    t_new = median(cc_results["new (Gram-matrix)"][variant])
    t_old = median(cc_results["old (Cayley-Menger)"][variant])
    ratio = t_old.time / t_new.time
    println("Benchmark : $variant")
    println("  NEW  time=$(BenchmarkTools.prettytime(t_new.time))  mem=$(BenchmarkTools.prettymemory(t_new.memory))")
    println("  OLD  time=$(BenchmarkTools.prettytime(t_old.time))  mem=$(BenchmarkTools.prettymemory(t_old.memory))")
    @printf("  Speed-up : %.2fx\n\n", ratio)
end

end
