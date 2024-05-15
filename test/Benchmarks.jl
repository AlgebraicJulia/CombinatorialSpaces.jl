module Benchmarks

using Catlab
using Catlab.Graphics
using CombinatorialSpaces
using LinearAlgebra
using BenchmarkTools
using Random
using GeometryBasics: Point2, Point3
Point2D = Point2{Float64}
Point3D = Point3{Float64}

@info "Beginning DEC Operator Benchmarks"
begin 
    mesh_size = 5
    float_type::DataType = Float64
    primal_earth = loadmesh(Icosphere(mesh_size))
    orient!(primal_earth);
    earth = EmbeddedDeltaDualComplex2D{Bool,float_type,Point3D}(primal_earth);
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

suite = BenchmarkGroup()

begin
    suite["Exterior Derivative"] = BenchmarkGroup()

    suite["Exterior Derivative"]["New Form-0"] = @benchmarkable dec_differential(0, $earth)
    suite["Exterior Derivative"]["Old Form-0"] = @benchmarkable d(0, $earth)

    suite["Exterior Derivative"]["New Form-1"] = @benchmarkable dec_differential(1, $earth)
    suite["Exterior Derivative"]["Old Form-1"] = @benchmarkable d(1, $earth)
end

begin
    suite["Boundary"] = BenchmarkGroup()

    suite["Boundary"]["New Form-1"] = @benchmarkable dec_boundary(1, $earth)
    suite["Boundary"]["Old Form-1"] = @benchmarkable ∂(1, $earth)

    suite["Boundary"]["New Form-2"] = @benchmarkable dec_boundary(2, $earth)
    suite["Boundary"]["Old Form-2"] = @benchmarkable ∂(2, $earth)
end

begin
    suite["Dual Derivative"] = BenchmarkGroup()

    suite["Dual Derivative"]["New Dual-Form-0"] = @benchmarkable dec_dual_derivative(0, $earth)
    suite["Dual Derivative"]["Old Dual-Form-0"] = @benchmarkable dual_derivative(0, $earth)

    suite["Dual Derivative"]["New Dual-Form-1"] = @benchmarkable dec_dual_derivative(1, $earth)
    suite["Dual Derivative"]["Old Dual-Form-1"] = @benchmarkable dual_derivative(1, $earth)
end

begin
    suite["Diagonal Hodge"] = BenchmarkGroup()

    suite["Diagonal Hodge"]["New Form-0"] = @benchmarkable dec_hodge_star(0, $earth, DiagonalHodge())
    suite["Diagonal Hodge"]["Old Form-0"] = @benchmarkable hodge_star(0, $earth, DiagonalHodge())

    suite["Diagonal Hodge"]["New Form-1"] = @benchmarkable dec_hodge_star(1, $earth, DiagonalHodge())
    suite["Diagonal Hodge"]["Old Form-1"] = @benchmarkable hodge_star(1, $earth, DiagonalHodge())

    suite["Diagonal Hodge"]["New Form-2"] = @benchmarkable dec_hodge_star(2, $earth, DiagonalHodge())
    suite["Diagonal Hodge"]["Old Form-2"] = @benchmarkable hodge_star(2, $earth, DiagonalHodge())

end

begin
    suite["Geometric Hodge"] = BenchmarkGroup()

    suite["Geometric Hodge"]["New Form-1"] = @benchmarkable dec_hodge_star(1, $earth, GeometricHodge())
    suite["Geometric Hodge"]["Old Form-1"] = @benchmarkable hodge_star(1, $earth, GeometricHodge())
end

begin
    suite["Inverse Diagonal Hodge"] = BenchmarkGroup()

    suite["Inverse Diagonal Hodge"]["New Form-0"] = @benchmarkable dec_inv_hodge_star(0, $earth, DiagonalHodge())
    suite["Inverse Diagonal Hodge"]["Old Form-0"] = @benchmarkable inv_hodge_star(0, $earth, DiagonalHodge())

    suite["Inverse Diagonal Hodge"]["New Form-1"] = @benchmarkable dec_inv_hodge_star(1, $earth, DiagonalHodge())
    suite["Inverse Diagonal Hodge"]["Old Form-1"] = @benchmarkable inv_hodge_star(1, $earth, DiagonalHodge())

    suite["Inverse Diagonal Hodge"]["New Form-2"] = @benchmarkable dec_inv_hodge_star(2, $earth, DiagonalHodge())
    suite["Inverse Diagonal Hodge"]["Old Form-2"] = @benchmarkable inv_hodge_star(2, $earth, DiagonalHodge())
end

begin
    Random.seed!(7331)
    V_1 = rand(float_type, nv(earth))
    E_1, E_2 = rand(float_type, ne(earth)), rand(float_type, ne(earth))
    T_2 = rand(float_type, ntriangles(earth))

    suite["Wedge Product"] = BenchmarkGroup()

    suite["Wedge Product"]["New Form-0, Form-1"] = @benchmarkable dec_wedge_product(Tuple{0,1}, $earth)($V_1, $E_1)
    suite["Wedge Product"]["Old Form-0, Form-1"] = @benchmarkable wedge_product(Tuple{0,1}, $earth, $V_1, $E_1)

    suite["Wedge Product"]["New Form-1, Form-1"] = @benchmarkable dec_wedge_product(Tuple{1,1}, $earth)($E_1, $E_2)
    suite["Wedge Product"]["Old Form-1, Form-1"] = @benchmarkable wedge_product(Tuple{1,1}, $earth, $E_1, $E_2)

    suite["Wedge Product"]["New Form-0, Form-2"] = @benchmarkable dec_wedge_product(Tuple{0,2}, $earth)($V_1, $T_2)
    suite["Wedge Product"]["Old Form-0, Form-2"] = @benchmarkable wedge_product(Tuple{0,2}, $earth, $V_1, $T_2)
end

begin
    Random.seed!(7331)
    V_1 = rand(float_type, nv(earth))
    E_1, E_2 = rand(float_type, ne(earth)), rand(float_type, ne(earth))
    T_2 = rand(float_type, ntriangles(earth))

    suite["Wedge Product Computation"] = BenchmarkGroup()

    wdg01 = dec_wedge_product(Tuple{0,1}, earth)
    wdg11 = dec_wedge_product(Tuple{1,1}, earth)
    wdg02 = dec_wedge_product(Tuple{0,2}, earth)

    suite["Wedge Product Computation"]["New Form-0, Form-1"] = @benchmarkable wdg01($V_1, $E_1)
    suite["Wedge Product Computation"]["New Form-1, Form-1"] = @benchmarkable wdg11($E_1, $E_2)
    suite["Wedge Product Computation"]["New Form-0, Form-2"] = @benchmarkable wdg02($V_1, $T_2)
end

# tune!(suite)

@info "Running DEC Operator Benchmarks"

results = run(suite, verbose = true, seconds = 1)

for op in sort(collect(keys(results)))
    test = median(results[op])

    println("Operator: $op")
    for k in sort(collect(keys(test)))
        t = test[k].time / 1e6
        m = test[k].memory / 1e6
        println("Variant: $k, [$t ms, $m MB]")
    end
    println("----------------------------------------------------------------")
end

@info "Beginning Dual Mesh Generation Benchmarks"

s_precom = triangulated_grid(100, 100, 100, 100, Point3{Float64});
s = triangulated_grid(100, 100, 0.5, 0.5, Point3{Float64});
@info "Generated Primal Mesh"

function test(s)
    sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3{Float64}}(s)
    subdivide_duals!(sd, Barycenter())
    sd;
end

function fast_test(s)
    sd_c = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3{Float64}}(s, FastMesh())
    subdivide_duals!(sd_c, FastMesh(), Barycenter())
    sd_c;
end

@info "Original Dual Mesh Generation"
test(s_precom);
@time test(s);

GC.gc(true)

@info "New Dual Mesh Generation"
fast_test(s_precom);
@time fast_test(s);

@info "Done"

end