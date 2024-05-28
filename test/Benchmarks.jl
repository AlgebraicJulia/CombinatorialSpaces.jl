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

for op in sort(collect(keys(dec_op_results)))
    test = median(dec_op_results[op])

    println("Operator: $op")
    for k in sort(collect(keys(test)))
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

for center in sort(collect(keys(dual_mesh_results)))
    trial = median(dual_mesh_results[center][filler_name])

    println("Center: $center")
    for k in sort(collect(keys(trial)), rev=true)
        t = trial[k].time / 1e9
        m = trial[k].memory / 1e9
        println("$k, [$t s, $m GB]")
    end
    println("----------------------------------------------------------------")
end

end
