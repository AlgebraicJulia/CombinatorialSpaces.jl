module TestDualMeshGen

using Test
using SparseArrays
using LinearAlgebra
using CombinatorialSpaces
using GeometryBasics: Point2
using Catlab: copy_parts!

Point2D = Point2{Float64}

primal_line = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(primal_line, 3, point=[Point2D(1,0), Point2D(0,0), Point2D(0,2)])
add_edges!(primal_line, [1,2], [2,3])

primal_cycle = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(primal_cycle, 3, point=[Point2D(1,0), Point2D(0,0), Point2D(0,1)])
add_edges!(primal_cycle, [1,2,3], [2,3,1])

dual_meshes_1D = [primal_line, primal_cycle]

primal_triangle = EmbeddedDeltaSet2D{Bool,Point2D}()
add_vertices!(primal_triangle, 3, point=[Point2D(1,0), Point2D(0,0), Point2D(0,1)])
add_edges!(primal_triangle, [1,2,1], [2,3,3])
glue_triangle!(primal_triangle, 1, 2, 3)

dual_meshes_2D = [(loadmesh ∘ Icosphere).(1:2)...,
               loadmesh(Rectangle_30x10()),
               triangulated_grid(10,10,8,8,Point3D), primal_triangle];

for s in dual_meshes_1D
    sd_c = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(s, FastMesh())
    subdivide_duals!(sd_c, FastMesh(), Barycenter())

    sd = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(s)
    subdivide_duals!(sd, Barycenter())

    @test sd_c == sd
end

for s in dual_meshes_2D
    sd_c = EmbeddedDeltaDualComplex2D{Bool,Float64, Point2D}(s, FastMesh())
    subdivide_duals!(sd_c, FastMesh(), Barycenter())

    sd = EmbeddedDeltaDualComplex2D{Bool,Float64, Point2D}(s)
    subdivide_duals!(sd, Barycenter())

    @test sd_c == sd
end
end