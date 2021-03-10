using CombinatorialSpaces
using StaticArrays: SVector
using GLMakie
const Point3D = SVector{3,Float64}

GLMakie.AbstractPlotting.inline!(false)

cat = EmbeddedDeltaSet2D("cat.obj")

sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(cat);
subdivide_duals!(sd, Barycenter())

fig, ax, ob = wireframe(sd;  color=(:blue, 1), transparency=true, linewidth=3)
wireframe!(cat; color=(:red, 1), transparency=true, linewidth=6)
display(fig)
println("Press ENTER to end display server")
readline()
