using AbstractPlotting
using CombinatorialSpaces
using StaticArrays: SVector
const Point3D = SVector{3,Float64}

AbstractPlotting.inline!(false)

cat = DeltaSet2D("cat.obj")

sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(cat);
subdivide_duals!(sd, Barycenter())

fig, ax, ob = CombinatorialSpaces.Visualization.plot_deltaset(sd;  color=(:blue, 1), transparency=true, linewidth=3)
CombinatorialSpaces.Visualization.plot_deltaset!(cat; color=(:red, 1), transparency=true, linewidth=6)
display(fig)
println("Press ENTER to end WebGL server")
readline()
