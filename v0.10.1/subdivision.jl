using CombinatorialSpaces
using GeometryBasics
using GeometryBasics: Point3d
using CairoMakie, StaticArrays
using Colors

Blue =  colorant"#578AB7"
Orange =  colorant"#E0A578"
Green =  colorant"#ADCF94"
Beige =  colorant"#F9EECB"
Teal =  colorant"#B3DBD9"
Purple =  colorant"#D7C9F1"


s = EmbeddedDeltaSet2D{Float64, Point3d}()
add_vertices!(s, 3, point=[Point3d(0,0,0), Point3d(1,0,0), Point3d(0.5,0.866,0)])
glue_sorted_triangle!(s, 1,2,3)
t = binary_subdivision(s)
u = binary_subdivision(t)
v = binary_subdivision(u)
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="Initial Triangle", aspect=1.25)
wireframe!(ax, s, linewidth=6, color=Blue)
f
save("0level_binary_subdivision.pdf",f)
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="1 Level of Binary Subdivision", aspect=1.25)
wireframe!(ax, t, linewidth=4, color=Orange)
wireframe!(ax, s, linewidth=6, color=Blue)
f
save("1level_binary_subdivision.pdf",f)
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="2 Levels of Binary Subdivision", aspect=1.25)
wireframe!(ax, u, linewidth=4, color=Green)
wireframe!(ax, t, linewidth=6, color=Orange)
wireframe!(ax, s, linewidth=6, color=Blue)
f
save("2level_binary_subdivision.pdf",f)
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="3 Levels of Binary Subdivision", aspect=1.25)
wireframe!(ax, v, linewidth=4, color=Teal)
wireframe!(ax, u, linewidth=6, color=Green)
wireframe!(ax, t, linewidth=6, color=Orange)
wireframe!(ax, s, linewidth=6, color=Blue)
f
save("3level_binary_subdivision.pdf",f)

s = EmbeddedDeltaSet2D{Float64, Point3d}()
add_vertices!(s, 3, point=[Point3d(0,0,0), Point3d(1,0,0), Point3d(0.5,0.866,0)])
glue_sorted_triangle!(s, 1,2,3)
t = binary_subdivision(s)
u = binary_subdivision(t)
v = binary_subdivision(u)
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="Binary Subdivision", aspect=1.25)
wireframe!(ax, v, linewidth=2, color=Teal)
wireframe!(ax, u, linewidth=4, color=Green)
wireframe!(ax, t, linewidth=6, color=Orange)
wireframe!(ax, s, linewidth=8, color=Blue)
f
save("binary_subdivision.pdf",f)



s = EmbeddedDeltaSet2D{Float64, Point3d}()
add_vertices!(s, 3, point=[Point3d(0,0,0), Point3d(1,0,0), Point3d(0.5,0.866,0)])
glue_sorted_triangle!(s, 1,2,3)
t = cubic_subdivision(s)
u = cubic_subdivision(t)
v = cubic_subdivision(u)
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="Initial Triangle", aspect=1.25)
wireframe!(ax, s, linewidth=6, color=Blue)
f
save("0level_cubic_subdivision.pdf",f)
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="1 Level of Cubic Subdivision", aspect=1.25)
wireframe!(ax, t, linewidth=4, color=Orange)
wireframe!(ax, s, linewidth=6, color=Blue)
f
save("1level_cubic_subdivision.pdf",f)
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="2 Levels of Cubic Subdivision", aspect=1.25)
wireframe!(ax, u, linewidth=4, color=Green)
wireframe!(ax, t, linewidth=6, color=Orange)
wireframe!(ax, s, linewidth=6, color=Blue)
f
save("2level_cubic_subdivision.pdf",f)
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="3 Levels of Cubic Subdivision", aspect=1.25)
wireframe!(ax, v, linewidth=4, color=Teal)
wireframe!(ax, u, linewidth=6, color=Green)
wireframe!(ax, t, linewidth=6, color=Orange)
wireframe!(ax, s, linewidth=6, color=Blue)
f
save("3level_cubic_subdivision.pdf",f)

s = EmbeddedDeltaSet2D{Float64, Point3d}()
add_vertices!(s, 3, point=[Point3d(0,0,0), Point3d(1,0,0), Point3d(0.5,0.866,0)])
glue_sorted_triangle!(s, 1,2,3)
t = cubic_subdivision(s)
u = cubic_subdivision(t)
v = cubic_subdivision(u)
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="Cubic Subdivision", aspect=1.25)
wireframe!(ax, v, linewidth=2, color=Teal)
wireframe!(ax, u, linewidth=4, color=Green)
wireframe!(ax, t, linewidth=6, color=Orange)
wireframe!(ax, s, linewidth=8, color=Blue)
f
save("cubic_subdivision.pdf",f)