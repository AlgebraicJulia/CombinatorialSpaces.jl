using CombinatorialSpaces
using CairoMakie
using ACSets

s = EmbeddedDeltaSet3D{Bool,Point3d}()
add_vertices!(s, 4, point=[Point3d(1,1,1), Point3d(1,-1,-1),
  Point3d(-1,1,-1), Point3d(-1,-1,1)])
glue_tetrahedron!(s, 1, 2, 3, 4)

sd = binary_subdivision(s)

# 4 -- 6 -- 3 -- 6 -- 4
#  \  / \ /  \ /  \ /
#   9-- 8 -- 5 -- 7
#    \ / \  /  \ /
#    1 -- 10 -- 2
#     \   /\   /
#       9 -- 7
#       \   /
#         4

# Take outer most tetrahedra, connected to one of the original vertices
glue_sorted_tetrahedron!(sd, 1, 8, 9, 10)
glue_sorted_tetrahedron!(sd, 2, 5, 7, 10)
glue_sorted_tetrahedron!(sd, 3, 5, 6, 8)
glue_sorted_tetrahedron!(sd, 4, 6, 7, 9)

# Take infer most tetrahedra, only connected to midpoints, need to take interior diagonal
# Interior diagonal in this case is 6 -- 10

glue_sorted_tetrahedron!(sd, 6, 10, 8, 9)
glue_sorted_tetrahedron!(sd, 6, 10, 7, 9)
glue_sorted_tetrahedron!(sd, 6, 10, 8, 5)
glue_sorted_tetrahedron!(sd, 6, 10, 7, 5)

fig = Figure();
lscene = LScene(fig[1, 1], scenekw = (lights = [],))
p = CairoMakie.wireframe!(lscene, sd)
fig

s = EmbeddedDeltaSet3D{Bool,Point3d}()
add_vertices!(s, 8, point=[
  Point3d(0,1,0), Point3d(0,0,0), Point3d(1,1,0), Point3d(1,0,0),
  Point3d(0,1,1), Point3d(0,0,1), Point3d(1,1,1), Point3d(1,0,1)])
# See Table 3.1 "Mesh connectivity".
glue_sorted_tetrahedron!(s, 3, 5, 4, 2)
glue_sorted_tetrahedron!(s, 7, 6, 8, 4)
glue_sorted_tetrahedron!(s, 5, 6, 7, 4)
glue_sorted_tetrahedron!(s, 3, 5, 7, 4)
glue_sorted_tetrahedron!(s, 5, 6, 4, 2)
glue_sorted_tetrahedron!(s, 1, 5, 3, 2)
s[:edge_orientation] = true
s[:tri_orientation] = true
s[:tet_orientation] = true
orient!(s)

sd = EmbeddedDeltaDualComplex3D{Bool,Float64,Point3d}(s)
subdivide_duals!(sd, Barycenter())

d₀_mat = d(0,sd)
star₁_mat = hodge_star(1,sd,DiagonalHodge())
dual_d₀_mat = dual_derivative(2,sd)
inv_star₃_mat = inv_hodge_star(0,sd,DiagonalHodge())

lap_mat = inv_star₃_mat * dual_d₀_mat * star₁_mat * d₀_mat

C₀ = map(p -> (p[1] + 1)^2, sd[:point])
C = deepcopy(C₀)
D = 0.005

# t_end = 250
# Cs = zeros(t_end, length(C))
# for t in 1:t_end
#   dt_C = D * lap_mat*C
#   C .+= dt_C
#   Cs[t, :] .= C
# end

fig = Figure();
lscene = LScene(fig[1, 1], scenekw = (lights = [],))
p = CairoMakie.mesh!(lscene, s, color = C₀, colormap = :jet)
fig

# s = EmbeddedDeltaSet3D{Bool,Point3d}()
# add_vertices!(s, 4, point=[Point3d(1,1,1), Point3d(1,-1,-1),
#   Point3d(-1,1,-1), Point3d(-1,-1,1)])
# glue_tetrahedron!(s, 1, 2, 3, 4)
# fig = Figure();
# lscene = LScene(fig[1, 1], scenekw = (lights = [],))
# p = CairoMakie.mesh!(lscene, s, color = [1,2,3,4])
# fig

# function save_heat()
#   time = Observable(1)
#   fig = Figure(title = @lift("Temperature at $($time)"))
#   ax = lscene = LScene(fig[1, 1], scenekw = (lights = [],))
#   msh = CairoMakie.mesh!(ax, s,
#     color=@lift(Cs[$time, :]), colorrange = (1, 3),
#     colormap=:jet)

#   Colorbar(fig[1,2], msh)
#   record(fig, "heat.mp4", 1:t_end; framerate = 15) do t
#     time[] = t
#   end
# end
# save_heat()
