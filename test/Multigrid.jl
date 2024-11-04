module TestMultigrid
using Krylov, CombinatorialSpaces, LinearAlgebra, Test
using GeometryBasics: Point3, Point2
const Point3D = Point3{Float64}
const Point2D = Point2{Float64}

s = triangulated_grid(1,1,1/4,sqrt(3)/2*1/4,Point3D,false)
series = PrimitiveGeometricMapSeries(s, binary_subdivision_map, 4);

md = MultigridData(series, sd -> ∇²(0, sd), 3)

sd = finest_mesh(series)

u0 = zeros(nv(sd))
b = Ls[1]*rand(nv(sd)) #put into range of the Laplacian for solvability
md = MultigridData(Ls,rs,ps,3) #3,10 chosen empirically, presumably there's deep lore and chaos here
u = multigrid_vcycles(u0,b,md,5)
#@info "Relative error for V: $(norm(Ls[1]*u-b)/norm(b))"

@test norm(Ls[1]*u-b)/norm(b) < 10^-5
u = multigrid_wcycles(u0,b,md,5)
#@info "Relative error for W: $(norm(Ls[1]*u-b)/norm(b))"
@test norm(Ls[1]*u-b)/norm(b) < 10^-7
u = full_multigrid(b,md,5)
@test norm(Ls[1]*u-b)/norm(b) < 10^-4
#@info "Relative error for FMG_V: $(norm(Ls[1]*u-b)/norm(b))"
u = full_multigrid(b,md,5,cg,2)
@test norm(Ls[1]*u-b)/norm(b) < 10^-6
#@info "Relative error for FMG_W: $(norm(Ls[1]*u-b)/norm(b))"
end
