module TestMultigrid
using Krylov, CombinatorialSpaces, LinearAlgebra, Test
using GeometryBasics: Point3, Point2
const Point3D = Point3{Float64}
const Point2D = Point2{Float64}

s = triangulated_grid(1,1,1/4,sqrt(3)/2*1/4,Point3D,false)
series = PrimalGeometricMapSeries(s, binary_subdivision_map, 4);

md = MultigridData(series, sd -> ∇²(0, sd), 3) #3, (and 5 below) chosen empirically, presumably there's deep lore and chaos here
sd = finest_mesh(series)
L = first(md.operators)
b = L*rand(nv(sd)) #put into range of the Laplacian for solvability

mgv_lapl = dec_Δ⁻¹(Val{0}, series)
u = mgv_lapl(b)
@test norm(L*u-b)/norm(b) < 10^-6

u0 = zeros(nv(sd))
u = multigrid_vcycles(u0,b,md,5)
#@info "Relative error for V: $(norm(L*u-b)/norm(b))"

@test norm(L*u-b)/norm(b) < 10^-5
u = multigrid_wcycles(u0,b,md,5)
#@info "Relative error for W: $(norm(L*u-b)/norm(b))"
@test norm(L*u-b)/norm(b) < 10^-7
u = full_multigrid(b,md,5)
@test norm(L*u-b)/norm(b) < 10^-4
#@info "Relative error for FMG_V: $(norm(L*u-b)/norm(b))"
u = full_multigrid(b,md,5,cg,2)
@test norm(L*u-b)/norm(b) < 10^-6
#@info "Relative error for FMG_W: $(norm(L*u-b)/norm(b))"
end
