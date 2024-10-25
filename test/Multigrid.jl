module TestMultigrid
using Krylov, CombinatorialSpaces, LinearAlgebra, Test
using GeometryBasics: Point3, Point2
const Point3D = Point3{Float64}
const Point2D = Point2{Float64}

s = triangulated_grid(1,1,1/4,sqrt(3)/2*1/4,Point3D)
fs = reverse(repeated_subdivisions(4,s,triforce_subdivision_map));
sses = map(fs) do f dom(f) end
push!(sses,s)
sds = map(sses) do s dualize(s,Circumcenter()) end
Ls = map(sds) do sd ∇²(0,sd) end
ps = transpose.(as_matrix.(fs))
rs = transpose.(ps)./4.0 #4 is the biggest row sum that occurs for triforce, this is not clearly the correct scaling

u0 = zeros(nv(sds[1]))
b = Ls[1]*rand(nv(sds[1])) #put into range of the Laplacian for solvability
md = MultigridData(Ls,rs,ps,3) #3,10 chosen empirically, presumably there's deep lore and chaos here
u = multigrid_vcycles(u0,b,md,5)
#@info "Relative error for V: $(norm(Ls[1]*u-b)/norm(b))"

@test norm(Ls[1]*u-b)/norm(b) < 10^-6
u = multigrid_wcycles(u0,b,md,5)
#@info "Relative error for W: $(norm(Ls[1]*u-b)/norm(b))"
@test norm(Ls[1]*u-b)/norm(b) < 10^-7
u = full_multigrid(b,md,5)
@test norm(Ls[1]*u-b)/norm(b) < 10^-5
#@info "Relative error for FMG_V: $(norm(Ls[1]*u-b)/norm(b))"
u = full_multigrid(b,md,5,cg,2)
@test norm(Ls[1]*u-b)/norm(b) < 10^-7
#@info "Relative error for FMG_W: $(norm(Ls[1]*u-b)/norm(b))"
end