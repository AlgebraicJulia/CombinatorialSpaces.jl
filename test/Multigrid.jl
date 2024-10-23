module TestMultigrid
using Krylov, CombinatorialSpaces, LinearAlgebra, Test

s = triangulated_grid(1,1,1/4,sqrt(3)/2*1/4,Point3D)
fs = reverse(repeated_subdivisions(4,s,triforce_subdivision_map));
sses = map(fs) do f dom(f).delta_set end
push!(sses,s)
sds = map(sses) do s dualize(s,Circumcenter()) end
Ls = map(sds) do sd ∇²(0,sd) end
ps = transpose.(as_matrix.(fs))
rs = transpose.(ps)./4.0 #4 is the biggest row sum that occurs for triforce, this is not clearly the correct scaling

u0 = zeros(nv(sds[1]))
b = Ls[1]*rand(nv(sds[1])) #put into range of the Laplacian for solvability
u = multigrid_vcycles(u0,b,Ls,rs,ps,3,10) #3,10 chosen empirically, presumably there's deep lore and chaos here

  @test norm(Ls[1]*u-b)/norm(b) < 10^-6
end