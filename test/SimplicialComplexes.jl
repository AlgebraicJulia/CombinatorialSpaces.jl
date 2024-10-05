#module TestSimplicialComplexes
using Test 
using CombinatorialSpaces
using Catlab:@acset
using LinearAlgebra, Krylov
using GeometryBasics: Point2, Point3
using SparseArrays
Point2D = Point2{Float64}

# Triangulated commutative square.
ss = DeltaSet2D()
add_vertices!(ss, 4)
glue_triangle!(ss, 1, 2, 3)
glue_triangle!(ss, 1, 4, 3)

sc = SimplicialComplex(ss)
@test nv(sc) == 4 && ne(sc) == 5 && ntriangles(sc) == 2
sc′ = SimplicialComplex(DeltaSet2D,sc.cache).delta_set
@test nv(sc′) == 4 && ne(sc′) == 5 && ntriangles(sc′) == 2 #identifies this up to iso
#awkward height=0 edge case, technically can think of the empty sset as -1-dimensional.
sc′′=SimplicialComplex(Trie{Int,Int}()) 
@test dimension(sc′′) == 0 && nv(sc′′) == 0

vl = VertexList(ss,Simplex{2}(1))
@test vl.vs == [1,2,3]
@test has_simplex(sc,vl)
@test !has_simplex(sc,VertexList([1,2,4]))
@test sc[vl] == Simplex{2}(1)

vl′ = VertexList([1,2])∪VertexList([2,3])
@test has_simplex(sc,vl′)
@test !has_simplex(sc,VertexList([1,2])∪VertexList([2,4]))

p = [0.2,0,0.5,0.3]
q = [0,0.2,0.5,0.3]
r = [0.5,0,0,0.5]
s = [0.5,0,0.5,0]
t = [0,0.5,0,0.5]
@test has_point(sc,p) && !has_point(sc,q) 
@test has_span(sc,[r, s]) && !has_span(sc,[r, t])

Δ⁰ = SimplicialComplex(@acset DeltaSet0D begin V=1 end)
Δ¹ = SimplicialComplex(@acset DeltaSet1D begin V=2; E=1; ∂v0 = [2]; ∂v1 = [1] end)
f = GeometricMap(Δ⁰,Δ¹,[1/3,2/3])
A  = [0.2 0.4
      0   0
      0.5 0
      0.3 0.6]
g = GeometricMap(Δ¹,sc,A)
@test A == as_matrix(g)
h = compose(f,g)
@test as_matrix(h) ≈ [1/3, 0, 1/6, 1/2]
isc = id(sc)
@test as_matrix(h) == as_matrix(compose(h,isc))

primal_s = EmbeddedDeltaSet2D{Bool,Point2D}()
add_vertices!(primal_s, 3, point=[Point2D(0,0), Point2D(1,0), Point2D(0,1)])
glue_triangle!(primal_s, 1, 2, 3, tri_orientation=true)
primal_s[:edge_orientation] = true
#f,g = subdivision_maps(primal_s)
#@test as_matrix(compose(g,f)) = I(3)*1.0

fake_laplacian_builder(s::HasDeltaSet) = I(nv(s))*1.0
fake_laplacian_builder(s::SimplicialComplex) = I(nv(s))*1.0
laplacian_builder(s::HasDeltaSet) = -∇²(0,s)
function heat_equation_multiscale(primal_s::HasDeltaSet, laplacian_builder::Function, initial::Vector,
  step_size::Float64, n_steps_inner::Int, n_steps_outer::Int)
  f, g = subdivision_maps(primal_s)
  sc_fine, sc_coarse = dom(f), codom(f)
  f, g = as_matrix.([f, g])
  dual_s_fine,dual_s_coarse = dualize.([sc_fine.delta_set,sc_coarse.delta_set])
  subdivide_duals!(dual_s_fine,Barycenter())
  subdivide_duals!(dual_s_coarse,Barycenter())
  Δ_fine, Δ_coarse = laplacian_builder.([dual_s_fine,dual_s_coarse])
  u_fine = transpose(initial)
  u_coarse = u_fine * g
  for i in 1:n_steps_outer
    # do a fine step
    u_fine += step_size * u_fine * Δ_fine
    u_coarse = u_fine * g
    for j in 1:n_steps_inner
      u_coarse += step_size * u_coarse * Δ_coarse
    end
    u_fine = u_coarse * f
  end
  transpose(u_fine)
end

#ACTION ITEM: Reproduce Golub van Loan as doc page
#Probably work on the hexagonal grid, compare "first order finite difference scheme"
#Make sure to think about the difference between primal 0-form and dual 2-form

#XX might be handy to make this an iterator
"""
A weighted Jacobi iteration iterating toward a solution of 
Au=b.
For Poisson's equation on a grid, it's known that ω=2/3 is optimal.
Experimentally, ω around .85 is best for a subdivided 45-45-90 triangle.
In general this will converge for all u₀ with ω=1 if A is strictly 
diagonally dominant.
I'm not sure about convergence for general ω.
See Golub Van Loan 11.2 and 11.6.2. 
"""
function WJ(A,b,ω)
  D = SparseMatrixCSC(diagm(diag(A)))
  c = ω * (D \ b)
  G = SparseMatrixCSC((1-ω)*I-ω * (D\(A-D)))
  G,c
end
spectral_radius(A) = maximum(abs.(eigvals(A)))
sub_spectral_radius(A) = sub_max(abs.(eigvals(A)))
function sub_max(v)
  length(v)>1 || error("don't") 
  a,b = sort([v[1],v[2]])
  for i in v[3:end]
    if i > b
      a,b = b,i
    elseif i > a
      a = i
    end
  end
  a
end
function it(G,c,u₀,n) 
  u = u₀ 
  for i in 1:n u = G*u+c end 
  u
end
u₀ = zeros(7)
A = rand(7,7)+diagm(fill(10.0,7))
b = ones(7)
G,c = WJ(A,b,1)
@test norm(A*it(G,c,u₀,25)- b)<10^-10

"""
Solve ∇²u = b on a `depth`-fold subdivided simplical complex using a multigrid V-cycle.
"""
function multigrid_vcycle(u,b,primal_s,depth,smoothing_steps)
  center(_multigrid_vcycle(u,b,multigrid_setup(primal_s,depth)...,smoothing_steps))
end
function multigrid_setup(primal_s,depth,alg=Barycenter())
  prim = primal_s
  duco = dualize(prim,alg)
  pt = typeof(prim)()
  add_vertex!(pt)
  laplacians = [∇²(0,duco)]
  interpolations = [GeometricMap(SimplicialComplex(prim),SimplicialComplex(pt),[1. 1. 1.])]
  for i in 1:depth 
    duco = dualize(prim,alg)
    dual = extract_dual(duco)
    mat = zeros(Float64,nv(prim),nv(dual))
    pvs = map(i->primal_vertices(duco,i),1:nv(dual))
    weights = 1 ./(length.(pvs))
    for j in 1:nv(dual)
      for v in pvs[j]
        mat[v,j] = weights[j]
      end
    end
    f=GeometricMap(SimplicialComplex(dual),SimplicialComplex(prim),mat)
    prim = dual
    duco = dualize(prim,alg)
    L = ∇²(0,duco)
    push!(laplacians,L)
    push!(interpolations,f)
  end
  reverse(laplacians), reverse(as_matrix.(interpolations[2:end]))
end
function _multigrid_vcycle(u,b,laplacians,interpolations,prolongations,smoothing_steps,cycles=1)
  cycles == 0 && return u
  u = gmres(laplacians[1],b,u,itmax=smoothing_steps)[1]
  if length(laplacians) == 1 #on coarsest grid
    return center(u)
  end
  #smooth, update error, restrict, recurse, prolong, smooth
  r_f = b - laplacians[1]*u #residual
  r_c = interpolations[1] * r_f #restrict
  z = _multigrid_vcycle(zeros(size(r_c)),r_c,laplacians[2:end],interpolations[2:end],prolongations[2:end],smoothing_steps,cycles) #recurse
  u += prolongations[1] * z #prolong. 
  u = gmres(laplacians[1],b,u,itmax=smoothing_steps)[1] #smooth
  _multigrid_vcycle(u,b,laplacians,interpolations,prolongations,smoothing_steps,cycles-1)
end
center(u) = u .- sum(u)/length(u)
row_normalize(A) = A./(sum.(eachrow(A)))
re(u,b) = norm(u-b)/norm(b)

N = 4
function test_vcycle_heat(N)
  ls, is = multigrid_setup(primal_s,N)
  n_fine = size(ls[1],1)
  b = ls[1]*[i for i in 1:n_fine]
  u = zeros(n_fine)
  tts = []
  for i in 1:n_fine
    if i % 10 == 0 println(i) end
    push!(tts,re(ls[1]*(gmres(ls[1],b,itmax=i)[1]),b)/re(ls[1]*(_multigrid_vcycle(u,b,ls,is,i)),b))
  end
  tts
end

square_laplacian(N) = diagm(0 => 2*ones(N),1 =>-1*ones(N-1),-1 =>-1*ones(N-1))
function sparse_square_laplacian(k)
  N,h = 2^k-1, 1/(2^k)
  A = spzeros(N,N)
  for i in 1:N
    A[i,i] = 2
    if i > 1 A[i,i-1] = -1 end
    if i < N A[i,i+1] = -1 end
  end
  1/h^2 * A
end
function sparse_restriction(k)
  N,M = 2^k-1, 2^(k-1)-1
  A = spzeros(M,N)
  for i in 1:M
    A[i,2i-1:2i+1] = [1,2,1]
  end
  0.25*A
end
sparse_prolongation(k) = 2*transpose(sparse_restriction(k))
#Note that for k=10 a single v-cycle maximizes payoff at around i = 58!
#You get insane accuracy with smoothing_steps=7 and cycles=3. Jesus.
function test_vcycle_square(k,s,c)
  b=rand(2^k-1)
  N = 2^k-1 
  ls = reverse([sparse_square_laplacian(k′) for k′ in 1:k])
  is = reverse([sparse_restriction(k′) for k′ in 2:k])
  ps = reverse([sparse_prolongation(k′) for k′ in 2:k])
  u = zeros(N)
  re(ls[1]*_multigrid_vcycle(u,b,ls,is,ps,s,c),b)
end
#This takes a few dozen seconds on a cheap machine and would take
#maybe days for weighted Jacobi or GMRES (although it's a tridiagonal 
#matrix so there's a faster way in this case))
#Most of the work is in the setup; to go bigger than this you'd probably
#want to compute the coarser matrices from the finer by indexing instead
#of building them all manually.
@test test_vcycle_square(15,10,3)< 10^-8