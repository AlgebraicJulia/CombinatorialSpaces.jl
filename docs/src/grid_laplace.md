# Solving Poisson's equation in multiscale

CombinatorialSpaces provides advanced capabilities for working with irregular and complex meshes
in up to three dimensions. For a first example of working across meshes of multiple scales at once,
we reproduce a 1-D Poisson equation example from Golub and van Loan's "Matrix Computations", 11.6.

## Poisson equation

In general, the Poisson equation asks for a function on a manifold ``M`` with boundary with a fixed Laplacian on the interior, satisfying
boundary conditions that may be given in various forms, such as the Dirichlet conditions:

```math
\Delta u = -f,u\!\mid_{\partial M} = f_0
```

In one dimension, on the interval ``[0,1]``, this specializes to the equation
```math
\frac{d^2u}{dx^2} = -f(x), u(0)=u_0, u(1)=u_1.
```

If we subdivide the interval into ``m`` congruent pieces of width ``h=1/m``, then we get the discretized equations
```math
\frac{u((i-1)h)-2u(ih)+u((i+1)h)}{h^2}\approx -f(ih)
```
for ``i\in \{1,\ldots,m-1\}``. Since ``u(0)=u_0,u(1)=u_1`` are given by the boundary conditions, we can move them to 
the right-hand side of the first and last equations, producing the linear system ``Au=b`` for 
```math
u=[u(h),u(2h),\ldots,u((m-1)h)],
```
```math
b=[h^2f(h)+u_0,h^2f(2h),\ldots,h^2f((m-1)h),h^2f(mh)+u_1], \text{ and }
```
```math
A=\left(\begin{matrix}
2&-1&0&0&\cdots&0\\
-1&2&-1&0&\cdots&0\\
0&-1&2&-1&\cdots&0\\
\vdots&&&&\vdots\\
0&\cdots&0&-1&2&-1\\
0&\cdots&0&0&-1&2
\end{matrix}\right)
```

We are thus led to consider the solution of  ``Au=b`` for this tridiagonal ``A``. Tridiagonal systems are easy to solve naively, 
of course, but this example also gives a nice illustration of the multi-grid method. The latter proceeds by mixing steps of solution
via some iterative solver with approximate corrections obtained on a coarser grid, and works particularly well for this equation
where there is a neat division between high-frequency and low-frequency contributors to the solution.

Specifically, we will proceed by restricting discretized functions from a grid of radius ``h`` to one of radius ``2h`` and
prolonging back from there, by taking the weighted average of values near a coarse-grid point, weighting the point itself double,
for restriction, and making the value at a fine-grid point not in the coarse grid average the adjacent coarse values for prolongation.
It's interesting to note that restriction after prolongation is not idempotent, but instead smears some heat around away from
where it started.

## The problem solved directly via multigrid

```@example gvl
using Random # hide
Random.seed!(77777) # hide
using SparseArrays
using LinearAlgebra
using CombinatorialSpaces

#The tridiagonal Laplacian discussed above, with single-variable method
#for power-of-2 grids. 
sparse_square_laplacian(k) = sparse_square_laplacian(2^k-1,1/(2^k))
function sparse_square_laplacian(N,h)
  A = spzeros(N,N)
  for i in 1:N
    A[i,i] = 2
    if i > 1 A[i,i-1] = -1 end
    if i < N A[i,i+1] = -1 end
  end
  1/h^2 * A
end
#The restriction matrix to half as fine a grid.
function sparse_restriction(k)
  N,M = 2^k-1, 2^(k-1)-1
  A = spzeros(M,N)
  for i in 1:M
    A[i,2i-1:2i+1] = [1,2,1]
  end
  1/4*A
end
#The prolongation matrix from coarse to fine.
sparse_prolongation(k) = 2*transpose(sparse_restriction(k))

sparse_square_laplacian(3)
```
```@example gvl
sparse_restriction(3)
```
```@example gvl
sparse_prolongation(3)
```

Here is a function that sets up and runs a v-cycle for the 
Poisson problem on a mesh with ``2^k+1`` points, on all
meshes down to ``3`` points,
smoothing using ``s`` steps of the Krylov method on each mesh,
with a random target vector,
and continuing through the entire cycle ``c`` times. 

In the example, we are solving the Poisson equation on a grid
with ``2^{15}+1`` points using just ``15\cdot 7\cdot 3`` 
total steps of the conjugate gradient method. 

```@example gvl
function test_vcycle_1D_gvl(k,s,c)
  b=rand(2^k-1)
  N = 2^k-1 
  ls = reverse([sparse_square_laplacian(k′) for k′ in 1:k])
  is = reverse([sparse_restriction(k′) for k′ in 2:k])
  ps = reverse([sparse_prolongation(k′) for k′ in 2:k])
  u = zeros(N)
  norm(ls[1]*multigrid_vcycles(u,b,ls,is,ps,s,c)-b)/norm(b)
end
test_vcycle_1D_gvl(15,7,3)
```

## Reproducing the same solution with CombinatorialSpaces

Now we can show how to do the same thing with CombinatorialSpaces.
We'll use the same `multigrid_vcycles` function as before but
produce its inputs via types and data structures in CombinatorialSpaces.

In particular, `repeated_subdivisions` below produces a sequence of barycentric
subdivisions of a delta-set, which is exactly what we need to produce the
repeated halvings of the radius of the 1-D mesh in our example.

```@example cs
using Random # hide
Random.seed!(77777) # hide
using CombinatorialSpaces
using StaticArrays
using LinearAlgebra: norm
```

We first construct the *coarsest* stage in the 1-D mesh, with just two vertices
and one edge running from ``(0,0)`` to ``(1,0)``.

```@example cs
ss = EmbeddedDeltaSet1D{Bool,Point3D}()
add_vertices!(ss, 2, point=[(0,0,0),(1,0,0)])
add_edge!(ss, 1, 2, edge_orientation=true)

repeated_subdivisions(4,ss,subdivision_map)[1]
```

The setup function below constructs ``k`` subdivision maps and
their domains, then computes their Laplacians using CombinatorialSpaces'
general capabilities, as well as the prolongation matrices straight from the
subdivision maps and the interpolation matrices be renormalizing the transposed
prolongations.

We first construct everything with a sort on the vertices to show that 
we get the exact same results as in the first example.

```@example cs
function test_vcycle_1D_cs_setup_sorted(k)
  b=rand(2^k-1)
  N = 2^k-1 
  u = zeros(N)

  sds = reverse(repeated_subdivisions(k,ss,subdivision_map))
  sses = [sd.dom.delta_set for sd in sds]
  sorts = [sort(vertices(ss),by=x->ss[:point][x]) for ss in sses]
  ls = [laplacian(sses[i])[sorts[i],sorts[i]][2:end-1,2:end-1] for i in eachindex(sses)]
  ps = transpose.([as_matrix(sds[i])[sorts[i+1],sorts[i]][2:end-1,2:end-1] for i in 1:length(sds)-1])
  is = transpose.(ps)*1/2
  u,b,ls,is,ps
end
u,b,ls,is,ps = test_vcycle_1D_cs_setup_sorted(3)
ls[1]
```

```@example cs
ps[1]
```

Finally, we run a faster and simpler algorithm by avoiding all the sorting.
This version makes the truncation of each matrix to ignore the boundary vertices
more obvious (and truncates different rows and columns because of skipping the sort.) This is mathematically correct as long as the boundary conditions
are zero.

```@example cs
function test_vcycle_1D_cs_setup(k)
  b=rand(2^k-1)
  N = 2^k-1 
  u = zeros(N)

  sds = reverse(repeated_subdivisions(k,ss,subdivision_map))
  sses = [sd.dom.delta_set for sd in sds]
  ls = [laplacian(sses[i])[3:end,3:end] for i in eachindex(sses)]
  ps = transpose.([as_matrix(sds[i])[3:end,3:end] for i in 1:length(sds)-1])
  is = transpose.(ps)*1/2
  u,b,ls,is,ps
end
uu,bb,lls,iis,pps = test_vcycle_1D_cs_setup(15)
norm(ls[1]*multigrid_vcycles(u,b,ls,is,ps,7,3)-b)/norm(b)
```


# The 2-D Poisson equation

Next we consider the two-dimensional Poisson equation ``\Delta u = -F(x,y)`` on the unit square with Dirichlet boundary conditions; for concreteness we'll again focus on the case where
the boundary values are zero.

## A traditional approach

Divide the unit square ``[0,1]\times [0,1]`` into a square mesh with squares of side length
``h``. For each interior point ``(ih,jh)``, divided differences produce the equation
```math
4u(ih,jh)-u(ih,(j+1)h)-u(ih,(j-1)h)-u((i+1)h,jh)-u((i-1)h,jh) = h^2F(ih,jh).
```

If we write ``L(n)`` for the 1-D discretized Laplacian in ``n`` pieces on ``[0,1]``, thus with 
diameter ``h=1/n``, then it can be shown that, if we index the off-boundary grid points 
lexicographically by rows, the matrix encoding all the above equations is given by
```math
I_{n-1}\otimes L(n-1) + L(n-1)\otimes I_{n-1},
```
where ``I_{n-1}`` is the identity matrix of size ``n-1`` and ``\otimes`` is the Kronecker product.
In code, with the Laplacian for the interior of a ``5\times 5`` grid:

```@example gvl
sym_kron(A,B) = kron(A,B)+kron(B,A)
sparse_square_laplacian_2D(N,h) = sym_kron(I(N),sparse_square_laplacian(N,h))
sparse_square_laplacian_2D(k) = sparse_square_laplacian_2D(2^k-1,1/(2^k))
sparse_square_laplacian_2D(2)
```

To prolong a scalar field from a coarse grid (taking every other row and every other column)
to a fine one, the natural rule is to send a coarse grid value to itself, 
a value in an even row and odd column or vice versa to the average of its directly 
adjacent coarse grid values, and a value in an odd row and column to the average of its
four diagonally adjacent coarse grid valus. This produces the prolongation
matrix below:

```@example gvl
sparse_prolongation_2D(k) = kron(sparse_prolongation(k),sparse_prolongation(k))
sparse_prolongation_2D(3)[1:14,:]
```

We'll impose a Galerkin condition that the prolongation and restriction operators be adjoints of each other up to constants. This leads to the interesting consequence that the restriction
operator takens a weighted average of all nine nearby values, including those at the diagonally
nearest points, even though those points don't come up in computing second-order divided
differences.

```@example gvl
sparse_restriction_2D(k) = transpose(sparse_prolongation_2D(k))/4
sparse_restriction_2D(3)[1,:]
```

Now we can do the same multigrid v-cycles as before, but with the 2-D Laplacian and prolongation operators! Here we'll solve on a grid with about a million points in just a few seconds.

```@example gvl
function test_vcycle_2D_gvl(k,s,c)
  ls = reverse([sparse_square_laplacian_2D(k′) for k′ in 1:k])
  is = reverse([sparse_restriction_2D(k′) for k′ in 2:k])
  ps = reverse([sparse_prolongation_2D(k′) for k′ in 2:k])
  b=rand(size(ls[1],1))
  u = zeros(size(ls[1],1))
  norm(ls[1]*multigrid_vcycles(u,b,ls,is,ps,s,c)-b)/norm(b)
end

test_vcycle_2D_gvl(8,20,3)
```



Below we show how to reconstruct the grid Laplacian using 
CombinatorialSpaces.
```@example cs
using Random # hide
Random.seed!(77777) # hide
using Krylov
using CombinatorialSpaces
using GeometryBasics
using LinearAlgebra: norm
Point2D = Point2{Float64}

 

laplacian(ss) = ∇²(0,dualize(ss,Barycenter()))

#Copies of the primal square above in an N x N grid covering unit square in plane
function square_tiling(N)
  ss = EmbeddedDeltaSet2D{Bool,Point3D}()
  h = 1/(N-1)
  points = Point3D.([[i*h,1-j*h,0] for j in 0:N-1 for i in 0:N-1])
  add_vertices!(ss, N^2, point=points)
  for i in 1:N^2
    #vertices not in the left column or bottom row
    if (i-1)%N != 0 && (i-1) ÷ N < N-1
      glue_sorted_triangle!(ss, i, i+N-1,i+N)
    end
    #vertices not in the right column or bottom row
    if i %N != 0 && (i-1) ÷ N < N-1
      glue_sorted_triangle!(ss, i, i+1,i+N)
    end
  end
  orient!(ss)
  ss
end


inner(N) = vcat([2+k*N:N-1+k*N for k ∈ 1:N-2]...)
inlap(N) = laplacian(square_tiling(N))[inner(N),inner(N)]
inlap(5)
```

It was a bit annoying to have to manually subdivide the 
square grid; we can automate this with the `repeated_subdivisions` 
function, but the non-uniformity of neighborhoods in a barycentric
subdivision of a 2-D simplicial set means we should use a different
subdivider. We focus on the "triforce" subdivision, which splits
each triangle into four by connecting the midpoints of its edges.

```math
    v̇ == μ * Δ(v)-d₀(P) + φ
    Ṗ == ⋆₀⁻¹(dual_d₁(⋆₁(v)))
```

```@example stokes
using Krylov, LinearOperators, CombinatorialSpaces, LinearAlgebra
s = triangulated_grid(1,1,1/4,sqrt(3)/2*1/4,Point3D)
fs = reverse(repeated_subdivisions(4,s,triforce_subdivision_map))
sses = map(fs) do f dom(f).delta_set end
push!(sses,s)

function form_operator(μ,s)
  @info "Forming operator for $(nv(s)) vertices"
  sd = dualize(s,Circumcenter())
  L1 = ∇²(1,sd)
  d0 = dec_differential(0,sd)
  s1 = dec_hodge_star(1,sd)
  s0inv = inv_hodge_star(0,sd)
  d1 = dec_dual_derivative(1,sd)
  [μ*L1 -d0
  s0inv*d1*s1 0*I]
end

ops = map(s->form_operator(1,s),sses)

#=
len(s,e) = sqrt(sum((s[:point][s[:∂v0][e]]- s[:point][s[:∂v1][e]]) .^2))
diam(s) = minimum(len(s,e) for e in edges(s))
ε = diam(s0)
bvs(s) = findall(x -> abs(x[1]) < ε || abs(x[1]-1) < ε || x[2] == 0 || abs(x[2]-1)< ε*sqrt(3)/2, s[:point])
ivs(s) = filter(x -> !(x in bvs(s)), 1:nv(s))
bes(s) = begin bvs_s = bvs(s) ; 
  findall(x -> s[:∂v0][x] ∈ bvs_s ||  s[:∂v1][x] ∈ bvs_s , parts(s,:E)) end
ies(s) = begin b = bes(s);  [e for e in edges(s) if !(e in b)] end
sd = dualize(s,Circumcenter())
gensim(StokesDynamics)
f! = evalsim(StokesDynamics)(sd,nothing)
=#
nw(s) = ne(s)+nv(s)
fine_op = ops[1]

b = fine_op* ones(nw(sses[1]))
sol = gmres(fine_op,b)
norm(fine_op*sol[1]-b)/norm(b)
rs = transpose.(as_matrix.(fs))
ps = transpose.(rs) .* 1/4
S = ♯_mat(dualize(s,Circumcenter()),AltPPSharp())

#XXX: need to sharpen and flatten
u = zeros(nw(sses[1]))
multigrid_vcycles(u,b,ops,rs,ps,5,3)
```



Let's back up for a minute and make sure we can run the heat equation with our lovely triangular meshes.

```@example heat-on-triangles
using Krylov, CombinatorialSpaces, LinearAlgebra

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
norm(Ls[1]*u-b)/norm(b)
```