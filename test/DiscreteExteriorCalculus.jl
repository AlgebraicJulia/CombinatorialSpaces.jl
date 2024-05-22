module TestDiscreteExteriorCalculus
using Test

using LinearAlgebra: Diagonal, mul!, norm, dot
using SparseArrays, StaticArrays

using Catlab.CategoricalAlgebra.CSets
using ACSets
using ACSets.DenseACSets: attrtype_type
using CombinatorialSpaces
using CombinatorialSpaces.Meshes: tri_345, tri_345_false, grid_345, right_scalene_unit_hypot
using CombinatorialSpaces.DiscreteExteriorCalculus: eval_constant_primal_form, eval_constant_dual_form
using GeometryBasics: Point2, Point3

const Point2D = SVector{2,Float64}
const Point3D = SVector{3,Float64}

# 1D dual complex
#################

primal_s = DeltaSet1D()
add_vertices!(primal_s, 5)
add_edges!(primal_s, 1:4, repeat([5], 4))
s = DeltaDualComplex1D(primal_s)
@test nparts(s, :DualV) == nv(primal_s) + ne(primal_s)
@test nparts(s, :DualE) == 2 * ne(primal_s)

dual_v = elementary_duals(1,s,4)
@test dual_v == [edge_center(s, 4)]
@test elementary_duals(s, E(4)) == DualV(dual_v)

dual_es = elementary_duals(0,s,5)
@test length(dual_es) == 4
@test s[dual_es, :D_∂v0] == edge_center(s, 1:4)
@test elementary_duals(s, V(5)) == DualE(dual_es)

primal_s′ = subdivide(primal_s)
@test nv(primal_s′) == nv(primal_s) + ne(primal_s)
@test ne(primal_s′) == 2*ne(primal_s)

# 1D oriented dual complex
#-------------------------

primal_s = OrientedDeltaSet1D{Bool}()
add_vertices!(primal_s, 3)
add_edges!(primal_s, [1,2], [2,3], edge_orientation=[true,false])
s = OrientedDeltaDualComplex1D{Bool}(primal_s)
@test s[only(elementary_duals(0,s,1)), :D_edge_orientation] == true
@test s[only(elementary_duals(0,s,3)), :D_edge_orientation] == true

@test ∂(s, DualChain{1}([1,0,1])) isa DualChain{0}
@test d(s, DualForm{0}([1,1])) isa DualForm{1}
@test dual_boundary(1,s) == ∂(1,s)'
@test dual_derivative(0,s) == -d(0,s)'

primal_s′ = subdivide(primal_s)
@test nv(primal_s′) == nv(primal_s) + ne(primal_s)
@test ne(primal_s′) == 2*ne(primal_s)
@test orient!(primal_s′)

# 1D embedded dual complex
#-------------------------

# Path graph on 3 vertices with irregular lengths.
explicit_s = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(explicit_s, 3, point=[Point2D(1,0), Point2D(0,0), Point2D(0,2)])
add_edges!(explicit_s, [1,2], [2,3], edge_orientation=true)

# Path graph on 3 vertices without orientation set beforehand.
implicit_s = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(implicit_s, 3, point=[Point2D(1,0), Point2D(0,0), Point2D(0,2)])
add_edges!(implicit_s, [1,2], [2,3])

for primal_s in [explicit_s, implicit_s]
  s = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(primal_s)
  subdivide_duals!(s, Barycenter())
  @test dual_point(s, edge_center(s, [1,2])) ≈ [Point2D(0.5,0), Point2D(0,1)]
  @test volume(s, E(1:2)) ≈ [1.0, 2.0]
  @test volume(s, elementary_duals(s, V(2))) ≈ [0.5, 1.0]
  @test ⋆(0,s) ≈ Diagonal([0.5, 1.5, 1.0])
  @test ⋆(1,s) ≈ Diagonal([1, 0.5])
  @test ⋆(s, VForm([0,2,0]))::DualForm{1} ≈ DualForm{1}([0,3,0])

  @test ∧(0,0,s, [1,2,3], [3,4,7]) ≈ [3,8,21]
  @test ∧(s, VForm([1,2,3]), VForm([3,4,7]))::VForm ≈ VForm([3,8,21])
  @test ∧(s, VForm([1,1,1]), EForm([2.5, 5.0]))::EForm ≈ EForm([2.5, 5.0])
  @test ∧(s, VForm([1,1,0]), EForm([2.5, 5.0])) ≈ EForm([2.5, 2.5])
  vform, eform = VForm([1.5, 2, 2.5]), EForm([13, 7])
  @test ∧(s, vform, eform) ≈ ∧(s, eform, vform)
end

# Path graph on 5 vertices with regular lengths.
#
# Equals the graph Laplacian of the underlying graph, except at the boundary.
# Note that the DEC Laplace-Beltrami operator is *not* symmetric.
primal_s = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(primal_s, 5, point=[Point2D(i,0) for i in -2:2])
add_edges!(primal_s, 1:4, 2:5, edge_orientation=true)
s = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(primal_s)
subdivide_duals!(s, Barycenter())
@test ∇²(s, VForm([0,0,1,0,0])) ≈ VForm([0,-1,2,-1,0])
@test ∇²(0,s) ≈ [ 2 -2  0  0  0;
                 -1  2 -1  0  0;
                  0 -1  2 -1  0;
                  0  0 -1  2 -1;
                  0  0  0 -2  2]
f = VForm([0,1,2,1,0])
@test Δ(s,f) ≈ -∇²(s,f)
@test Δ(s, EForm([0,1,1,0])) isa EForm
@test Δ(s, EForm([0,1,1,0]), hodge=DiagonalHodge()) isa EForm

@test isapprox(Δ(0, s), [-2  2  0  0  0;
                          1 -2  1  0  0;
                          0  1 -2  1  0;
                          0  0  1 -2  1;
                          0  0  0  2 -2], atol=1e-3)

@test isapprox(Δ(1, s), [-3.0  1.0  0.0  0.0;
                          1.0 -2.0  1.0  0.0;
                          0.0  1.0 -2.0  1.0;
                          0.0  0.0  1.0 -3.0], atol=1e-3)

# 2D dual complex
#################

# Triangulated square.
primal_s = DeltaSet2D()
add_vertices!(primal_s, 4)
glue_triangle!(primal_s, 1, 2, 3)
glue_triangle!(primal_s, 1, 3, 4)
s = DeltaDualComplex2D(primal_s)
@test nparts(s, :DualV) == nv(primal_s) + ne(primal_s) + ntriangles(primal_s)
@test nparts(s, :DualE) == 2*ne(primal_s) + 6*ntriangles(primal_s)
@test nparts(s, :DualTri) == 6*ntriangles(primal_s)
@test primal_vertex(s, subsimplices(s, Tri(1)))::V == V([1,1,2,2,3,3])

dual_vs = elementary_duals(2,s,2)
@test dual_vs == [triangle_center(s,2)]
@test elementary_duals(s, Tri(2)) == DualV(dual_vs)
@test s[elementary_duals(1,s,2), :D_∂v1] == [edge_center(s,2)]
@test s[elementary_duals(1,s,3), :D_∂v1] == repeat([edge_center(s,3)], 2)
@test [length(elementary_duals(s, V(i))) for i in 1:4] == [4,2,4,2]
@test dual_triangle_vertices(s, 1) == [1,7,10]

# 2D oriented dual complex
#-------------------------

# Triangulated square with consistent orientation.
explicit_s = OrientedDeltaSet2D{Bool}()
add_vertices!(explicit_s, 4)
glue_triangle!(explicit_s, 1, 2, 3, tri_orientation=true)
glue_triangle!(explicit_s, 1, 3, 4, tri_orientation=true)
explicit_s[:edge_orientation] = true

# Triangulated square without explicit orientation set beforehand.
implicit_s = OrientedDeltaSet2D{Bool}()
add_vertices!(implicit_s, 4)
glue_triangle!(implicit_s, 1, 2, 3)
glue_triangle!(implicit_s, 1, 3, 4)

for primal_s in [explicit_s, implicit_s]
  s = OrientedDeltaDualComplex2D{Bool}(primal_s)
  @test sum(s[:D_tri_orientation]) == nparts(s, :DualTri) ÷ 2
  @test [sum(s[elementary_duals(0,s,i), :D_tri_orientation])
        for i in 1:4] == [2,1,2,1]
  @test sum(s[elementary_duals(1,s,3), :D_edge_orientation]) == 1

  for k in 0:1
    @test dual_boundary(2-k,s) == (-1)^k * ∂(k+1,s)'
  end
  for k in 1:2
    # Desbrun, Kanso, Tong 2008, Equation 4.2.
    @test dual_derivative(2-k,s) == (-1)^k * d(k-1,s)'
  end
end

# 2D embedded dual complex
#-------------------------

unit_vector(θ) = cos(θ), sin(θ), 0

function get_regular_polygon(n::Int)
  n < 3 && error("Cannot construct a polygon with fewer than 3 points.")
  primal_s = EmbeddedDeltaSet2D{Bool,Point3D}()
  exterior_points = map(Point3D∘unit_vector, range(0, 2pi-(pi/(n/2)), length=n))
  add_vertices!(primal_s, n+1, point=[exterior_points..., Point3D(0, 0, 0)])
  foreach(1:n-1) do x
    glue_sorted_triangle!(primal_s, n+1, x, x+1)
  end
  glue_sorted_triangle!(primal_s, n+1, n, 1)
  primal_s
end

# Triangulated hexagon in ℝ³,
# from Hirani §5.5 Figure 5.5, showing ♭ᵈᵖᵖ(X) to be 0 for a non-trivial X.
primal_s = get_regular_polygon(6)
# Rotate counter-clockwise by pi/6 to match the Hirani figure.
θ = -pi/6
primal_s[:point] = [[[cos(θ), -sin(θ), 0];; [sin(θ), cos(θ), 0];; [0,0,1]] * p for p in primal_s[:point]]
s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(primal_s)
subdivide_duals!(s, Circumcenter())
X = map([8,4,0,8,4,0]) do i
  SVector(unit_vector(i*(2pi/12)))
end
@test all(♭(s, DualVectorField(X))) do i
  isapprox(i, 0.0; atol=1e-15)
end
♭_m = ♭_mat(s)
@test all(reinterpret(Float64, ♭_m * X)) do i
  isapprox(i, 0.0; atol=1e-15)
end
# Half of Proposition 5.5.3: ♭ᵈᵖᵖ is not injective:
@test all(map(♭_m * X, ♭_m * 2X) do x♭, x2♭
  isapprox(x♭, x2♭; atol=1e-15)
end)

#using Random
#Random.seed!(1)
#Y = map(1:6) do i
#  10 .* SVector(randn(), randn(), randn())
#end
Y = SVector{3, Float64}[[-0.7058313895389791, 5.314767537831963, -8.06852326006714], [24.56991333983293, 11.648740735275195, 2.6756441862888507], [17.499336925282453, -8.260207919192975, -10.427524178910968], [-3.291338458564041, -4.822519154091156, 11.822427034001585], [4.806914145449797, -0.025581805543075972, 14.346125517139171], [5.320444346188966, 2.4384867352385236, 0.20245933893191853]]
Y♭ = zeros(SVector{1, Float64}, ne(s))
mul!(Y♭, ♭_m, Y)
Y♭_floats = reinterpret(Float64, Y♭)
@test all(map(♭(s, DualVectorField(Y)), Y♭_floats) do orig, new
  isapprox(orig, new; atol=20*eps(Float64))
end)

# Single triangle: numerical example from Gillette's notes on DEC, §2.13.
#
# Compared with Gillette, edges #2 and #3 are swapped in the ordering, which
# changes the discrete exterior derivative and other operators. The numerical
# values remain the same, as we verify.
primal_s = EmbeddedDeltaSet2D{Bool,Point2D}()
add_vertices!(primal_s, 3, point=[Point2D(0,0), Point2D(1,0), Point2D(0,1)])
glue_triangle!(primal_s, 1, 2, 3, tri_orientation=true)
primal_s[:edge_orientation] = true
s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(primal_s)

subdivide_duals!(s, Barycenter())
@test dual_point(s, triangle_center(s, 1)) ≈ Point2D(1/3, 1/3)
@test volume(s, Tri(1)) ≈ 1/2
@test volume(s, elementary_duals(s, V(1))) ≈ [1/12, 1/12]
@test [sum(volume(s, elementary_duals(s, V(i)))) for i in 1:3] ≈ [1/6, 1/6, 1/6]

# These values are consistent with the Gillette paper, as described above
@test ⋆(0,s) ≈ Diagonal([1/6, 1/6, 1/6])
@test ⋆(1,s; hodge=DiagonalHodge()) ≈ Diagonal([√5/6, 1/6, √5/6])
@test isapprox(δ(1,s; hodge=DiagonalHodge()), [ 2.236  0  2.236;
                                              -2.236  1  0;
                                               0     -1 -2.236], atol=1e-3)

# This test is consistent with Ayoub et al 2020 page 13 (up to permutation of
# vertices)
@test ⋆(1,s) ≈ [1/3 0.0 1/6;
                0.0 1/6 0.0;
                1/6 0.0 1/3]

# Test consistency regardless of base triangle orientation (relevant for
# geometric hodge star)
flipped_ps = deepcopy(primal_s)
orient_component!(flipped_ps, 1, false)
flipped_s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(flipped_ps)
subdivide_duals!(flipped_s, Barycenter())
@test ⋆(1,s) ≈ ⋆(1,flipped_s)

# NOTICE:
# Tests beneath this comment are not backed up by any external source, and are
# included to determine consistency as the operators are modified.
#
# If a test beneath this comment fails due to a new implementation, it is
# possible that the values for the test itself need to be modified.
@test inv_hodge_star(2, s)[1,1] ≈ 0.5
@test inv_hodge_star(2, s, [2.0])[1,1] ≈ 1.0
@test inv_hodge_star(1, s, hodge=DiagonalHodge()) ≈ Diagonal([-6/√5, -6, -6/√5])
@test inv_hodge_star(1, s, [0.5, 2.0, 0.5], hodge=DiagonalHodge()) ≈ [-3/√5, -12.0, -3/√5]
@test ⋆(s, VForm([1,2,3]))::DualForm{2} ≈ DualForm{2}([1/6, 1/3, 1/2])
@test isapprox(δ(1,s), [ 3.0  0  3.0;
                        -2.0  1 -1.0;
                         -1  -1 -2.0], atol=1e-3)
@test δ(s, EForm([0.5,1.5,0.5])) isa VForm
@test Δ(s, EForm([1.,2.,1.])) isa EForm
@test Δ(s, EForm([1.,2.,1.]); hodge=DiagonalHodge()) isa EForm
@test Δ(s, TriForm([1.])) isa TriForm
@test Δ(s, TriForm([1.]); hodge=DiagonalHodge()) isa TriForm
@test isapprox(Δ(0, s), [-6  3  3;
                          3 -3  0;
                          3  0 -3], atol=1e-3)

@test isapprox(Δ(1, s), [-17 -11  8;
                         -11 -14  11;
                           8  11 -17], atol=1e-3)

@test isapprox(Δ(1, s; hodge=DiagonalHodge()),
                        [-9.838  -4.366  3.130;
                         -9.763  -14.0   9.763;
                          3.130   4.366 -9.838], atol=1e-2)

@test isapprox(Δ(2, s), reshape([-36.0], (1,1)), atol=1e-3)
@test isapprox(Δ(2, s; hodge=DiagonalHodge()), reshape([-22.733], (1,1)), atol=1e-3)

subdivide_duals!(s, Circumcenter())
@test dual_point(s, triangle_center(s, 1)) ≈ Point2D(1/2, 1/2)
@test ⋆(0,s) ≈ Diagonal([1/4, 1/8, 1/8])
@test ⋆(1,s) ≈ Diagonal([0.5, 0.0, 0.5])
@test δ(1,s) ≈ [ 2  0  2;
                -4  0  0;
                 0  0 -4]

subdivide_duals!(s, Incenter())
@test dual_point(s, triangle_center(s, 1)) ≈ Point2D(1/(2+√2), 1/(2+√2))
@test isapprox(⋆(0,s), Diagonal([0.146, 0.177, 0.177]), atol=1e-3)
@test isapprox(⋆(1,s), [0.293 0.000 0.207;
                        0.000 0.207 0.000;
                        0.207 0.000 0.293], atol=1e-3)

@test isapprox(δ(1,s; hodge=DiagonalHodge()), [ 2.449  0      2.449;
                                              -2.029  1.172  0;
                                               0     -1.172 -2.029], atol=1e-3)

@test isapprox(δ(1,s), [ 3.414  0.000  3.414;
                        -1.657  1.172 -1.172;
                        -1.172 -1.172 -1.657], atol=1e-3)

# This plots a primal vector field over a simplicial complex.
# This is helpful for debugging operators like sharp and flat.
#using CairoMakie
#""" Plot the primal vector field X♯ over the simplicial complex primal_s.
#"""
#function plot_pvf(primal_s, X♯; ls=1f0, title="Primal Vector Field")
#  # Makie will throw an error if the colorrange end points are not unique:
#  extX = extrema(norm.(X♯))
#  range = extX[1] ≈ extX[2] ? (0,extX[2]) : extX
#  f = Figure()
#  ax = Axis(f[1, 1], title=title)
#  wireframe!(ax, primal_s, color=:gray95)
#  scatter!(ax, getindex.(primal_s[:point],1), getindex.(primal_s[:point],2), color = norm.(X♯), colorrange=range)
#  arrows!(ax, getindex.(primal_s[:point],1), getindex.(primal_s[:point],2), getindex.(X♯,1), getindex.(X♯,2), lengthscale=ls)
#  Colorbar(f[1,2], limits=range)
#  hidedecorations!(ax)
#  f
#end
#plot_pvf(primal_s, ♯(s, E))
#plot_pvf(primal_s, ♯(s, E, AltPPSharp()))

# Evaluate a constant 1-form α assuming Euclidean space. (Inner product is dot)
eval_constant_form(s, α::SVector) = map(edges(s)) do e
  dot(α, point(s, tgt(s,e)) - point(s, src(s,e))) * sign(1,s,e)
end |> EForm

function test_♯(s, covector::SVector; atol=1e-8)
  X = eval_constant_form(s, covector)
  # Test that the Hirani field is approximately parallel to the given field.
  X♯ = ♯(s, X)
  @test all(map(X♯) do x
    isapprox(dot(x, covector), norm(x) * norm(covector), atol=atol)
  end)
  # Test that the Alternate field approximately equals the given field.
  X♯ = ♯(s, X, AltPPSharp())
  @test all(isapprox.(X♯, [covector]))
  # Test that the matrix and non-matrix versions yield the same result.
  @test all(isapprox.(♯_mat(s, PPSharp()) * X, ♯(s, X, PPSharp())))
  @test all(isapprox.(♯_mat(s, AltPPSharp()) * X, ♯(s, X, AltPPSharp())))
end
#using Random
#Random.seed!(1)
#Point3D(randn(), randn(), 0)
vfs = [Point3D(1,0,0), Point3D(1,1,0), Point3D(-3,2,0), Point3D(0,0,0),
  Point3D(-0.07058313895389791, 0.5314767537831963, 0.0)]

# 3,4,5 triangle.
primal_s, s = tri_345()
foreach(vf -> test_♯(s, vf), vfs)
♯_m = ♯_mat(s, DesbrunSharp())
X = eval_constant_form(s, Point3D(1,0,0))
X♯ = ♯_m * X
@test all(X♯ .≈ [Point3D(.8,.4,0), Point3D(0,-1,0), Point3D(-.8,.6,0)])

# Grid of 3,4,5 triangles.
primal_s, s = grid_345()
foreach(vf -> test_♯(s, vf), vfs)
# TODO: Compute results for Desbrun's ♯ by hand.

# Triangulated regular dodecagon.
primal_s = get_regular_polygon(12)
primal_s[:point] = [Point3D(1/4,1/5,0) + p for p in primal_s[:point]]
s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(primal_s)
subdivide_duals!(s, Circumcenter())
foreach(vf -> test_♯(s, vf), vfs)
# TODO: Compute results for Desbrun's ♯ by hand.

# Triangulated square with consistent orientation.
primal_s = EmbeddedDeltaSet2D{Bool,Point2D}()
add_vertices!(primal_s, 4, point=[Point2D(-1,+1), Point2D(+1,+1),
                                  Point2D(+1,-1), Point2D(-1,-1)])
glue_triangle!(primal_s, 1, 2, 3, tri_orientation=true)
glue_triangle!(primal_s, 1, 3, 4, tri_orientation=true)
primal_s[:edge_orientation] = true
s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(primal_s)
subdivide_duals!(s, Barycenter())
♭_m = ♭_mat(s)

x̂, ŷ, ẑ = @SVector([1,0]), @SVector([0,1]), @SVector([0,0])
@test ♭(s, DualVectorField([x̂, -x̂])) ≈ EForm([2,0,0,2,0])
@test ♭(s, DualVectorField([ŷ, -ŷ])) ≈ EForm([0,-2,0,0,2])
@test ♭(s, DualVectorField([(x̂-ŷ)/√2, (x̂-ŷ)/√2]))[3] ≈ 2*√2
@test ♭(s, DualVectorField([(x̂-ŷ)/√2, ẑ]))[3] ≈ √2
@test reinterpret(Float64, ♭_m * [x̂, -x̂]) ≈ EForm([2,0,0,2,0])
@test reinterpret(Float64, ♭_m * [ŷ, -ŷ]) ≈ EForm([0,-2,0,0,2])
@test reinterpret(Float64, ♭_m * [(x̂-ŷ)/√2, (x̂-ŷ)/√2])[3] ≈ 2*√2
@test reinterpret(Float64, ♭_m * [(x̂-ŷ)/√2, ẑ])[3] ≈ √2
X = ♯(s, EForm([2,0,0,2,0]))::PrimalVectorField
@test X[2][1] > 0 && X[4][1] < 0
X = ♯(s, EForm([0,-2,0,0,2]))
@test X[2][2] > 0 && X[4][2] < 0

@test all(∧(s, VForm([2,2,2,2]), TriForm([2.5, 5]))::TriForm .≈ TriForm([5.0, 10.0]))
vform, triform = VForm([1.5, 2, 2.5, 3]), TriForm([5, 7.5])
@test ∧(s, vform, triform) ≈ ∧(s, triform, vform)
eform1, eform2 = EForm([1.5, 2, 2.5, 3, 3.5]), EForm([3, 7, 10, 11, 15])
@test ∧(s, eform1, eform1)::TriForm ≈ TriForm([0, 0])
@test ∧(s, eform1, eform2) ≈ -∧(s, eform2, eform1)

# Lie derivative of flattened vector-field on dual 0-form
X♭, α = EForm([1.5, 2, 2.5, 3, 3.5]), DualForm{0}([3, 7])
@test ℒ(s, X♭, α; hodge=GeometricHodge()) isa DualForm{0}
@test length(lie_derivative_flat(0,s, X♭.data, α.data)) == 2

# Lie derivative of flattened vector-field on dual 1-form
X♭, α = EForm([1.5, 2, 2.5, 3, 3.5]), DualForm{1}([3, 7, 10, 11, 15])
@test interior_product(s, X♭, α) isa DualForm{0}
@test length(interior_product_flat(1,s, X♭.data, α.data)) == 2
@test ℒ(s, X♭, α; hodge=GeometricHodge()) isa DualForm{1}
@test length(lie_derivative_flat(1,s, X♭.data, α.data)) == 5

# Lie derivative of flattened vector-field on dual 2-form
X♭, α = EForm([1.5, 2, 2.5, 3, 3.5]), DualForm{2}([3, 7, 10, 11])
@test interior_product(s, X♭, α) isa DualForm{1}
@test length(interior_product_flat(2,s, X♭.data, α.data)) == 5
@test ℒ(s, X♭, α; hodge=GeometricHodge()) isa DualForm{2}
@test length(lie_derivative_flat(2,s, X♭.data, α.data)) == 4

# Equilateral triangle (to compare the diagonal w/ geometric hodge results)
primal_s = EmbeddedDeltaSet2D{Bool,Point2D}()
add_vertices!(primal_s, 3, point=[Point2D(0,0), Point2D(1,0), Point2D(0.5,sqrt(0.75))])
glue_triangle!(primal_s, 1, 2, 3, tri_orientation=true)
primal_s[:edge_orientation] = true
s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(primal_s)
subdivide_duals!(s, Barycenter())

@test isapprox(Δ(1, s), [-12 -6    6;
                          -6 -12   6;
                           6   6 -12], atol=1e-3)

@test isapprox(Δ(1, s; hodge=DiagonalHodge()),
                        [-12 -6    6;
                          -6 -12   6;
                           6   6 -12], atol=1e-2)

@test isapprox(Δ(2, s), reshape([-24.0], (1,1)), atol=1e-3)
@test isapprox(Δ(2, s; hodge=DiagonalHodge()), reshape([-24.0], (1,1)), atol=1e-3)

# 3,4,5 triangle of unit area.
# Unit area makes computing integrals by hand simple.
primal_s = EmbeddedDeltaSet2D{Bool,Point2D}()
add_vertices!(primal_s, 3, point=[Point2D(0,0), Point2D(6/8,0), Point2D(6/8,8/3)])
glue_triangle!(primal_s, 1, 2, 3, tri_orientation=true)
primal_s[:edge_orientation] = true
s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(primal_s)
subdivide_duals!(s, Barycenter())
#@assert only(s[:area]) == 1.0

# The wedge product of 0-form and a 2-form "is multiplication."
# α = 3
# β = 5 dx ∧ dy
# α ∧ β = 15 dx ∧ dy
# We also write just αβ, when α is a 0-form.
@test only(∧(s, VForm([3,3,3]), TriForm([only(s[:area])*5]))) ≈ 15

# Grid of 3,4,5 triangles.
primal_s, s = grid_345()

# ∀ βᵏ ∧(α,βᵏ) = id(βᵏ), where α = 1.
α = VForm(ones(nv(s)))
for k = 0:2
  βᵏ = SimplexForm{k}(collect(1:nsimplices(k,s)))
  @test all(∧(s, α, βᵏ) .≈ βᵏ)
end

# 1dx ∧ 1dy = 1 dx∧dy
onedx = eval_constant_form(s, @SVector [1.0,0.0,0.0])
onedy = eval_constant_form(s, @SVector [0.0,1.0,0.0])
@test ∧(s, onedx, onedy) == map(s[:tri_orientation], s[:area]) do o,a
  # Note by the order of -1 and 1 here that 
  a * (o ? -1 : 1)
end

# 1dx∧dy = -1dy∧dx
@test ∧(s, onedx, onedy) == -∧(s, onedy, onedx)

# 3dx ∧ 2dy = 2 dx ∧ 3dy
@test ∧(s, EForm(3*onedx), EForm(2*onedy)) == ∧(s, EForm(2*onedx), EForm(3*onedy))

# A triangulated quadrilateral where edges are all of distinct length.
primal_s = EmbeddedDeltaSet2D{Bool,Point2D}()
add_vertices!(primal_s, 4, point=[Point2D(0,0), Point2D(1,0), Point2D(0,2), Point2D(-2,5)])
glue_triangle!(primal_s, 1, 2, 3)
glue_triangle!(primal_s, 1, 3, 4)
s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(primal_s)
subdivide_duals!(s, Barycenter())
X = [SVector(2,3), SVector(5,7)]
♭_m = ♭_mat(s)
X♭ = zeros(ne(s))
mul!(X♭, ♭_m,  DualVectorField(X))
@test all(map(♭(s, DualVectorField(X)), X♭) do orig, new
  isapprox(orig, new; atol=20*eps(Float64))
end)
@test all(map(♭(s, DualVectorField(X)), ♭_m * DualVectorField(X)) do orig, new
  isapprox(orig, new; atol=20*eps(Float64))
end)

# Test whether the implementation of ♭ assumes that the DualVectorField vector
# is indexed in the same order as Tri i.e. 1:ntriangles(s).
primal_s = EmbeddedDeltaSet2D{Bool,Point2D}()
add_vertices!(primal_s, 4, point=[Point2D(0,0), Point2D(1,0), Point2D(0,2), Point2D(-2,5)])
glue_triangle!(primal_s, 1, 2, 3)
glue_triangle!(primal_s, 1, 3, 4)
s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(primal_s)
subdivide_duals!(s, Barycenter())

primal_s′ = EmbeddedDeltaSet2D{Bool,Point2D}()
add_vertices!(primal_s′, 4, point=[Point2D(0,0), Point2D(1,0), Point2D(0,2), Point2D(-2,5)])
glue_triangle!(primal_s′, 1, 2, 3)
glue_triangle!(primal_s′, 1, 3, 4)
s′ = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(primal_s′)
s′[1, :tri_center] = 11
s′[2, :tri_center] = 10
s′[[11,13,15,17,19,21], :D_∂v0] = 11
s′[[12,14,16,18,20,22], :D_∂v0] = 10
subdivide_duals!(s′, Barycenter())
#@assert is_isomorphic(s,s′)

X = [SVector(2,3), SVector(5,7)]

@test ♭(s, DualVectorField(X)) == ♭(s′, DualVectorField(X))
@test ♭_mat(s) * DualVectorField(X) == ♭_mat(s′) * DualVectorField(X)

tg′ = triangulated_grid(100,100,10,10,Point2D);
tg = EmbeddedDeltaDualComplex2D{Bool,Float64,Point2D}(tg′);
subdivide_duals!(tg, Barycenter());

rect′ = loadmesh(Rectangle_30x10());
rect = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(rect′);
subdivide_duals!(rect, Barycenter());

flat_meshes = [tri_345(), tri_345_false(), right_scalene_unit_hypot(), grid_345(), (tg′, tg), (rect′, rect)];

# Test that the technique for evaluating 1-forms is consistent across primal
# and dual forms.
-# ♭ᵈ_discrete(f_continuous) = ⋆₁_discrete∘♭ᵖ_discrete(⋆_continuous f_continuous)
for (primal_s,s) in flat_meshes
  α = SVector(1/√2,1/√2,0)
  left_hand_α = SVector(1/√2,-1/√2,0)
  f′ = eval_constant_primal_form(s, left_hand_α)
  f̃ = eval_constant_dual_form(s, α)
  @test all(map((x,y) -> isapprox( x,y,atol=1e-15), hodge_star(1,s) * f′, f̃))
end

function all_are_approx(f::Vector{SVector{3,Float64}}, g::SVector{3,Float64}; atol)
  all(map(f) do f̂
    all(isapprox.(f̂, g, atol=atol))
  end)
end
function all_are_approx(f::Vector{SVector{2,Float64}}, g::SVector{3,Float64}; atol)
  all(map(f) do f̂
        all(isapprox.(f̂, SVector{2,Float64}(g[1],g[2]), atol=atol))
  end)
end

for (primal_s,s) in flat_meshes
  # Test that the least-squares ♯ from dual vector fields to dual 1-forms
  # preserves constant fields.
  ♯_m = ♯_mat(s, LLSDDSharp())

  # This tests that numerically raising indices is equivalent to analytically
  # raising indices.
  # X_continuous = ♯ᵖ_discrete∘♭ᵖ_discrete(X_continuous)
  X♯ = SVector{3,Float64}(1/√2,1/√2,0)
  X♭ = eval_constant_dual_form(s, X♯)
  @test all_are_approx(♯_m * X♭,  X♯; atol=1e-13) ||
        all_are_approx(♯_m * X♭, -X♯; atol=1e-13)

  # This tests that numerically raising-then-lowering indices is equivalent to
  # evaluating a 1-form directly.
  # Demonstrate how to cache the chained musical isomorphisms.
  ♭_m = ♭_mat(s)
  ♭_♯_m = ♭_m * ♯_m
  # Note that we check explicitly both cases of signedness, because orient!
  # picks in/outside without regard to the embedding.
  # This is the "right/left-hand-rule trick."
  @test all(isapprox.(only.(♭_♯_m * X♭),  eval_constant_primal_form(s, X♯), atol=1e-13)) ||
        all(isapprox.(only.(♭_♯_m * X♭), -eval_constant_primal_form(s, X♯), atol=1e-12))

  # This test shows how the musical isomorphism chaining lets you further
  # define a primal-dual wedge that preserves properties from the continuous
  # exterior calculus.
  Λpd = dec_wedge_product_pd(Tuple{1,1}, s)
  Λdp = dec_wedge_product_dp(Tuple{1,1}, s)

  f_def = SVector{3,Float64}(2,7,0)
  g_def = SVector{3,Float64}(8,1,0)
  f′ = eval_constant_primal_form(s, f_def)
  g̃  = eval_constant_dual_form(s, g_def)
  f̃  = eval_constant_dual_form(s, f_def)
  g′ = eval_constant_primal_form(s, g_def)

  # Antisymmetry:
  @test all(Λpd(f′, g̃) .≈ -1 * Λpd(g′, f̃))
  @test all(Λdp(f̃, g′) .≈ -1 * Λdp(g̃, f′))

  # Antisymmetry (across Λpd and Λdp!):
  @test all(Λpd(f′, g̃) .== -1 * Λdp(g̃, f′))
  @test all(Λdp(f̃, g′) .== -1 * Λpd(g′, f̃))

  # f∧f = 0 (implied by antisymmetry):
  @test all(isapprox.(Λpd(f′, f̃), 0, atol=1e-11))
  @test all(isapprox.(Λpd(g′, g̃), 0, atol=1e-11))

  # Test and demonstrate the convenience functions:
  @test all(∧(s, SimplexForm{1}(f′), DualForm{1}(g̃)) .≈ -1 * ∧(s, SimplexForm{1}(g′), DualForm{1}(f̃)))
  @test all(∧(s, DualForm{1}(f̃), SimplexForm{1}(g′)) .≈ -1 * ∧(s, DualForm{1}(g̃), SimplexForm{1}(f′)))

  # TODO: Test the Leibniz rule.
end

# Discretely test the continuous property:
# u :=  fdx + gdy
# ⋆u = -gdx + fdy
# u∧⋆u = (fdx + gdy)∧(-gdx + fdy)
#      = ffdxdy - ggdydx
#      = (ff+gg)dxdy
# ⋆(u∧⋆u) = ff+gg
for (primal_s,s) in flat_meshes
  f = 2
  g = 7
  ff_gg = f*f + g*g

  u_def = SVector{3,Float64}(f,g,0)

  u = eval_constant_primal_form(s, u_def)
  u_star = hodge_star(1,s) * u

  @test all(isapprox.(
    dec_hodge_star(2,s) * ∧(s, SimplexForm{1}(u), DualForm{1}(u_star)),
    ff_gg,
    atol=1e-12))
end

end

