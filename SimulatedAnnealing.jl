using CairoMakie
using Catlab
using CombinatorialSpaces
using CombinatorialSpaces.SimplicialSets: boundary_inds
using GeometryBasics: Point3
using LinearAlgebra
using Random
using SparseArrays
using StaticArrays: SVector
Point3D = Point3{Float64}
Random.seed!(0)

lx, ly = 40.0, 20.0
function mirrored_mesh()
  s = EmbeddedDeltaSet2D{Bool, Point{3, Float64}}();
  lx, ly = 40.0, 20.0
  xs = range(0, lx; length = 3)
  ys = range(0, ly; length = 3)
  
  add_vertices!(s, 9)
  i = 1
  for y in ys
    for x in xs
      s[i, :point] = Point3([x, y, 0.0])
      i += 1
    end
  end
  glue_sorted_triangle!(s, 1, 2, 4)
  glue_sorted_triangle!(s, 2, 4, 5)
  glue_sorted_triangle!(s, 2, 5, 6)
  glue_sorted_triangle!(s, 2, 3, 6)
  glue_sorted_triangle!(s, 4, 5, 7)
  glue_sorted_triangle!(s, 5, 7, 8)
  glue_sorted_triangle!(s, 5, 8, 9)
  glue_sorted_triangle!(s, 5, 6, 9)
  orient!(s)
  s[:edge_orientation] = false
  s
end

function square_mesh()
  s = EmbeddedDeltaSet2D{Bool, Point{3, Float64}}();
  add_vertices!(s, 4,
    point=[Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0), Point3D(1,1,0)])
  glue_sorted_triangle!(s, 1, 2, 3)
  glue_sorted_triangle!(s, 2, 3, 4)
  orient!(s)
  s[:edge_orientation] = false
  s
end

function e_lens(s, ∂₂)
  rows = rowvals(∂₂)
  vals = nonzeros(∂₂)
  m, n = size(∂₂)
  map(1:n) do j
    [CombinatorialSpaces.volume(1,s,rows[i]) for i in nzrange(∂₂, j)]
  end
end

function e_eqs(els)
  map(x -> equilaterality(x...), els)
end

# 0 -> perfectly equilateral.
function equilaterality(e0, e1, e2)
  e0, e1, e2 = normalize([e0, e1, e2], 1)
  avg = (e0+e1+e2)/3
  (e0 - avg)^2 + (e1 - avg)^2 + (e2 - avg)^2
end

function equilaterality(s, ∂₂)
  els = e_lens(s, ∂₂)
  eeqs = e_eqs(els)
  sum(abs2, eeqs)
end

function optimize!(s, ϵ, epochs=1:100, hold_boundaries=true, anneal=true, jitter3D = false, spherical = false)
  ∂₂ = ∂(2,s)
  boundaries = boundary_inds(Val{0}, s)
  interior = setdiff(vertices(s), boundaries)
  map(epochs) do epoch
    # TODO: Could try vectorizing with a MVN.
    for v in (hold_boundaries ? interior : vertices(s))
      jitter = jitter3D ?
        Point3D(randn() * ϵ, randn() * ϵ, randn() * ϵ) :
        Point3D(randn() * ϵ, randn() * ϵ, 0)
      orig = s[v, :point]
      jittered = spherical ? normalize(orig + jitter) : orig + jitter
      orig_eq = equilaterality(s, ∂₂)
      s[v, :point] = jittered
      temp_eq = equilaterality(s, ∂₂)
      # To get out of local minima, select whether to accept with some
      # probability, decreasing the selection probability (i.e. "temperature")
      # as time decreases.
      jump_anyway = anneal ?
        (rand() < range(0.05, .001; length=length(epochs))[epoch]) :
        false
      s[v, :point] = if temp_eq < orig_eq || jump_anyway
        jittered
      else
        orig 
      end
    end
    epoch % 100 == 0 && println(equilaterality(s, ∂₂))
    equilaterality(s, ∂₂)
  end
end

function wf(s_orig, s)
  f = Figure()
  ax = CairoMakie.Axis(f[1,1])
  wireframe!(ax, s_orig)
  wireframe!(ax, s)
  f
end

function wf(s)
  f = Figure()
  ax = CairoMakie.Axis(f[1,1])
  wireframe!(ax, s)
  f
end

function eqs_dist(s_orig, s)
  f = Figure()
  ax = CairoMakie.Axis(f[1,1])
  hist!(ax, e_eqs(e_lens(s, ∂₂)))
  hist!(ax, e_eqs(e_lens(s_orig, ∂₂)))
  f
end

s = binary_subdivision(mirrored_mesh());
orient!(s);
s[:edge_orientation] = false;
s_orig = copy(s);
∂₂ = ∂(2,s)
equilaterality(s, ∂₂)
eqs = optimize!(s, 1e-3, 1:5000)
lines(eqs)
equilaterality(s, ∂₂)
wf(s_orig, s)
eqs_dist(s_orig, s)

s = binary_subdivision(binary_subdivision(mirrored_mesh()));
orient!(s);
s[:edge_orientation] = false;
s_orig = copy(s);
∂₂ = ∂(2,s)
equilaterality(s, ∂₂)
eqs = optimize!(s, 1e-3, 1:1000)
lines(eqs)
equilaterality(s, ∂₂)
wf(s_orig, s)
eqs_dist(s_orig, s)

s = binary_subdivision(mirrored_mesh());
orient!(s);
s[:edge_orientation] = false;
s_orig = copy(s);
∂₂ = ∂(2,s)
equilaterality(s, ∂₂)
eqs = optimize!(s, 1e-2, 1:1000, true, true, true)
lines(eqs)
equilaterality(s, ∂₂)
wf(s_orig, s)
eqs_dist(s_orig, s)

s = cubic_subdivision(square_mesh());
orient!(s);
s[:edge_orientation] = false;
s_orig = copy(s);
∂₂ = ∂(2,s)
equilaterality(s, ∂₂)
eqs = optimize!(s, 1e-3, 1:10000)
lines(eqs)
equilaterality(s, ∂₂)
wf(s_orig, s)

s = triangulated_grid(40,20,5,5,Point3D);
s_orig = copy(s);
∂₂ = ∂(2,s)
equilaterality(s, ∂₂)
eqs = optimize!(s, 1e-2, 1:5000, true, true, true)
lines([equilaterality(s_orig, ∂₂), eqs...])
equilaterality(s, ∂₂)
wf(s_orig, s)
eqs_dist(s_orig, s)

s, npi, spi = makeSphere(0, 90, 20, 0, 360, 20, 1);
orient!(s);
s[:edge_orientation] = false;
s_orig = copy(s);
∂₂ = ∂(2,s)
equilaterality(s, ∂₂)
eqs = optimize!(s, 1e-3, 1:200, true, false, true, true)
lines([equilaterality(s_orig, ∂₂), eqs...])
equilaterality(s, ∂₂)
wf(s_orig, s)
wf(s)
eqs_dist(s_orig, s)

s, npi, spi = makeSphere(70, 110, 10, 0, 360, 20, 1);
orient!(s);
s[:edge_orientation] = false;
s_orig = copy(s);
∂₂ = ∂(2,s)
equilaterality(s, ∂₂)
eqs = optimize!(s, 1e-3, 1:200, true, true, true, true)
wireframe(s)

s = loadmesh(Icosphere(3))
s[:edge_orientation] = false;
s_orig = copy(s);
∂₂ = ∂(2,s)
eqs = optimize!(s, 1e-3, 1:100, true, false, true, true)
lines(eqs)
wireframe(s)
eqs_dist(s_orig, s)
function viz(s)
  sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(msh)
  subdivide_duals!(sd, Circumcenter())
  s0 = dec_hodge_star(0, sd)
  f = Figure()
  ax = CairoMakie.LScene(f[1, 1], scenekw=(lights=[],))
  scatter!(ax, sd[sd[:tri_center], :dual_point], color=e_eqs(e_lens(s, ∂₂)))
  f
end
viz(s)

Diffusion = @decapode begin
  C::Form0
  k::Constant
  ∂ₜ(C) == k * Δ(C)
end
sim = evalsim(Diffusion)

function dome_comparison(sim)
  #s, npi, spi = makeSphere(0, 90, 20, 0, 360, 20, 1);
  #orient!(s);
  #s[:edge_orientation] = false;
  #s_orig = copy(s);
  #∂₂ = ∂(2,s)
  #eqs = optimize!(s, 1e-3, 1:200, true, true, true, true)

  s = loadmesh(Icosphere(3))
  s[:edge_orientation] = false;
  s_orig = copy(s);
  ∂₂ = ∂(2,s)
  eqs = optimize!(s, 1e-3, 1:100, true, false, true, true)

  function solve_sim(msh)
    sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(msh)
    #subdivide_duals!(sd, Circumcenter())
    subdivide_duals!(sd, Barycenter())

    f = sim(sd, nothing, GeometricHodge())
    
    u₀ = ComponentArray(C = getindex.(point(msh), 3))
    cs_ps = (k = 1e-1,)
    
    tₑ = 1e5
    prob = ODEProblem(f, u₀, (0, tₑ), cs_ps)
    soln = solve(prob, Tsit5(); progress=true, progress_steps=1);
    s0 = dec_hodge_star(0, sd)
    s0, soln
  end

  @info "Solving original mesh"
  s0_orig, soln_orig = solve_sim(s_orig)
  @info "Solving optimized mesh"
  s0, soln = solve_sim(s)

  function compare_integrals()
    f = Figure();
    ax = CairoMakie.Axis(f[1,1]);
    lines!(ax, map(range(0,tₑ;length=100)) do x
      abs(sum(s0 * soln(x).C) - sum(s0 * soln(0).C))
    end)
    lines!(ax, map(range(0,tₑ;length=100)) do x
      abs(sum(s0_orig * soln_orig(x).C) - sum(s0_orig * soln_orig(0).C))
    end)
    f
  end
  
  function viz(msh, msh_soln)
    f = Figure()
    ax = CairoMakie.LScene(f[1, 1], scenekw=(lights=[],))
    mesh!(ax, msh, color=msh_soln(tₑ).C)
    f
  end
  compare_integrals(), viz(s_orig, soln_orig), viz(s, soln), wf(s), wf(s_orig),
  eqs_dist(s_orig, s)
end

plots = dome_comparison(sim)
display.(plots)

int_diff = map(range(0,tₑ;length=100)) do x
  abs(sum(s0 * soln(x).C) - sum(s0 * soln(0).C))
end
int_diff_orig = map(range(0,tₑ;length=100)) do x
  abs(sum(s0_orig * soln_orig(x).C) - sum(s0_orig * soln_orig(0).C))
end

