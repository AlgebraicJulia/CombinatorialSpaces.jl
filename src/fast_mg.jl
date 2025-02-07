using Krylov, CombinatorialSpaces, LinearAlgebra, Test
using GeometryBasics: Point3, Point2
using Random
using BenchmarkTools
using SparseArrays
const Point3D = Point3{Float64}
const Point2D = Point2{Float64}

s = triangulated_grid(5,5,1,1,Point3D,false)
series = PrimalGeometricMapSeries(s, binary_subdivision_map, 4);

md = MultigridData(series, sd -> ∇²(0, sd), 3);
sd = finest_mesh(series);
L = first(md.operators)

Random.seed!(0)
b = L*rand(nv(sd))

"""
Access the leading grid, restriction, prolongation, and step radius at the desired index of the multigrid data list.
"""
function mg_level(md::MultigridData, idx::Int)
  if idx == length(md)
    return md.operators[idx],nothing,nothing,md.steps[idx]
  end
  return md.operators[idx],md.restrictions[idx],md.prolongations[idx],md.steps[idx]
end

struct MultiGridCache{F, S}
  zeros_list::Vector{Vector{F}}
  delta_list::Vector{Vector{F}}
  res_checkout::Vector{Bool}
  res_list::Vector{Vector{F}}
  solver_list::Vector{S}
end

# Only cache will change, Krylov doesn't change warm start value
function test_multigrid_v_cycle!(cache::MultiGridCache, md::MultigridData, alg!, idx, x_0, b)
  A = md.operators[idx]
  steps = md.steps[idx]
  solver = cache.solver_list[idx]

  pre_smooth = alg!(solver, A, b, x_0; itmax=steps)
  x_h = solution(pre_smooth)
  if idx == length(md)
    return x_h
  end

  R_map = md.restrictions[idx]
  P_map = md.prolongations[idx]

  # Next index is safe cause of "if"
  @assert !cache.res_checkout[idx+1]

  # TODO: Checking can be removed if there is no cache overwriting issue
  cache.res_checkout[idx+1] = true

  r_h = pre_smooth.r
  r_H = cache.res_list[idx+1]
  mul!(r_H, R_map, r_h)

  δ0_H = cache.zeros_list[idx + 1]
  δ_H = test_multigrid_v_cycle!(cache, md, alg!, idx + 1, δ0_H, r_H)

  cache.res_checkout[idx+1] = false

  δ_h = cache.delta_list[idx]
  mul!(δ_h, P_map, δ_H)
  x_h .= x_h .+ δ_h

  post_smooth = alg!(solver, A, b, x_h; itmax=steps)
  x_h = solution(post_smooth)

  return x_h
end

function run_test_multigrid_v_cycle!(md::MultigridData, alg!, cycles, x_0, b)
  zeros_list = map(op->zeros(size(op, 1)), md.operators)
  delta_list = map(op->zeros(size(op, 1)), md.operators)
  res_checkout = fill(false, length(md))
  res_list = map(op->zeros(size(op, 1)), md.operators)
  solver_list = map(op->CgSolver(size(op)..., Vector{Float64}), md.operators) # TODO: Make type user settable
  solver_type = typeof(first(solver_list))
  cache = MultiGridCache{Float64, solver_type}(zeros_list, delta_list, res_checkout, res_list, solver_list)

  x = x_0
  for i in 1:cycles
    x = test_multigrid_v_cycle!(cache, md, alg!, 1, x, b)
  end

  return x
end

res(L, x, b) = norm(L * x - b)

x0 = zeros(nv(sd));

test_x = multigrid_vcycles(x0, b, md, 25);
res(L, test_x, b)

in_mg_x = run_test_multigrid_v_cycle!(md, cg!, 25, x0, b);
res(L, in_mg_x, b)

@time run_test_multigrid_v_cycle!(md, cg!, 25, x0, b);
@time multigrid_vcycles(x0, b, md, 25);

# cg_solver = CgSolver(L, b)
