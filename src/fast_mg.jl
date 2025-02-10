using Krylov, CombinatorialSpaces, LinearAlgebra, Test
using GeometryBasics: Point3, Point2
using Random
using BenchmarkTools
using SparseArrays
using KrylovPreconditioners
const Point3D = Point3{Float64}
const Point2D = Point2{Float64}

s = triangulated_grid(5,5,1,1,Point3D,false)
series = PrimalGeometricMapSeries(s, binary_subdivision_map, 4);

md = MultigridData(series, sd -> ∇²(0, sd), 3);
sd = finest_mesh(series);
L = first(md.operators)

Random.seed!(0)
b = L*rand(nv(sd))

struct MultigridCache{F, S}
  zeros::Vector{Vector{F}}
  delta::Vector{Vector{F}}
  res_checkout::Vector{Bool}
  res::Vector{Vector{F}}
  solver::Vector{S}
end

# Only cache will change, Krylov doesn't change warm start value
function test_multigrid_v_cycle!(cache::MultigridCache, md::MultigridData, alg!, idx, x0, b)
  A = md.operators[idx]
  steps = md.steps[idx]
  solver = cache.solver[idx]

  pre_smooth = alg!(solver, A, b, x0; itmax=steps)
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
  r_H = cache.res[idx+1]
  mul!(r_H, R_map, r_h)

  δ0_H = cache.zeros[idx + 1]
  δ_H = test_multigrid_v_cycle!(cache, md, alg!, idx + 1, δ0_H, r_H)

  cache.res_checkout[idx+1] = false

  δ_h = cache.delta[idx]
  mul!(δ_h, P_map, δ_H)
  x_h .= x_h .+ δ_h

  post_smooth = alg!(solver, A, b, x_h; itmax=steps)
  x_h = solution(post_smooth)

  return x_h
end

# GPU support?
# TODO: Check alt-float support
function generate_mg_cache(md::MultigridData, kr_solver, float_type::DataType = Float64)
  zero = map(op->zeros(float_type, size(op, 1)), md.operators)
  delta = map(op->zeros(float_type, size(op, 1)), md.operators)
  res_checkout = fill(false, length(md))
  res = map(op->zeros(float_type, size(op, 1)), md.operators)
  solver = map(op->kr_solver(size(op)..., Vector{float_type}), md.operators)
  solver_type = typeof(first(solver))

  MultigridCache{Float64, solver_type}(zero, delta, res_checkout, res, solver)
end

cache = generate_mg_cache(md, CgSolver, Float64);

function run_test_multigrid_v_cycle!(cache::MultigridCache, md::MultigridData, alg!, cycles, x0, b)
  x = deepcopy(x0)
  for i in 1:cycles
    x .= test_multigrid_v_cycle!(cache, md, alg!, 1, x, b)
  end

  return x
end

res(L, x, b) = norm(L * x - b)

x0 = zeros(nv(sd));

test_x = multigrid_vcycles(x0, b, md, 25);
res(L, test_x, b)

in_mg_x = run_test_multigrid_v_cycle!(cache, md, cg!, 25, x0, b);
res(L, in_mg_x, b)

@time run_test_multigrid_v_cycle!(cache, md, cg!, 25, x0, b);
@time multigrid_vcycles(x0, b, md, 25);

struct MultiGridSolverKit
  md::MultigridData
  cache::MultigridCache
  alg
  x0::AbstractVector
  cycles::Int
end

function MultiGridSolverKit(md::MultigridData, kr_solver, float_type::DataType, kr_alg, x0::AbstractVector, cycles::Int)
  return MultiGridSolverKit(md, generate_mg_cache(md, kr_solver, float_type), kr_alg, x0, cycles)
end

mg = MultiGridSolverKit(md::MultigridData, CgSolver, Float64, cg!, zeros(nv(sd)), 2);

import LinearAlgebra: ldiv!
function ldiv!(y, A::MultiGridSolverKit, b)
  y .= run_test_multigrid_v_cycle!(A.cache, A.md, A.alg, A.cycles, A.x0, b);
end

println("Geometric Multigrid Preconditioned CG")
kr_mg_u, kr_mg_stats = cg(L, b, M = mg, ldiv = true, history = true);
@show kr_mg_stats.timer
@show last(kr_mg_stats.residuals)

println("Incomplete LU Preconditioned CG")
ilu0_L = ilu(L);
kr_u, kr_stats = cg(L, b, M = ilu0_L, ldiv = true, history = true);
@show kr_stats.timer
@show last(kr_stats.residuals)

println("Direct Solve")
splu_L = lu(L);
@time ds_u = splu_L \ b;
