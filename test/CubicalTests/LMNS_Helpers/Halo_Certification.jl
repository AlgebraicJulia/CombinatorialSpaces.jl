using Random

Base.@kwdef struct HaloCertificationConfig{FT <: AbstractFloat}
  max_shell::Int
  trials_per_shell::Int = 4
  amplitude::FT = FT(1e-8)
  atol::FT = FT(1e-12)
  rtol::FT = FT(1e-9)
  seed::Int = 20260504
end

struct HaloShellInfluence{FT <: AbstractFloat}
  shell::Int
  max_delta_u::FT
  max_delta_rho::FT
  activated_u::Bool
  activated_rho::Bool
  tested_u_indices::Int
  tested_rho_indices::Int
end

struct HaloCertificationReport{FT <: AbstractFloat}
  required_halo_u::Int
  required_halo_rho::Int
  required_halo::Int
  max_shell_tested::Int
  stats::Vector{HaloShellInfluence{FT}}
  config::HaloCertificationConfig{FT}
end

available_halo_depth(s::AbstractCubicalComplex2D) = min(hx(s), hy(s))

@inline function _axis_distance(i::Int, lo::Int, hi::Int)
  if i < lo
    return lo - i
  elseif i > hi
    return i - hi
  else
    return 0
  end
end

function shell_distance(::Val{2}, s::AbstractCubicalComplex2D, q::Int)
  x, y = quad_to_coord(s, q)
  dx_shell = _axis_distance(x, hx(s) + 1, nxq(s) - hx(s))
  dy_shell = _axis_distance(y, hy(s) + 1, nyq(s) - hy(s))
  return max(dx_shell, dy_shell)
end

function shell_distance(::Val{1}, s::AbstractCubicalComplex2D, e::Int)
  x, y, align = edge_to_coord(s, e)

  if align == X_ALIGN
    dx_shell = _axis_distance(x, hx(s) + 1, nxe(s) - hx(s))
    dy_shell = _axis_distance(y, hy(s) + 1, ny(s) - hy(s))
  else
    dx_shell = _axis_distance(x, hx(s) + 1, nx(s) - hx(s))
    dy_shell = _axis_distance(y, hy(s) + 1, nye(s) - hy(s))
  end

  return max(dx_shell, dy_shell)
end

function shell_indices(::Val{2}, s::AbstractCubicalComplex2D, shell::Int)
  shell >= 0 || throw(ArgumentError("shell must be non-negative, got $(shell)."))
  idx = Int[]
  for q in quads(s)
    shell_distance(Val(2), s, q) == shell && push!(idx, q)
  end
  return idx
end

function shell_indices(::Val{1}, s::AbstractCubicalComplex2D, shell::Int)
  shell >= 0 || throw(ArgumentError("shell must be non-negative, got $(shell)."))
  idx = Int[]
  for e in edges(s)
    shell_distance(Val(1), s, e) == shell && push!(idx, e)
  end
  return idx
end

@inline function _allclose_violation(a::FT, b::FT, atol::FT, rtol::FT) where FT <: AbstractFloat
  return abs(a - b) > atol + rtol * max(abs(a), abs(b))
end

function _activation_and_maxdelta(a::AbstractVector{FT}, b::AbstractVector{FT}, atol::FT, rtol::FT) where FT <: AbstractFloat
  length(a) == length(b) || throw(ArgumentError("Input vectors must have same length."))

  activated = false
  max_delta = zero(FT)
  @inbounds for i in eachindex(a)
    delta = abs(a[i] - b[i])
    if delta > max_delta
      max_delta = delta
    end
    if _allclose_violation(a[i], b[i], atol, rtol)
      activated = true
    end
  end

  return activated, max_delta
end

function _perturb_shell!(u::AbstractVector{FT}, rho::AbstractVector{FT}, u_idx::Vector{Int}, rho_idx::Vector{Int}, amp::FT, rng::AbstractRNG) where FT <: AbstractFloat
  @inbounds for i in u_idx
    u[i] += rand(rng, Bool) ? amp : -amp
  end

  @inbounds for i in rho_idx
    rho[i] += rand(rng, Bool) ? amp : -amp
  end

  return nothing
end

function _max_activated_shell(stats::Vector{HaloShellInfluence}, field::Symbol)
  max_shell = 0
  for s in stats
    activated = field === :U_star ? s.activated_u : s.activated_rho
    if activated
      max_shell = max(max_shell, s.shell)
    end
  end
  return max_shell
end

function certify_halo_influence(
  stepper::Function,
  s::AbstractCubicalComplex2D,
  state0;
  config::HaloCertificationConfig{FT},
) where FT <: AbstractFloat

  max_available = available_halo_depth(s)
  if config.max_shell < 1
    throw(ArgumentError("`config.max_shell` must be >= 1, got $(config.max_shell)."))
  end

  if config.max_shell > max_available
    throw(ArgumentError("`config.max_shell=$(config.max_shell)` exceeds available halo depth $(max_available)."))
  end

  if config.trials_per_shell < 1
    throw(ArgumentError("`config.trials_per_shell` must be >= 1, got $(config.trials_per_shell)."))
  end

  rng = MersenneTwister(config.seed)

  baseline_out = stepper(state0)
  baseline_u_int = interior(Val(1), baseline_out.U_star, s)
  baseline_rho_int = interior(Val(2), baseline_out.rho_star, s)

  stats = HaloShellInfluence{FT}[]

  for shell in 1:config.max_shell
    u_idx = shell_indices(Val(1), s, shell)
    rho_idx = shell_indices(Val(2), s, shell)

    shell_activated_u = false
    shell_activated_rho = false
    shell_max_delta_u = zero(FT)
    shell_max_delta_rho = zero(FT)

    for _ in 1:config.trials_per_shell
      pert_state = ComponentArray(
        U_star = deepcopy(state0.U_star),
        rho_star = deepcopy(state0.rho_star),
      )

      _perturb_shell!(pert_state.U_star, pert_state.rho_star, u_idx, rho_idx, config.amplitude, rng)

      pert_out = stepper(pert_state)
      pert_u_int = interior(Val(1), pert_out.U_star, s)
      pert_rho_int = interior(Val(2), pert_out.rho_star, s)

      activated_u, max_delta_u = _activation_and_maxdelta(pert_u_int, baseline_u_int, config.atol, config.rtol)
      activated_rho, max_delta_rho = _activation_and_maxdelta(pert_rho_int, baseline_rho_int, config.atol, config.rtol)

      shell_activated_u |= activated_u
      shell_activated_rho |= activated_rho
      shell_max_delta_u = max(shell_max_delta_u, max_delta_u)
      shell_max_delta_rho = max(shell_max_delta_rho, max_delta_rho)
    end

    push!(stats, HaloShellInfluence{FT}(
      shell,
      shell_max_delta_u,
      shell_max_delta_rho,
      shell_activated_u,
      shell_activated_rho,
      length(u_idx),
      length(rho_idx),
    ))
  end

  required_u = _max_activated_shell(stats, :U_star)
  required_rho = _max_activated_shell(stats, :rho_star)

  return HaloCertificationReport{FT}(
    required_u,
    required_rho,
    max(required_u, required_rho),
    config.max_shell,
    stats,
    config,
  )
end

function summarize_halo_report(report::HaloCertificationReport)
  println("Halo certification summary")
  println("  tested shells: 1:$(report.max_shell_tested)")
  println("  required halo (U_star): $(report.required_halo_u)")
  println("  required halo (rho_star): $(report.required_halo_rho)")
  println("  required halo (combined): $(report.required_halo)")
  println("  trials/shell: $(report.config.trials_per_shell), amplitude=$(report.config.amplitude)")

  for s in report.stats
    println(
      "  shell $(s.shell): ",
      "U(active=$(s.activated_u), max_delta=$(s.max_delta_u), n=$(s.tested_u_indices)); ",
      "rho(active=$(s.activated_rho), max_delta=$(s.max_delta_rho), n=$(s.tested_rho_indices))",
    )
  end

  return nothing
end
