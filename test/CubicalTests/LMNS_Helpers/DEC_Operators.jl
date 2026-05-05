function build_dec_operators(model::AbstractSimulationModel, s::UniformCubicalComplex2D, to_device::Function = identity; smoothing_coefficients::NamedTuple = (;))
  cpu_d0 = exterior_derivative(Val(0), s)
  cpu_d1 = exterior_derivative(Val(1), s)

  # Enforce no-flux boundary condition on density.
  cpu_dual_d0 = no_flux_dual_derivative(Val(0), s)
  cpu_dual_d1 = dual_derivative(Val(1), s)

  # Closing the dual cells.
  cpu_d_beta = dual_derivative_beta(Val(1), s)

  cpu_hdg_1 = hodge_star(Val(1), s)
  cpu_hdg_2 = hodge_star(Val(2), s)

  cpu_inv_hdg_0 = inv_hodge_star(Val(0), s)
  cpu_inv_hdg_1 = inv_hodge_star(Val(1), s)
  cpu_inv_hdg_2 = inv_hodge_star(Val(2), s)

  cpu_delta1 = codifferential(Val(1), s)
  cpu_dual_delta1 = dual_codifferential(Val(1), s)

  # No gradient through boundary edges.
  cpu_dlap0 = cpu_hdg_2 * cpu_d1 * cpu_inv_hdg_1 * cpu_dual_d0

  # Enforce closure of dual cells.
  cpu_dlap1 = cpu_hdg_1 * cpu_d0 * cpu_inv_hdg_0 * cpu_dual_d1 + cpu_dual_d0 * cpu_hdg_2 * cpu_d1 * cpu_inv_hdg_1
  cpu_dlap1_v = cpu_hdg_1 * cpu_d0 * cpu_inv_hdg_0 * cpu_d_beta

  cpu_dd0_h2 = cpu_dual_d0 * cpu_hdg_2
  cpu_ih0_dd1 = cpu_inv_hdg_0 * cpu_dual_d1
  cpu_ih0_db = cpu_inv_hdg_0 * cpu_d_beta

  return (
    d0 = to_device(cpu_d0),
    d1 = to_device(cpu_d1),
    dual_d0 = to_device(cpu_dual_d0),
    dual_d1 = to_device(cpu_dual_d1),
    d_beta = to_device(cpu_d_beta),
    hdg_1 = to_device(cpu_hdg_1),
    hdg_2 = to_device(cpu_hdg_2),
    inv_hdg_0 = to_device(cpu_inv_hdg_0),
    inv_hdg_1 = to_device(cpu_inv_hdg_1),
    inv_hdg_2 = to_device(cpu_inv_hdg_2),
    delta1 = to_device(cpu_delta1),
    dual_delta1 = to_device(cpu_dual_delta1),
    dlap0 = to_device(cpu_dlap0),
    dlap1 = to_device(cpu_dlap1),
    dlap1_v = to_device(cpu_dlap1_v),
    dd0_h2 = to_device(cpu_dd0_h2),
    ih0_dd1 = to_device(cpu_ih0_dd1),
    ih0_db = to_device(cpu_ih0_db),
    smoothing = build_dec_smoothing(model, s, to_device, smoothing_coefficients),
  )
end

function build_smoothing(s::UniformCubicalComplex2D, to_device::Function, coefficients::NamedTuple)
  smoothing_pairs = (
    name => to_device(smoothing_dual0(s, -Float64(c)) * smoothing_dual0(s, Float64(c)))
    for (name, c) in pairs(coefficients)
  )
  return (; smoothing_pairs...)
end

build_dec_smoothing(::LMNSModel, s::UniformCubicalComplex2D, to_device::Function, coefficients::NamedTuple) = (;)

function build_dec_smoothing(::LMNSHModel, s::UniformCubicalComplex2D, to_device::Function, coefficients::NamedTuple)
  return build_smoothing(s, to_device, coefficients)
end

build_dec_smoothing(::MHDModel, s::UniformCubicalComplex2D, to_device::Function, coefficients::NamedTuple) = (;)
