using KernelAbstractions

abstract type AdvectionScheme end
struct Upwind <: AdvectionScheme end
struct WENO5   <: AdvectionScheme end

abstract type FluxLimiter end

"""Minmod limiter – most dissipative TVD limiter (first-order near extrema)."""
struct MinmodLimiter    <: FluxLimiter end

"""Van Leer limiter – smooth, second-order TVD limiter."""
struct VanLeerLimiter   <: FluxLimiter end

"""Superbee limiter – most compressive TVD limiter (Roe 1985)."""
struct SuperbeeLimiter  <: FluxLimiter end

"""Monotonized Central (MC) limiter – between minmod and superbee."""
struct MCLimiter        <: FluxLimiter end

"""Van Albada limiter – smooth and differentiable TVD limiter."""
struct VanAlbadaLimiter <: FluxLimiter end

@inline apply_limiter(::MinmodLimiter,    r::FT) where FT <: AbstractFloat =
  max(zero(FT), min(one(FT), r))

@inline apply_limiter(::VanLeerLimiter,   r::FT) where FT <: AbstractFloat =
  (r + abs(r)) / (one(FT) + abs(r))

@inline apply_limiter(::SuperbeeLimiter,  r::FT) where FT <: AbstractFloat =
  max(zero(FT), min(2 * r, one(FT)), min(r, 2 * one(FT)))

@inline apply_limiter(::MCLimiter,        r::FT) where FT <: AbstractFloat =
  max(zero(FT), min(min(2 * r, (one(FT) + r) / 2), 2 * one(FT)))

@inline apply_limiter(::VanAlbadaLimiter, r::FT) where FT <: AbstractFloat =
  (r * r + r) / (r * r + one(FT))

@kernel function wedge_product_01_upwind!(res, s, f0, f1)
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  f0_src = f0[src(s, x, y, align)]
  f0_tgt = f0[tgt(s, x, y, align)]
  f1_val = f1[idx]

  @inbounds res[idx] = f0_src * max(f1_val, 0) + f0_tgt * min(f1_val, 0)
end

function wedge_product_01(sch::AdvectionScheme, s::UniformCubicalComplex2D, f0, f1)
  res = KernelAbstractions.zeros(get_backend(f0), eltype(f0), ne(s))
  return wedge_product_01!(res, sch, s, f0, f1)
end

function wedge_product_01!(res, sch::Upwind, s::UniformCubicalComplex2D, f0, f1)
  backend = get_backend(f0)
  kernel =  wedge_product_01_upwind!(backend)
  kernel(res, s, f0, f1; ndrange = ne(s))
  return res
end

@kernel function wedge_product_11_upwind!(res, s, f1a, f1b)
  idx = @index(Global)
  x, y = quad_to_coord(s, idx)

  # Edges of the quad, ordered as: bottom, right, top, left
  e1, e2, e3, e4 = quad_edges(s, x, y)

  f1a_e1 = f1a[e1]; f1a_e2 = f1a[e2]; f1a_e3 = f1a[e3]; f1a_e4 = f1a[e4]
  f1b_e1 = f1b[e1]; f1b_e2 = f1b[e2]; f1b_e3 = f1b[e3]; f1b_e4 = f1b[e4]

  avg_x_flow = (f1a_e2 + f1a_e4) / 2
  avg_y_flow = (f1a_e1 + f1a_e3) / 2

  # Upwind selection for each f1b edge value
  x_upwind = avg_x_flow >= 0 ? f1b_e1 : f1b_e3
  y_upwind = avg_y_flow >= 0 ? f1b_e4 : f1b_e2

  @inbounds res[idx] = y_upwind * avg_y_flow - x_upwind * avg_x_flow
end

@kernel function wedge_product_11_WENO5!(res, s, f1a, f1b, eps)
  idx = @index(Global)
  x, y = quad_to_coord(s, idx)

  # Edges of the quad, ordered as: bottom, right, top, left
  e1, e2, e3, e4 = quad_edges(s, x, y)

  f1a_e1 = f1a[e1]; f1a_e2 = f1a[e2]; f1a_e3 = f1a[e3]; f1a_e4 = f1a[e4]

  f1b_x = f1b[e1]; f1b_yp1 = f1b[e2]; f1b_xp1 = f1b[e3]; f1b_y = f1b[e4]

  avg_x_flow = (f1a_e2 + f1a_e4) / 2
  avg_y_flow = (f1a_e1 + f1a_e3) / 2

  # WENO5 reconstruction for each direction
  # If too close to the boundary, fall back to upwind selection

  # Need room for fm2 and fp3 stencils in both directions.
  if x <= 2 || x >= nx(s) - 2 || y <= 2 || y >= ny(s) - 2
    # Near boundaries, use upwind selection
     x_upwind = avg_x_flow >= 0 ? f1b_x : f1b_xp1
     y_upwind = avg_y_flow >= 0 ? f1b_y : f1b_yp1
  else
    f1b_xm1 = f1b[quad_edge_offset(s, x, y, X_ALIGN, -1)]; f1b_xp2 = f1b[quad_edge_offset(s, x, y, X_ALIGN, 2)]
    f1b_xm2 = f1b[quad_edge_offset(s, x, y, X_ALIGN, -2)]; f1b_xp3 = f1b[quad_edge_offset(s, x, y, X_ALIGN, 3)]

    f1b_ym1 = f1b[quad_edge_offset(s, x, y, Y_ALIGN, -1)]; f1b_yp2 = f1b[quad_edge_offset(s, x, y, Y_ALIGN, 2)]
    f1b_ym2 = f1b[quad_edge_offset(s, x, y, Y_ALIGN, -2)]; f1b_yp3 = f1b[quad_edge_offset(s, x, y, Y_ALIGN, 3)]

    # WENO5 reconstruction for x-direction
    x_upwind = if avg_x_flow >= 0
      weno5_point(f1b_xm2, f1b_xm1, f1b_x, f1b_xp1, f1b_xp2, eps)
    else
      weno5_point(f1b_xp3, f1b_xp2, f1b_xp1, f1b_x, f1b_xm1, eps)
    end

    y_upwind = if avg_y_flow >= 0
      weno5_point(f1b_ym2, f1b_ym1, f1b_y, f1b_yp1, f1b_yp2, eps)
    else
      weno5_point(f1b_yp3, f1b_yp2, f1b_yp1, f1b_y, f1b_ym1, eps)
    end
  end

  @inbounds res[idx] = y_upwind * avg_y_flow - x_upwind * avg_x_flow
end

function wedge_product_11(sch::AdvectionScheme, s::UniformCubicalComplex2D, f1a, f1b)
  res = KernelAbstractions.zeros(get_backend(f1a), eltype(f1a), nquads(s))
  return wedge_product_11!(res, sch, s, f1a, f1b)
end

function wedge_product_11!(res, sch::Upwind, s::UniformCubicalComplex2D, a, b)
  backend = get_backend(a)
  kernel = wedge_product_11_upwind!(backend)
  kernel(res, s, a, b; ndrange = nquads(s))
  return res
end

function wedge_product_11!(res, sch::WENO5, s::UniformCubicalComplex2D, a, b, eps = nothing)
  backend = get_backend(a)
  FT = eltype(a)
  eps_T = eps === nothing ? FT(1e-6) : FT(eps)
  kernel = wedge_product_11_WENO5!(backend)
  kernel(res, s, a, b, eps_T; ndrange = nquads(s))
  return res
end

function wedge_product(::Val{i}, ::Val{j}, sch::AdvectionScheme, s::UniformCubicalComplex2D, a, b) where {i, j}
  if i == 0 && j == 1
    return wedge_product_01(sch, s, a, b)
  elseif i == 1 && j == 0
    return wedge_product_01(sch, s, b, a)
  elseif i == 1 && j == 1
    return wedge_product_11(sch, s, a, b)
  else
    error("Wedge product not implemented for forms of degree ($i, $j) with scheme $(typeof(sch)).")
  end
end
