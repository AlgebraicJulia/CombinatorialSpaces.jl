function build_dec_kernels(s::UniformCubicalComplex2D{FT}) where FT <: AbstractFloat
  cache = Adapt.adapt(USE_CUDA ? CUDABackend() : CPU(), UniformDECCache(s))

  d0(x) = exterior_derivative(Val(1), cache, x)
  d1(x) = exterior_derivative(Val(1), cache, x)

  dd0(x) = no_flux_dual_derivative(Val(0), cache, x)
  dd1(x) = dual_derivative(Val(1), cache, x)

  hdg_0(x) = hodge_star(Val(0), cache, x)
  hdg_1(x) = hodge_star(Val(1), cache, x)
  hdg_2(x) = hodge_star(Val(2), cache, x)

  inv_hdg_0(x) = inv_hodge_star(Val(0), cache, x)
  inv_hdg_1(x) = inv_hodge_star(Val(1), cache, x)
  inv_hdg_2(x) = inv_hodge_star(Val(2), cache, x)

  d_beta(x) = d_beta_mul(cache, x)

  wdg_01(f, a) = wedge_product(Val(0), Val(1), cache, f, a)
  wdg_11(a, b) = wedge_product(Val(1), Val(1), cache, a, b)
  wdg_dd_01(f, a) = wedge_product_dd(Val(0), Val(1), cache, f, a)

  dcd_1(x) = dual_codifferential(Val(1), cache, x)
  dcd_2(x) = dual_codifferential(Val(2), cache, x)


  # TODO: Replace with the fused kernels
  dlap_0(x) = dcd_1(dd0(x))

  dlap_1(x) = dd0(dcd_1(x)) + dcd_2(dd1(x))
  dlap_1_v(x) = dcd_2(d_beta(x))

  interp_dp_1(x) = interpolate_dp(Val(1), cache, x)

  return (; d0=d0, d1=d1, dd0=dd0, dd1=dd1, 
  hdg_0=hdg_0, hdg_1=hdg_1, hdg_2=hdg_2, 
  inv_hdg_0=inv_hdg_0, inv_hdg_1=inv_hdg_1, inv_hdg_2=inv_hdg_2,
  d_beta=d_beta, wdg_01=wdg_01, wdg_11=wdg_11, wdg_dd_01=wdg_dd_01, 
  dcd_1=dcd_1, dcd_2=dcd_2, 
  dlap_0=dlap_0, dlap_1=dlap_1, dlap_1_v=dlap_1_v,
  interp_dp_1=interp_dp_1)
end