# ── Helper: allocate a zero vector on the active backend ─────────────────────
kzeros(n) = KernelAbstractions.zeros(BACKEND, FT, n)

# ── Pre-allocate ALL intermediate and output buffers ─────────────────────────
const _nv = cache.nv_
const _ne = cache.ne_
const _nq = cache.nquads_

# Buffers used inside momentum_conservation
const _mc_U          = kzeros(_ne)   # hdg_1(U_star)
const _mc_rho        = kzeros(_nq)   # hdg_2(rho_star)
const _mc_Theta      = kzeros(_nq)   # hdg_2(Theta_star)
const _mc_inv_rho    = kzeros(_nq)
const _mc_u_vel      = kzeros(_ne)   # wdg_dd_01(1/rho, U)
const _mc_v          = kzeros(_ne)   # interp_dp_1(u_vel)
const _mc_V          = kzeros(_ne)   # interp_dp_1(U)
const _mc_dcd1_u     = kzeros(_nq)   # dcd_1(u_vel)
const _mc_div_term   = kzeros(_ne)   # wdg_dd_01(dcd1_u, U)
const _mc_ihs1_U     = kzeros(_ne)   # inv_hdg_1(U)
const _mc_wdg11_vU   = kzeros(_nq)   # wdg_11(v, ihs1_U)
const _mc_hdg2_wdg   = kzeros(_nq)   # hdg_2(wdg11_vU)
const _mc_dd0_hdg2   = kzeros(_ne)   # dd0(hdg2_wdg)    → L_term left part
const _mc_dd1_U      = kzeros(_nv)   # dd1(U)
const _mc_dbeta_V    = kzeros(_nv)   # d_beta(V)
const _mc_dd1U_dbV   = kzeros(_nv)   # dd1_U + dbeta_V
const _mc_ihs0_sum   = kzeros(_nv)   # inv_hdg_0(dd1U + dbetaV)
const _mc_wdg01_sv   = kzeros(_ne)   # wdg_01(ihs0_sum, v)
const _mc_hdg1_wdg   = kzeros(_ne)   # hdg_1(wdg01_sv) → L_term right part
const _mc_L_term     = kzeros(_ne)   # dd0_hdg2 + hdg1_wdg
const _mc_ihs1_u     = kzeros(_ne)   # inv_hdg_1(u_vel)
const _mc_wdg11_vu   = kzeros(_nq)   # wdg_11(v, ihs1_u)
const _mc_hdg2_vu    = kzeros(_nq)   # hdg_2(wdg11_vu)
const _mc_dd0_hdg2u  = kzeros(_ne)   # dd0(hdg2_vu)
const _mc_wdg_energy = kzeros(_ne)   # wdg_dd_01(rho, dd0_hdg2u) → energy
const _mc_pressure   = kzeros(_nq)   # pressure(Theta)  — allocated separately below
const _mc_diff_p     = kzeros(_ne)   # dd0(pressure)
# dlap_1 = dd0(dcd_1(u)) + dcd_2(dd1(u)) — needs temps for three-pass lap1
const _mc_dlap1_tmp1 = kzeros(_nv)   # temp nv for laplacian(1) pass 1
const _mc_dlap1_dbeta= kzeros(_nv)
const _mc_dlap1_tmp2 = kzeros(_ne)   # temp nq for laplacian(1) pass 2
const _mc_dlap1      = kzeros(_ne)   # dlap_1(u_vel)
# dlap_1_v = dcd_2(d_beta(v)) — d_beta maps ne→nv, then dcd_2 maps nv→ne
const _mc_dbeta_v    = kzeros(_nv)   # d_beta(v)
const _mc_viscous    = kzeros(_ne)   # p.mu*(dlap1 + dlap1v)
const _mc_sum_terms  = kzeros(_ne)   # -div_term - L_term + energy - diff_p + viscous
const _mc_result     = kzeros(_ne)   # -inv_hdg_1(sum_terms)  → momentum output

# Buffers used inside potential_temperature_continuity
const _pt_U          = kzeros(_ne)   # hdg_1(U_star)
const _pt_rho        = kzeros(_nq)   # hdg_2(rho_star)
const _pt_Theta      = kzeros(_nq)   # hdg_2(Theta_star)
const _pt_inv_rho    = kzeros(_nq)
const _pt_u_vel      = kzeros(_ne)   # wdg_dd_01(1/rho, U)
const _pt_v          = kzeros(_ne)   # interp_dp_1(u_vel)
const _pt_creation   = kzeros(_nq)   # Theta .* dcd_1(u_vel)
const _pt_dcd1_u     = kzeros(_nq)   # dcd_1(u_vel)
const _pt_dd0_Theta  = kzeros(_ne)   # dd0(Theta)   — no-flux dd0 on a quad field → ne
const _pt_ihs1_dd0T  = kzeros(_ne)   # inv_hdg_1(dd0_Theta)
const _pt_wdg11_vT   = kzeros(_nq)   # wdg_11(v, ihs1_dd0T)
const _pt_advection  = kzeros(_nq)   # hdg_2(wdg11_vT)
const _pt_theta_val  = kzeros(_nq)   # Theta ./ rho
const _pt_dlap0_tmp  = kzeros(_ne)   # dd0(theta) intermediate for dlap_0 = dcd_1(dd0(theta))
const _pt_dlap0      = kzeros(_nq)   # dlap_0(theta) = dcd_1(dd0(theta))
const _pt_diffusion  = kzeros(_nq)   # p.alpha * dlap0
const _pt_sum        = kzeros(_nq)   # -creation - advection + diffusion
const _pt_result     = kzeros(_nq)   # inv_hdg_2(sum)  → theta output

# Buffer for rhs! continuity: d1(U_star) → nq
const _rhs_d1_U      = kzeros(_nq)

# Buffers for smoothing
const _sm_rho_tmp    = kzeros(_nq)   # intermediate buffer for two-pass smooth (rho)
const _sm_theta_tmp  = kzeros(_nq)   # intermediate buffer for two-pass smooth (Theta)

# Buffers for WENO
const _weno_tmp_x = kzeros(_nq)
const _weno_tmp_y = kzeros(_nq)

# Buffers for gravity
if GRAVITY
    _mc_grav = kzeros(_ne)
end