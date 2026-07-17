kzeros(n) = KernelAbstractions.zeros(BACKEND, FT, n)

# ── Pre-allocate ALL intermediate and output buffers ─────────────────────────
const _nv = nv(s)
const _ne = ne(s)
const _nq = nquads(s)
const _nb = nboids(s)

# ── momentum_conservation 3D ──────────────────────────────────────────────────
const _mc_U         = kzeros(_nq)
const _mc_rho       = kzeros(_nb)
const _mc_Theta     = kzeros(_nb)
const _mc_inv_rho   = kzeros(_nb)
const _mc_u       = kzeros(_nq)
const _mc_v         = kzeros(_ne)
const _mc_dcd1_vel  = kzeros(_nb)
const _mc_div_term  = kzeros(_nq)
const _mc_ihs2_U    = kzeros(_nq)
const _mc_wdg12_vU  = kzeros(_nb)
const _mc_hdg3_wdg  = kzeros(_nb)
const _mc_dd0_hdg3  = kzeros(_nq)
const _mc_dd1_U     = kzeros(_ne)
const _mc_ihs1_dd1U = kzeros(_ne)
const _mc_wdg11_v   = kzeros(_nq)
const _mc_hdg2_wdg  = kzeros(_nq)
const _mc_adv_term  = kzeros(_nq)
const _mc_ihs2_vel  = kzeros(_nq)
const _mc_wdg12_vvel= kzeros(_nb)
const _mc_hdg3_vvel = kzeros(_nb)
const _mc_dd0_hdg3v = kzeros(_nq)
const _mc_energy    = kzeros(_nq)
const _mc_pressure  = kzeros(_nb)
const _mc_diff_p    = kzeros(_nq)
const _mc_dlap1_tmp1= kzeros(_ne)
# const _mc_dlap1_tmp2= kzeros(_nb)
const _mc_viscous_1 = kzeros(_nq)
const _mc_viscous_2 = kzeros(_nq)
const _mc_viscous   = kzeros(_nq)
const _mc_sum_terms = kzeros(_nq)

# ── potential_temperature_continuity 3D ───────────────────────────────────────
const _pt_U          = kzeros(_nq)
const _pt_rho        = kzeros(_nb)
const _pt_Theta      = kzeros(_nb)
const _pt_inv_rho    = kzeros(_nb)
const _pt_u        = kzeros(_nq)
const _pt_v          = kzeros(_ne)
const _pt_theta      = kzeros(_nb)
const _pt_dcd1_vel   = kzeros(_nb)
const _pt_creation   = kzeros(_nb)
const _pt_dd0_Theta  = kzeros(_nq)
const _pt_ihs2_dd0T  = kzeros(_nq)
const _pt_wdg12_vT   = kzeros(_nb)
const _pt_advection  = kzeros(_nb)
const _pt_dd0_theta  = kzeros(_nq)
# const _pt_dlap0_tmp  = kzeros(_ne)
const _pt_diffusion  = kzeros(_nb)
const _pt_sum        = kzeros(_nb)

# ── rhs! scratch ──────────────────────────────────────────────────────────────
const _rhs_d2_U      = kzeros(_nb)

# For velocity interpolations
const _X_vel         =  kzeros(_nb)
const _Y_vel         =  kzeros(_nb)
const _Z_vel         =  kzeros(_nb)

# For expanded codifs
const _mc_tmp_1 = kzeros(_nq)
const _mc_tmp_2 = kzeros(_nb)

const _mc_tmp_3 = kzeros(_ne)
const _mc_tmp_4 = kzeros(_nq)

const _pt_tmp_5 = kzeros(_nq)
const _pt_tmp_6 = kzeros(_nb)

const _pt_tmp_7 = kzeros(_nq)
const _pt_tmp_8 = kzeros(_nb)

# For WENO

const _mc_wdg11_tmpa = kzeros(_nq)
const _mc_wdg11_tmpb = kzeros(_nq)

const _mc_wdg12_tmpx = kzeros(_nb)
const _mc_wdg12_tmpy = kzeros(_nb)
const _mc_wdg12_tmpz = kzeros(_nb)

const _mc_wdg12v_tmpx = kzeros(_nb)
const _mc_wdg12v_tmpy = kzeros(_nb)
const _mc_wdg12v_tmpz = kzeros(_nb)

const _pt_wdg12_tmpx = kzeros(_nb)
const _pt_wdg12_tmpy = kzeros(_nb)
const _pt_wdg12_tmpz = kzeros(_nb)