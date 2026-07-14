using OrdinaryDiffEqSSPRK
using DiffEqCallbacks
using ComponentArrays
using Distributions
using KernelAbstractions

# ── Mesh & topology ───────────────────────────────────────────────────────────
const neighbors = topo.cache.neighbors

const has_west = neighbors.west != MPI.PROC_NULL
const has_east = neighbors.east != MPI.PROC_NULL
const has_south = neighbors.south != MPI.PROC_NULL
const has_north = neighbors.north != MPI.PROC_NULL

const _halo_west = has_west ? HALO : 0
const _halo_east = has_east ? HALO : 0
const _halo_south = has_south ? HALO : 0
const _halo_north = has_north ? HALO : 0

const s = worker_mesh(topo, (LX, LY); halo_west = _halo_west, halo_east = _halo_east,
    halo_south = _halo_south, halo_north = _halo_north)

const cart_comm = topo.cart_comm
const cart_rank = topo.cart_rank
const cart_coords = MPI.Cart_coords(cart_comm)

if use_amdgpu(USE_XPU)
    devices = AMDGPU.devices()
    ndevices = length(devices)

    local_rank = MPI.Comm_rank(topo.local_comm)
    gpu_id = (local_rank % ndevices) + 1
    AMDGPU.device!(devices[gpu_id])

    node_name = MPI.Get_processor_name()
    mapping_str = "Rank $world_rank -> Node: $node_name, GPU: $gpu_id"
    all_mappings = MPI.gather(mapping_str, topo.cart_comm; root=0)
    if world_rank == 0
        println("=========================================================")
        println("MPI Topology & GPU Assignment:")
        for m in all_mappings
            println("   $m")
        end
        println("=========================================================")
    end
else
    world_rank == 0 && println("Using CPU for computation...")
end

function boundary_edges(topo::MPITopology{WorkerCache{2}}, s::AbstractCubicalComplex2D)
    west_edges  = has_west ? Int[] : left_edges(s)
    east_edges  = has_east ? Int[] : right_edges(s)
    south_edges = has_south ? Int[] : bottom_edges(s)
    north_edges = has_north ? Int[] : top_edges(s)

    return vcat(west_edges, east_edges, south_edges, north_edges)
end

# ── to_device ────────────────────────────────────────────────────────────────
function to_device(arr::AbstractVector{T}) where T
    use_amdgpu(USE_XPU) && return AMDGPU.ROCVector{T}(arr)
    return arr
end

function to_device(ca::ComponentVector)
    return ComponentArray(map(to_device, NamedTuple(ca)))
end

const bdry_edges = boundary_edges(topo, s)

cart_rank == 0 && println("MPI worker grid: $(w_dims[1])×$(w_dims[2])")
# println("Worker $cart_rank | coords=$cart_coords | mesh=$(nxr(s))×$(nyr(s)) real quads")
flush(stdout)

# TODO: Generalize this to be able to use CUDA
const BACKEND = use_amdgpu(USE_XPU) ? ROCBackend() : CPU()
const cache = Adapt.adapt(BACKEND, UniformDECCache(s))

#TODO: Turn off/on WENO
const weno_cache  = Adapt.adapt(BACKEND, WENO5Cache(s))

include(joinpath(@__DIR__, "buffers.jl"))

const hdg_1       = x -> hodge_star(Val(1), cache, x)
const hdg_2       = x -> hodge_star(Val(2), cache, x)

const inv_hdg_1   = x -> inv_hodge_star(Val(1), cache, x)
const inv_hdg_2   = x -> inv_hodge_star(Val(2), cache, x)

if haskey(CONFIG, "Smoothing")
    const rho_c = FT(get(CONFIG["Smoothing"], "rho_smooth_constant", 0))
    const theta_c = FT(get(CONFIG["Smoothing"], "theta_smooth_constant", 0))
else
    const rho_c = const theta_c = 0
end

const use_rho_smooth = (rho_c != 0)
const use_theta_smooth = (theta_c != 0)

if use_rho_smooth
    world_rank == 0 && println("Activated rho smoothing: $rho_c")
    const rho_smooth_cache = Adapt.adapt(BACKEND, SmoothingCache(s, rho_c))
end

if use_theta_smooth
    world_rank == 0 && println("Activated Theta smoothing: $theta_c")
    const theta_smooth_cache = Adapt.adapt(BACKEND, SmoothingCache(s, theta_c))
end

# ── Physics ───────────────────────────────────────────────────────────────────
struct PhysicalParameters{FT <: AbstractFloat}
    mu::FT # Viscosity
    alpha::FT # Thermal diffusivity
end

const p_phys = PhysicalParameters{FT}(FT(1) / RE, FT(1) / (RE * PR))

const P₀ = FT(1e5)
const Cₚ = FT(1006)
const R_gas = FT(287)
const gₐ = -FT(9.81)
const ρᵣ = FT(1) # Reference density
const θᵣ = FT(300) # Reference potential temperature, temperature at P₀
const Pᵣ = ρᵣ * R_gas * θᵣ

function pressure!(res::AbstractVector{FT}, Theta::AbstractVector{FT}) where FT <: AbstractFloat
    R_Cₚ = R_gas / Cₚ
    res .= (Theta .* R_gas .* (P₀ .^ .-R_Cₚ)) .^ (FT(1) ./ (FT(1) .- R_Cₚ))
    return res
end

# Assuming theta is constant, pass in a float
function hydrostatic_pressure(theta::FT, h::FT) where FT <: AbstractFloat
    return (1 / (Cₚ * theta) * (P₀)^(R_gas / Cₚ) * gₐ * h + Pᵣ^(R_gas / Cₚ))^(Cₚ / R_gas)
end
  
function hydrostatic_density(theta::FT, h::FT) where FT <: AbstractFloat
    Pₕ = hydrostatic_pressure(theta, h)
    return (Pₕ / (R_gas * theta)) * (P₀ / Pₕ)^(R_gas / Cₚ)
end 

if GRAVITY
    const g_dual = to_device(map(edges(s)) do e
        if is_X_aligned(e, s)
            return gₐ * dual_edge_len(s, e)
        else
            return 0.0
        end
    end)
end

# ── ExchangeHandler ───────────────────────────────────────────────────────────
const ex_stream = DataStream(Datum[Datum{Edge,2}("U_star", "fields", FT), Datum{Quad,2}("rho_star", "fields", FT), Datum{Quad,2}("Theta_star", "fields", FT)])
const ex_handler = ExchangeHandler(ex_stream, topo, s; backend = BACKEND)

# ── IO stream (real-cell data only, sent to output processes) ─────────────────
const io_stream = DataStream(Datum[
    Datum{Edge,2}("U", "fields", FT),
    Datum{Quad,2}("rho", "fields", FT), 
    Datum{Quad,2}("Theta", "fields", FT),
    ])

# ── Initial conditions (San & Kara 2015) ─────────────────────────────────────

include(joinpath(@__DIR__, "Examples", "$SIM_NAME.jl"))

const u0 = ComponentVector(; U_star = to_device(U_star_0), 
                             rho_star = to_device(rho_star_0), 
                             Theta_star = to_device(Theta_star_0))

MPI.Barrier(cart_comm)

# ── RHS ───────────────────────────────────────────────────────────────────────
# function momentum_conservation(u::ComponentVector, p::PhysicalParameters{FT}) where {FT <: AbstractFloat}
#     U   = hdg_1(u.U_star)
#     rho = hdg_2(u.rho_star)
#     Theta = hdg_2(u.Theta_star)

#     u = wdg_dd_01(FT(1) ./ rho, U)
#     v   = interp_dp_1(u)
#     V   = interp_dp_1(U)

#     enforce_bc_v!(v); enforce_bc_V!(V)

#     div_term = wdg_dd_01(dcd_1(u), U)

#     L_term   = dd0(hdg_2(wdg_11(v, inv_hdg_1(U)))) +
#                hdg_1(wdg_01(inv_hdg_0(dd1(U) + d_beta(V)), v))

#     energy   = FT(0.5) .* wdg_dd_01(rho, dd0(hdg_2(wdg_11(v, inv_hdg_1(u)))))

#     diff_p   = dd0(pressure(Theta))

#     viscous  = p.mu * (dlap_1(u) + dlap_1_v(v)) # TODO: Split this up for interior/exterior?

#     # TODO: Will also need to body forces later 

#     result = -inv_hdg_1(.-div_term .- L_term .+ energy .- diff_p .+ viscous)

#     enforce_bc_U!(result)
#     return result
# end

# function potential_temperature_continuity(u::ComponentVector, p::PhysicalParameters{FT}) where {FT <: AbstractFloat}
#     U     = hdg_1(u.U_star)
#     rho   = hdg_2(u.rho_star)
#     Theta = hdg_2(u.Theta_star)

#     u   = wdg_dd_01(FT(1) ./ rho, U)
#     v     = interp_dp_1(u)
#     theta = Theta ./ rho

#     creation  = Theta .* dcd_1(u)

#     advection = hdg_2(wdg_11(v, inv_hdg_1(dd0(Theta))))

#     diffusion = p.alpha * dlap_0(theta)

#     return inv_hdg_2(.-creation .- advection .+ diffusion)
# end

# function rhs!(du, u, p, t)
#     du.U_star     .= momentum_conservation(u, p)
#     du.Theta_star .= potential_temperature_continuity(u, p)
#     du.rho_star   .= d1(u.U_star)
#     return nothing
# end

function momentum_conservation!(result::AbstractVector{FT}, u::ComponentVector, p::PhysicalParameters{FT}) where {FT <: AbstractFloat}
    # ── hodge lifts ──────────────────────────────────────────────────────────
    hodge_star!(_mc_U,     Val(1), cache, u.U_star)
    hodge_star!(_mc_rho,   Val(2), cache, u.rho_star)
    hodge_star!(_mc_Theta, Val(2), cache, u.Theta_star)

    # ── u_vel = wdg_dd_01(1/rho, U) ─────────────────────────────────────────
    _mc_inv_rho .= FT(1) ./ _mc_rho
    wedge_product_dd!(_mc_u_vel, Val(0), Val(1), cache, _mc_inv_rho, _mc_U)

    # ── velocity interpolations ──────────────────────────────────────────────
    interpolate_dp!(_mc_v, Val(1), cache, _mc_u_vel)
    interpolate_dp!(_mc_V, Val(1), cache, _mc_U)
    enforce_bc_v!(_mc_v)
    enforce_bc_V!(_mc_V)

    # ── div_term = wdg_dd_01(dcd_1(u_vel), U) ───────────────────────────────
    dual_codifferential!(_mc_dcd1_u, Val(1), cache, _mc_u_vel)
    wedge_product_dd!(_mc_div_term, Val(0), Val(1), cache, _mc_dcd1_u, _mc_U)
    
    # ── L_term left: dd0(hdg_2(wdg_11(v, inv_hdg_1(U)))) ────────────────────
    inv_hodge_star!(_mc_ihs1_U,  Val(1), cache, _mc_U)

    # TODO: Better branching to WENO
    # wedge_product!(_mc_wdg11_vU, Val(1), Val(1), cache, _mc_v, _mc_ihs1_U)
    wedge_product_11!(_mc_wdg11_vU, _weno_tmp_x, _weno_tmp_y, WENO5(), weno_cache, _mc_v, _mc_ihs1_U)

    hodge_star!(_mc_hdg2_wdg,    Val(2), cache, _mc_wdg11_vU)
    no_flux_dual_derivative!(_mc_dd0_hdg2, Val(0), cache, _mc_hdg2_wdg)

    # ── L_term right: hdg_1(wdg_01(inv_hdg_0(dd1(U) + d_beta(V)), v)) ───────
    dual_derivative!(_mc_dd1_U,   Val(1), cache, _mc_U)
    d_beta_mul!(_mc_dbeta_V,      cache, _mc_V)
    _mc_dd1U_dbV .= _mc_dd1_U .+ _mc_dbeta_V
    inv_hodge_star!(_mc_ihs0_sum, Val(0), cache, _mc_dd1U_dbV)

    # wedge_product!(_mc_wdg01_sv,  Val(0), Val(1), cache, _mc_ihs0_sum, _mc_v)
    wedge_product_01!(_mc_wdg01_sv, WENO5(), weno_cache, _mc_ihs0_sum, _mc_v)

    hodge_star!(_mc_hdg1_wdg,     Val(1), cache, _mc_wdg01_sv)

    # ── L_term = left + right ────────────────────────────────────────────────
    _mc_L_term .= _mc_dd0_hdg2 .+ _mc_hdg1_wdg

    # ── energy = 0.5 * wdg_dd_01(rho, dd0(hdg_2(wdg_11(v, inv_hdg_1(u_vel))))) ──
    inv_hodge_star!(_mc_ihs1_u,  Val(1), cache, _mc_u_vel)

    # wedge_product!(_mc_wdg11_vu, Val(1), Val(1), cache, _mc_v, _mc_ihs1_u)
    wedge_product_11!(_mc_wdg11_vu, _weno_tmp_x, _weno_tmp_y, WENO5(), weno_cache, _mc_v, _mc_ihs1_u)

    hodge_star!(_mc_hdg2_vu,     Val(2), cache, _mc_wdg11_vu)
    no_flux_dual_derivative!(_mc_dd0_hdg2u, Val(0), cache, _mc_hdg2_vu)
    wedge_product_dd!(_mc_wdg_energy, Val(0), Val(1), cache, _mc_rho, _mc_dd0_hdg2u)
    _mc_wdg_energy .*= FT(0.5)

    # ── diff_p = dd0(pressure(Theta)) ────────────────────────────────────────
    pressure!(_mc_pressure, _mc_Theta)
    no_flux_dual_derivative!(_mc_diff_p, Val(0), cache, _mc_pressure)

    # Term 1: dcd_2(dd1(u_vel))
    dual_derivative!(_mc_dlap1_tmp1,  Val(1), cache, _mc_u_vel)   # dd1: ne → nv (tmp1 is nv)
    d_beta_mul!(_mc_dbeta_v, cache, _mc_v)
    _mc_dlap1_dbeta .= _mc_dlap1_tmp1 .+ _mc_dbeta_v
    dual_codifferential!(_mc_dlap1,   Val(2), cache, _mc_dlap1_dbeta)  # dcd_2: nv → ne

    # Term 2: dd0(dcd_1(u_vel))  — reuses _mc_dcd1_u (nq) already computed above
    no_flux_dual_derivative!(_mc_dlap1_tmp2, Val(0), cache, _mc_dcd1_u)  # dd0: nq → ne (tmp2 is nq, used as ne output here)

    _mc_viscous .= p.mu .* (_mc_dlap1 .+ _mc_dlap1_tmp2)   # full dlap_1(u_vel) in _mc_dlap1


    # ── assemble and apply inv_hdg_1 ─────────────────────────────────────────
    _mc_sum_terms .= .-_mc_div_term .- _mc_L_term .+ _mc_wdg_energy .- _mc_diff_p .+ _mc_viscous
    
    # TODO: Handle this a bit better
    if GRAVITY
        wedge_product_dd!(_mc_grav, Val(0), Val(1), cache, _mc_rho, g_dual)
        _mc_sum_terms .+= _mc_grav
    end
    
    inv_hodge_star!(_mc_result, Val(1), cache, _mc_sum_terms)
    result .= .-_mc_result
    enforce_bc_U!(result)
    return result
end

function potential_temperature_continuity!(result::AbstractVector{FT}, u::ComponentVector, p::PhysicalParameters{FT}) where {FT <: AbstractFloat}
    # ── hodge lifts ──────────────────────────────────────────────────────────
    hodge_star!(_pt_U,     Val(1), cache, u.U_star)
    hodge_star!(_pt_rho,   Val(2), cache, u.rho_star)
    hodge_star!(_pt_Theta, Val(2), cache, u.Theta_star)

    # ── u_vel = wdg_dd_01(1/rho, U) ─────────────────────────────────────────
    _pt_inv_rho .= FT(1) ./ _pt_rho
    wedge_product_dd!(_pt_u_vel, Val(0), Val(1), cache, _pt_inv_rho, _pt_U)

    # ── v = interp_dp_1(u_vel) ───────────────────────────────────────────────
    interpolate_dp!(_pt_v, Val(1), cache, _pt_u_vel)

    # ── creation = Theta .* dcd_1(u_vel) ────────────────────────────────────
    dual_codifferential!(_pt_dcd1_u, Val(1), cache, _pt_u_vel)
    _pt_creation .= _pt_Theta .* _pt_dcd1_u

    # ── advection = hdg_2(wdg_11(v, inv_hdg_1(dd0(Theta)))) ─────────────────
    no_flux_dual_derivative!(_pt_dd0_Theta, Val(0), cache, _pt_Theta)
    inv_hodge_star!(_pt_ihs1_dd0T, Val(1), cache, _pt_dd0_Theta)

    # wedge_product!(_pt_wdg11_vT,   Val(1), Val(1), cache, _pt_v, _pt_ihs1_dd0T)
    wedge_product_11!(_pt_wdg11_vT, _weno_tmp_x, _weno_tmp_y, WENO5(), weno_cache, _pt_v, _pt_ihs1_dd0T)

    hodge_star!(_pt_advection, Val(2), cache, _pt_wdg11_vT)

    # ── diffusion = alpha * dlap_0(theta) where dlap_0 = dcd_1(dd0(theta)) ──
    _pt_theta_val .= _pt_Theta ./ _pt_rho
    # dlap_0(theta) = dcd_1(dd0(theta)): dd0 maps nq→ne, dcd_1 maps ne→nq
    no_flux_dual_derivative!(_pt_dlap0_tmp, Val(0), cache, _pt_theta_val)
    dual_codifferential!(_pt_dlap0, Val(1), cache, _pt_dlap0_tmp)
    _pt_diffusion .= p.alpha .* _pt_dlap0

    # ── assemble and apply inv_hdg_2 ─────────────────────────────────────────
    _pt_sum .= .-_pt_creation .- _pt_advection .+ _pt_diffusion
    inv_hodge_star!(result, Val(2), cache, _pt_sum)
    return result
end

function rhs!(du, u, p, t)
    momentum_conservation!(du.U_star, u, p)
    potential_temperature_continuity!(du.Theta_star, u, p)

    exterior_derivative!(_rhs_d1_U, Val(1), cache, u.U_star)
    du.rho_star .= _rhs_d1_U
    return nothing
end


# ── Callbacks ─────────────────────────────────────────────────────────────────
const EXCHANGE_EVERY = 1

periodic_cb = DiscreteCallback(
    (u, t, integrator) -> integrator.iter > 0 && integrator.iter % EXCHANGE_EVERY == 0,
    integrator -> begin
        u = integrator.u
        exchange!(ex_handler, (
            U_star    = u.U_star,
            rho_star  = u.rho_star,
            Theta_star = u.Theta_star,
        ))
    end;
    initialize = (c, u, t, integrator) -> exchange!(ex_handler, (
        U_star     = u.U_star,
        rho_star   = u.rho_star,
        Theta_star = u.Theta_star,
    )),
    save_positions = (false, false),
)

# Note this is safe since we edit the tmp buffer first
function smoothing_run(integrator)
    use_rho_smooth && smooth_dual0_fused!(du.rho_star,   _sm_rho_tmp,   smooth_cache, du.rho_star)
    use_theta_smooth && smooth_dual0_fused!(du.Theta_star, _sm_theta_tmp, smooth_cache, du.Theta_star)
    return nothing
end

smoothing_cb = DiscreteCallback(
    (u, t, integrator) -> integrator.iter > 0 && false, 
    integrator -> smoothing_run(integrator); 
    save_positions = (false, false)
)

# TODO: Can revisit this to reduce allocations
const saving_list = Vector{AbstractVector{FT}}(undef, 3)
function saving_run(u, t, integrator)
    U_real = interior(Val(1), Array(hdg_1(u.U_star)), s)
    rho_real = interior(Val(2), Array(hdg_2(u.rho_star)), s)
    theta_real = interior(Val(2), Array(hdg_2(u.Theta_star)), s)
    
    saving_list[1] = U_real;
    saving_list[2] = rho_real;
    saving_list[3] = theta_real;

    send_output!(saving_list, io_stream, topo)
    if cart_rank == 0 
        @printf("  t = %.4f  (%.2f%%)\n", t, 100t / TE)
        flush(stdout)
    end
    return nothing
end

# TODO: Check if I can use an interval
save_cb = FunctionCallingCallback((u, t, integrator) -> saving_run(u, t, integrator); funcat = FT(0):SAVETIME:TE)

cb = CallbackSet(periodic_cb, smoothing_cb, save_cb)

# ── Solve ─────────────────────────────────────────────────────────────────────
cart_rank == 0 && println("Warming up RHS...")
let _du = to_device(zero(u0))
    rhs!(_du, u0, p_phys, FT(0))
end
cart_rank == 0 && println("Warmup complete. Solving...")

prob = ODEProblem(rhs!, u0, (FT(0), TE), p_phys)
t_start = MPI.API.MPI_Wtime()
sol = solve(prob, SSPRK33(); dt = DT, adaptive = false, save_everystep = false, save_start = false, save_end = false, dense = false, callback = cb)
t_end = MPI.API.MPI_Wtime()

solve_ms = (t_end - t_start) * 1e3

all_times = MPI.Gather(solve_ms, 0, cart_comm)

if cart_rank == 0
    println("Solve complete.")
    println("# --- Solve Timing ---")
    for r in 0:(length(all_times) - 1)
        @printf("  rank %3d : %10.2f ms\n", r, all_times[r + 1])
    end
    @printf("  mean     : %10.2f ms\n", sum(all_times) / length(all_times))
    @printf("  max      : %10.2f ms\n", maximum(all_times))
end

cart_rank == 0 && println("Solve complete.")

MPI.Barrier(cart_comm)
worker_to_output(SIGNAL_DONE, topo)
MPI.Barrier(cart_comm)

close!(ex_handler)
MPI.Barrier(cart_comm)
