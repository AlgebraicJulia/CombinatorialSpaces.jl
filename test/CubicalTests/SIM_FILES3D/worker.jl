# kelvin_helmholtz_3d_mpi/worker.jl
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
const has_down = neighbors.down != MPI.PROC_NULL
const has_up = neighbors.up != MPI.PROC_NULL

const _halo_west = has_west ? HALO : 0
const _halo_east = has_east ? HALO : 0
const _halo_south = has_south ? HALO : 0
const _halo_north = has_north ? HALO : 0
const _halo_down = has_down ? HALO : 0
const _halo_up = has_up ? HALO : 0

const s = worker_mesh(topo, (LX, LY, LZ); halo_west = _halo_west, halo_east = _halo_east, 
    halo_south = _halo_south, halo_north = _halo_north, halo_down = _halo_down, halo_up = _halo_up)

const cart_comm   = topo.cart_comm
const cart_rank   = topo.cart_rank
const cart_coords = MPI.Cart_coords(cart_comm)

const bdry_quads = boundary_quads(s)

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

const BACKEND = use_amdgpu(USE_XPU) ? ROCBackend() : CPU()
const adv_cache = Adapt.adapt(BACKEND, AdvectionCache(WENO5(), s))

# ── to_device ─────────────────────────────────────────────────────────────────
function to_device(arr::AbstractVector{T}) where T
    use_amdgpu(XPU) && return AMDGPU.ROCVector{T}(arr)
    return arr
end

function to_device(ca::ComponentVector)
    return ComponentArray(map(to_device, NamedTuple(ca)))
end

cart_rank == 0 && println("MPI worker grid: $(w_dims[1])×$(w_dims[2])×$(w_dims[3])")
# println("Worker $cart_rank | coords=$cart_coords | mesh=$(nxr(s))×$(nyr(s))×$(nzr(s)) real boids")
cart_rank == 0 && flush(stdout)

# ── DEC operators ─────────────────
const hdg_2 = x -> hodge_star(Val(2), s, x)   # quads  → edges  (U_star → U, i.e. primal 2-form → dual 1-form)
const hdg_3 = x -> hodge_star(Val(3), s, x)   # boids  → verts

const inv_hdg_2 = x -> inv_hodge_star(Val(2), s, x)  # edges  → quads  (dual 1-form → primal 2-form)
const inv_hdg_3 = x -> inv_hodge_star(Val(3), s, x)  # verts  → boids

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
    const rho_smooth_cache = Adapt.adapt(BACKEND, SmoothingCache3D(s, rho_c))
end

if use_theta_smooth
    world_rank == 0 && println("Activated Theta smoothing: $theta_c")
    const theta_smooth_cache = Adapt.adapt(BACKEND, SmoothingCache3D(s, theta_c))
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

function pressure(Theta::AbstractVector{FT}) where {FT}
    R_Cₚ = R_gas / Cₚ
    return (Theta .* R_gas .* (P₀ .^ -R_Cₚ)) .^ (FT(1) / (FT(1) - R_Cₚ))
end

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
    const g_dual_cpu = zeros(FT, nquads(s))

    for q in 1:nxyquads(s)
        x, y, z, align = quad_to_coord(s, q)
        g_dual_cpu[q] = gₐ * dual_edge_len(s, x, y, z, align)
    end
    
    const g_dual = to_device(g_dual_cpu)
end

# ── ExchangeHandler ───────────────────────────────────────────────────────────
# U_star is a primal 2-form → lives on Quads.
# rho_star and Theta_star are dual 0-forms (primal 3-forms) → live on Boids.
const ex_stream  = DataStream(Datum[
    Datum{Quad,3}("U_star",     "fields", FT),
    Datum{Boid,3}("rho_star",   "fields", FT),
    Datum{Boid,3}("Theta_star", "fields", FT),
])
const ex_handler = ExchangeHandler(ex_stream, topo, s; backend = BACKEND)

# ── IO stream ─────────────────────────────────────────────────────────────────
const io_stream = DataStream(Datum[
    Datum{Boid,3}("rho",   "fields", FT),
    Datum{Boid,3}("Theta", "fields", FT),
    Datum{Quad,3}("U",     "fields", FT),   # dual 1-form, quad-indexed
])

# ── Initial conditions ────────────────────────────────────────────────────────

const boundary_mask_d_cpu = zeros(FT, nquads(s))
boundary_mask_d_cpu[bdry_quads] .= FT(1.0)
const boundary_mask_d = Adapt.adapt(BACKEND, boundary_mask_d_cpu)

# For free-slip conditions, remove component of dd1 parallel to boundary

function generate_free_slip_mask(s::UniformCubicalComplex3D)
    dd1_emask = Vector{Int8}(undef, ne(s))

    for e in 1:ne(s)
        x, y, z, align = edge_to_coord(s, e)
        (q_idx, q_valid) = edge_quads(s, x, y, z, align)

        dd1_emask[e] =
            Int8(q_valid[1]) |
            (Int8(q_valid[2]) << 1) |
            (Int8(q_valid[3]) << 2) |
            (Int8(q_valid[4]) << 3)
    end

    free_dd1_emask = similar(dd1_emask)

    for e in 1:ne(s)
        x, y, z, align = edge_to_coord(s, e)
        mask = dd1_emask[e]

        if align == X_ALIGN
            if y == 1     ; mask &= Int8(0b1011) ; end  # drop q3
            if z == 1     ; mask &= Int8(0b0111) ; end  # drop q4
            if y == ny(s) ; mask &= Int8(0b1110) ; end  # drop q1
            if z == nz(s) ; mask &= Int8(0b1101) ; end  # drop q2
        elseif align == Y_ALIGN
            if z == 1     ; mask &= Int8(0b1011) ; end  # drop q3
            if x == 1     ; mask &= Int8(0b0111) ; end  # drop q4
            if z == nz(s) ; mask &= Int8(0b1110) ; end  # drop q1
            if x == nx(s) ; mask &= Int8(0b1101) ; end  # drop q2
        else # Z_ALIGN
            if x == 1     ; mask &= Int8(0b1011) ; end  # drop q3
            if y == 1     ; mask &= Int8(0b0111) ; end  # drop q4
            if x == nx(s) ; mask &= Int8(0b1110) ; end  # drop q1
            if y == ny(s) ; mask &= Int8(0b1101) ; end  # drop q2
        end

        free_dd1_emask[e] = mask
    end

    return free_dd1_emask
end
    
free_dd1_emask = Adapt.adapt(BACKEND, generate_free_slip_mask(s))

include(joinpath(@__DIR__, "Examples", "$SIM_NAME.jl"))

const u0 = ComponentVector(; U_star = to_device(U_star_0), rho_star = to_device(rho_star_0), Theta_star = to_device(Theta_star_0))

MPI.Barrier(cart_comm)

# ── RHS ───────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "buffers.jl"))

function momentum_conservation!(result::AbstractVector{FT}, u::ComponentVector,
                                 p::PhysicalParameters{FT}) where {FT <: AbstractFloat}
    # ── hodge lifts ───────────────────────────────────────────────────────────
    hodge_star!(_mc_U,     Val(2), s, u.U_star)
    hodge_star!(_mc_rho,   Val(3), s, u.rho_star)
    hodge_star!(_mc_Theta, Val(3), s, u.Theta_star)

    # ── vel = wdg_dd_01(1/rho, U) ─────────────────────────────────────────────
    _mc_inv_rho .= FT(1) ./ _mc_rho
    wedge_product_dd!(_mc_u, Val(0), Val(1), s, _mc_inv_rho, _mc_U)

    # ── v = interp_dp_1(vel) ──────────────────────────────────────────────────
    interpolate_dp!(_mc_v, _X_vel, _Y_vel, _Z_vel, Val(1), s, _mc_u)
    enforce_bc_v!(_mc_v)

    # ── div_term = wdg_dd_01(dcd_1(vel), U) ──────────────────────────────────
    # dual_codifferential!(_mc_dcd1_vel, Val(1), s, _mc_u)
    inv_hodge_star!(_mc_tmp_1, Val(2), s, _mc_u)
    exterior_derivative!(_mc_tmp_2, Val(2), s, _mc_tmp_1)
    hodge_star!(_mc_dcd1_vel, Val(3), s, _mc_tmp_2)
    wedge_product_dd!(_mc_div_term, Val(0), Val(1), s, _mc_dcd1_vel, _mc_U)

    # ── adv_term left: dd0(hdg_3(wdg_12(v, inv_hdg_2(U)))) ───────────────────
    inv_hodge_star!(_mc_ihs2_U, Val(2), s, _mc_U)
    # TODO: Find a better way to switch WENO
    wedge_product!(_mc_wdg12_vU, Val(1), Val(2), s, _mc_v, _mc_ihs2_U)
    # wedge_product_12!(_mc_wdg12_vU, _mc_wdg12_tmpx, _mc_wdg12_tmpy, _mc_wdg12_tmpz, WENO5(), adv_cache, _mc_v, _mc_ihs2_U)
    hodge_star!(_mc_hdg3_wdg,   Val(3), s, _mc_wdg12_vU)
    dual_derivative!(_mc_dd0_hdg3, Val(0), s, _mc_hdg3_wdg)

    # TODO: This is to zero out the fluxes, a single kernel would be nice
    _mc_dd0_hdg3 .= _mc_dd0_hdg3 .* (FT(1.0) .- boundary_mask_d) 
    
    # ── adv_term right: hdg_2(wdg_11(v, inv_hdg_1(dd1(U)))) ─────────────────
    # TODO: This is an insert for the free slip condition
    dual_derivative!(_mc_dd1_U,         Val(1), s, _mc_U)
    # free_slip_dd1!(_mc_dd1_U, s, free_dd1_emask, _mc_U)

    inv_hodge_star!(_mc_ihs1_dd1U,      Val(1), s, _mc_dd1_U)
    wedge_product!(_mc_wdg11_v,         Val(1), Val(1), s, _mc_v, _mc_ihs1_dd1U) # TODO: WENO switch
    # wedge_product_11!(_mc_wdg11_v, _mc_wdg11_tmpa, _mc_wdg11_tmpb, WENO5(), adv_cache, _mc_v, _mc_ihs1_dd1U)
    hodge_star!(_mc_hdg2_wdg,           Val(2), s, _mc_wdg11_v)

    _mc_adv_term .= _mc_dd0_hdg3 .+ _mc_hdg2_wdg

    # ── energy = 0.5 * wdg_dd_01(rho, dd0(hdg_3(wdg_12(v, inv_hdg_2(vel))))) ─
    inv_hodge_star!(_mc_ihs2_vel,  Val(2), s, _mc_u)
    wedge_product!(_mc_wdg12_vvel, Val(1), Val(2), s, _mc_v, _mc_ihs2_vel) # TODO: WENO switch
    # wedge_product_12!(_mc_wdg12_vvel, _mc_wdg12v_tmpx, _mc_wdg12v_tmpy, _mc_wdg12v_tmpz, WENO5(), adv_cache, _mc_v, _mc_ihs2_vel)
    hodge_star!(_mc_hdg3_vvel,     Val(3), s, _mc_wdg12_vvel)
    dual_derivative!(_mc_dd0_hdg3v, Val(0), s, _mc_hdg3_vvel)
    _mc_dd0_hdg3v .= _mc_dd0_hdg3v .* (FT(1.0) .- boundary_mask_d) 

    wedge_product_dd!(_mc_energy,  Val(0), Val(1), s, _mc_rho, _mc_dd0_hdg3v)
    _mc_energy .*= FT(0.5)

    # ── diff_p = dd0(pressure(Theta)) ─────────────────────────────────────────
    pressure!(_mc_pressure, _mc_Theta)
    dual_derivative!(_mc_diff_p, Val(0), s, _mc_pressure)
    _mc_diff_p .= _mc_diff_p .* (FT(1.0) .- boundary_mask_d)

    # ── viscous = mu * dlap_1(vel) ────────────────────────────────────────────
    # TODO: Free slip here
    dual_derivative!(_mc_dlap1_tmp1, Val(1), s, _mc_u)
    # free_slip_dd1!(_mc_dlap1_tmp1, s, free_dd1_emask, _mc_u)

    # dual_codifferential!(_mc_viscous_1, Val(2), s, _mc_dlap1_tmp1)
    inv_hodge_star!(_mc_tmp_3, Val(1), s, _mc_dlap1_tmp1)
    exterior_derivative!(_mc_tmp_4, Val(1), s, _mc_tmp_3)
    hodge_star!(_mc_viscous_1, Val(2), s, _mc_tmp_4)

    # dual_codifferential!(_mc_dlap1_tmp2, Val(1), s, _mc_u)
    dual_derivative!(_mc_viscous_2, Val(0), s, _mc_dcd1_vel) # Taken from div term
    _mc_viscous_2 .= _mc_viscous_2 .* (FT(1.0) .- boundary_mask_d) 

    _mc_viscous .= p.mu .* (_mc_viscous_2 .- _mc_viscous_1)

    # ── assemble and apply inv_hdg_2 ──────────────────────────────────────────
    _mc_sum_terms .= .-_mc_div_term .- _mc_adv_term .+ _mc_energy .- _mc_diff_p .+ _mc_viscous

    if GRAVITY
        wedge_product_dd!(_mc_gravity, Val(0), Val(1), s, _mc_rho, g_dual)
        _mc_sum_terms .+= _mc_gravity
    end
    
    inv_hodge_star!(result, Val(2), s, _mc_sum_terms)
    enforce_bc_U!(result)
    return result
end

function potential_temperature_continuity!(result::AbstractVector{FT}, u::ComponentVector,
                                            p::PhysicalParameters{FT}) where {FT <: AbstractFloat}
    # ── hodge lifts ───────────────────────────────────────────────────────────
    hodge_star!(_pt_U,     Val(2), s, u.U_star)
    hodge_star!(_pt_rho,   Val(3), s, u.rho_star)
    hodge_star!(_pt_Theta, Val(3), s, u.Theta_star)

    # ── vel = wdg_dd_01(1/rho, U) ─────────────────────────────────────────────
    _pt_inv_rho .= FT(1) ./ _pt_rho
    wedge_product_dd!(_pt_u, Val(0), Val(1), s, _pt_inv_rho, _pt_U)

    # ── v = interp_dp_1(vel) ──────────────────────────────────────────────────
    interpolate_dp!(_pt_v, _X_vel, _Y_vel, _Z_vel, Val(1), s, _pt_u)

    # ── theta = Theta / rho ───────────────────────────────────────────────────
    _pt_theta .= _pt_Theta ./ _pt_rho

    # ── creation = Theta .* dcd_1(vel) ───────────────────────────────────────
    # dual_codifferential!(_pt_dcd1_vel, Val(1), s, _pt_u)
    inv_hodge_star!(_pt_tmp_5, Val(2), s, _pt_u)
    exterior_derivative!(_pt_tmp_6, Val(2), s, _pt_tmp_5)
    hodge_star!(_pt_dcd1_vel, Val(3), s, _pt_tmp_6)

    _pt_creation .= _pt_Theta .* _pt_dcd1_vel

    # ── advection = hdg_3(wdg_12(v, inv_hdg_2(dd0(Theta)))) ─────────────────
    dual_derivative!(_pt_dd0_Theta,  Val(0), s, _pt_Theta)
    _pt_dd0_Theta .= _pt_dd0_Theta .* (FT(1.0) .- boundary_mask_d)

    inv_hodge_star!(_pt_ihs2_dd0T,   Val(2), s, _pt_dd0_Theta)
    wedge_product!(_pt_wdg12_vT,     Val(1), Val(2), s, _pt_v, _pt_ihs2_dd0T) # TODO: WENO switch
    # wedge_product_12!(_pt_wdg12_vT, _pt_wdg12_tmpx, _pt_wdg12_tmpy, _pt_wdg12_tmpz, WENO5(), adv_cache, _pt_v, _pt_ihs2_dd0T)
    hodge_star!(_pt_advection,       Val(3), s, _pt_wdg12_vT)

    # ── diffusion = alpha * dlap_0(theta) ─────────────────────────────────────
    dual_derivative!(_pt_dd0_theta, Val(0), s, _pt_theta)   # dd0: nboids→nquads
    _pt_dd0_theta .= _pt_dd0_theta .* (FT(1.0) .- boundary_mask_d)

    # dual_codifferential!(_pt_diffusion, Val(1), s, _pt_dd0_theta)  # dcd_1: nquads→nboids
    inv_hodge_star!(_pt_tmp_7, Val(2), s, _pt_dd0_theta)
    exterior_derivative!(_pt_tmp_8, Val(2), s, _pt_tmp_7)
    hodge_star!(_pt_diffusion, Val(3), s, _pt_tmp_8)

    _pt_diffusion .*= p.alpha

    # ── assemble and apply inv_hdg_3 ──────────────────────────────────────────
    _pt_sum .= .-_pt_creation .- _pt_advection .+ _pt_diffusion
    inv_hodge_star!(result, Val(3), s, _pt_sum)
    return result
end

function rhs!(du, u, p, t)
    momentum_conservation!(du.U_star, u, p)
    potential_temperature_continuity!(du.Theta_star, u, p)
    exterior_derivative!(_rhs_d2_U, Val(2), s, u.U_star)
    du.rho_star .= .-_rhs_d2_U
    return nothing
end

# ── Callbacks ─────────────────────────────────────────────────────────────────
periodic_cb = DiscreteCallback(
    (u, t, integrator) -> integrator.iter > 0,
    integrator -> begin
        u = integrator.u
        exchange!(ex_handler, (
            U_star     = u.U_star,
            rho_star   = u.rho_star,
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

save_cb = FunctionCallingCallback((u, t, integrator) -> begin
    U_real     = interior(Val(2), Array(hdg_2(u.U_star)), s)[:]
    rho_real   = interior(Val(3), Array(hdg_3(u.rho_star)), s)[:]
    theta_real = interior(Val(3), Array(hdg_3(u.Theta_star)), s)[:]
    send_output!([rho_real, theta_real, U_real], io_stream, topo)
    if cart_rank == 0
        @printf("  t = %.4f  (%.3f%%)\n", t, 100t / TE)
        flush(stdout)
    end
end; funcat = FT(0):SAVETIME:TE)

cb = CallbackSet(periodic_cb, smoothing_cb, save_cb)

# ── Solve ─────────────────────────────────────────────────────────────────────
cart_rank == 0 && println("Warming up RHS...")
let _du = to_device(zero(u0))
    rhs!(_du, u0, p_phys, FT(0))
end
cart_rank == 0 && println("Warmup complete. Solving...")

prob    = ODEProblem(rhs!, u0, (FT(0), TE), p_phys)
t_start = MPI.API.MPI_Wtime()
try
    sol     = solve(prob, SSPRK33(); dt = DT, adaptive = false,
                save_everystep = false, save_start = false,
                save_end = false, dense = false, callback = cb)
catch e
    println("Caught error: ", e)
end
t_end   = MPI.API.MPI_Wtime()

solve_ms  = (t_end - t_start)
all_times = MPI.Gather(solve_ms, 0, cart_comm)

if cart_rank == 0
    println("Solve complete.")
    println("# --- Solve Timing ---")
    for r in 0:(length(all_times) - 1)
        @printf("  rank %3d : %10.2f s\n", r, all_times[r + 1])
    end
    @printf("  mean     : %10.2f s\n", sum(all_times) / length(all_times))
    @printf("  max      : %10.2f s\n", maximum(all_times))
end

MPI.Barrier(cart_comm)
worker_to_output(SIGNAL_DONE, topo)
MPI.Barrier(cart_comm)

close!(ex_handler)
MPI.Barrier(cart_comm)