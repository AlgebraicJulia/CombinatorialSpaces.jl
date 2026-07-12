# kelvin_helmholtz_mpi/kh_worker.jl
using OrdinaryDiffEqSSPRK
using DiffEqCallbacks
using ComponentArrays
using Distributions
using KernelAbstractions

# ── Mesh & topology ───────────────────────────────────────────────────────────
const s = worker_mesh(topo, (LX, LY); halo = HALO)
const cart_comm = topo.cart_comm
const cart_rank = topo.cart_rank
const cart_coords = MPI.Cart_coords(cart_comm)

if USE_AMDGPU
    devices = AMDGPU.devices()
    ndevices = length(devices)

    local_rank = MPI.Comm_rank(topo.local_comm)
    gpu_id = (local_rank % ndevices) + 1
    AMDGPU.device!(devices[gpu_id])

    node_name = MPI.Get_processor_name()
    mapping_str = "Rank $rank -> Node: $node_name, GPU: $gpu_id"
    all_mappings = MPI.gather(mapping_str, topo.cart_comm; root=0)
    if rank == 0
        println("=========================================================")
        println("MPI Topology & GPU Assignment:")
        for m in all_mappings
            println("   $m")
        end
        println("=========================================================")
    end
end

# ── to_device ────────────────────────────────────────────────────────────────
function to_device(arr::AbstractVector{T}) where T
    USE_AMDGPU && return AMDGPU.ROCVector{T}(arr)
    return arr
end

function to_device(ca::ComponentVector)
    return ComponentArray(map(to_device, NamedTuple(ca)))
end

cart_rank == 0 && println("MPI worker grid: $(w_dims[1])×$(w_dims[2])")
println("Worker $cart_rank | coords=$cart_coords | mesh=$(nxr(s))×$(nyr(s)) real quads")
flush(stdout)

const cache       = Adapt.adapt(USE_AMDGPU ? ROCBackend() : CPU(), UniformDECCache(s))

# # TODO: Test this and move to kernels
# function build_real_emask(s::UniformCubicalComplex2D)
#     ne_  = ne(s)
#     nx_  = nx(s);  ny_  = ny(s)
#     hx_  = hx(s);  hy_  = hy(s)

#     # Real domain vertex extents [1]:
#     x_lo = hx_ + 1;      x_hi = nx_ - hx_
#     y_lo = hy_ + 1;      y_hi = ny_ - hy_

#     real_emask = Vector{Int8}(undef, ne_)

#     for e in 1:ne_
#         x, y, align = edge_to_coord(s, e)
#         if align == X_ALIGN
#             # X-edge: positive quad is above (y), negative quad is below (y-1)
#             has_pos = (y  <= y_hi - 1) && (y  >= y_lo)
#             has_neg = (y  >  y_lo)     && (y  <= y_hi)
#         else  # Y_ALIGN
#             # Y-edge: positive quad is left (x-1), negative quad is right (x)
#             has_pos = (x  >  x_lo)     && (x  <= x_hi)
#             has_neg = (x  <= x_hi - 1) && (x  >= x_lo)
#         end
#         real_emask[e] = Int8(has_pos) | (Int8(has_neg) << 1)
#     end

#     return real_emask
# end

# @kernel function kernel_no_flux_dd0_real!(res, @Const(dd0_qp), @Const(dd0_qn),
#                                           @Const(dd0_emask), @Const(real_emask),
#                                           @Const(f))
#     e = @index(Global)
#     @inbounds begin
#         cache_mask = dd0_emask[e]
#         real_mask  = real_emask[e]

#         # Interior in both the mesh topology AND the real-boundary sense
#         cache_interior = Int8(cache_mask & Int8(1)) & Int8((cache_mask >> Int8(1)) & Int8(1))
#         real_interior  = Int8(real_mask  & Int8(1)) & Int8((real_mask  >> Int8(1)) & Int8(1))
#         interior       = cache_interior | real_interior

#         z   = zero(eltype(f))
#         pos = ifelse(Bool(interior), f[dd0_qp[e]], z)
#         neg = ifelse(Bool(interior), f[dd0_qn[e]], z)
#         res[e] = pos - neg
#     end
# end

# function no_flux_dd0_real!(res, ::Val{0}, cache::UniformDECCache,
#                            real_emask::AbstractVector{Int8}, f)
#     backend = get_backend(f)
#     kernel_no_flux_dd0_real!(backend)(res, cache.dd0_qp, cache.dd0_qn,
#                                       cache.dd0_emask, real_emask, f;
#                                       ndrange = cache.ne_)
#     return res
# end

# function no_flux_dd0_real(::Val{0}, cache::UniformDECCache,
#                           real_emask::AbstractVector{Int8},
#                           f::AbstractVector{FT}) where FT
#     backend = get_backend(f)
#     res = KernelAbstractions.zeros(backend, FT, cache.ne_)
#     return no_flux_dd0_real!(res, Val(0), cache, real_emask, f)
# end

# real_emask = build_real_emask(s)

const d1          = x -> exterior_derivative(Val(1), cache, x)
const dd0         = x -> no_flux_dual_derivative!(Val(0), cache, x)
const dd1         = x -> dual_derivative(Val(1), cache, x)

const hdg_1       = x -> hodge_star(Val(1), cache, x)
const hdg_2       = x -> hodge_star(Val(2), cache, x)

const inv_hdg_0   = x -> inv_hodge_star(Val(0), cache, x)
const inv_hdg_1   = x -> inv_hodge_star(Val(1), cache, x)
const inv_hdg_2   = x -> inv_hodge_star(Val(2), cache, x)

const d_beta      = x -> d_beta_mul(cache, x)

const wdg_01      = (f, a) -> wedge_product(Val(0), Val(1), cache, f, a)
const wdg_11      = (a, b) -> wedge_product(Val(1), Val(1), cache, a, b)
const wdg_dd_01   = (f, a) -> wedge_product_dd(Val(0), Val(1), cache, f, a)

const dcd_1       = x -> dual_codifferential(Val(1), cache, x)
const dcd_2       = x -> dual_codifferential(Val(2), cache, x)

const dlap_0      = x -> dcd_1(dd0(x))
const dlap_1      = x -> dd0(dcd_1(x)) + dcd_2(dd1(x))
const dlap_1_v    = x -> dcd_2(d_beta(x))

const interp_dp_1 = x -> interpolate_dp(Val(1), cache, x)

rho_c = FT(CONFIG["Smoothing"]["rho_smooth_constant"])
theta_c = FT(CONFIG["Smoothing"]["theta_smooth_constant"])
const rho_smooth_cache = Adapt.adapt(USE_AMDGPU ? ROCBackend() : CPU(), SmoothingCache(s, rho_c))
const theta_smooth_cache = Adapt.adapt(USE_AMDGPU ? ROCBackend() : CPU(), SmoothingCache(s, theta_c))

# ── Physics ───────────────────────────────────────────────────────────────────
const p_phys = (mu = FT(1) / RE, kappa = FT(1) / (RE * PR))

# Minimal pressure law needed by momentum_conservation [10]
const P₀ = FT(1e5)
const Cₚ = FT(1006)
const R_gas = FT(287)

function pressure(Theta::AbstractVector{FT}) where {FT}
    R_Cₚ = R_gas / Cₚ
    return (Theta .* R_gas .* (P₀ .^ -R_Cₚ)) .^ (FT(1) / (FT(1) - R_Cₚ))
end

# ── ExchangeHandler ───────────────────────────────────────────────────────────
const ex_stream = DataStream(Datum[Datum{Edge,2}("U_star", "fields", FT), Datum{Quad,2}("rho_star", "fields", FT), Datum{Quad,2}("Theta_star", "fields", FT)])
const ex_handler = ExchangeHandler(ex_stream, topo, s; backend = USE_AMDGPU ? ROCBackend() : CPU())

# ── IO stream (real-cell data only, sent to output processes) ─────────────────
# TODO: Add momentum data
const io_stream = DataStream(Datum[
    Datum{Quad,2}("rho", "fields", FT), 
    Datum{Quad,2}("Theta", "fields", FT),
    Datum{Edge,2}("U", "fields", FT),
    ])

# ── Initial conditions (San & Kara 2015) ─────────────────────────────────────
const inv_hdg_2_mat = inv_hodge_star(Val(2), s)   # matrix backend for IC only

# TODO: Should streamline this process to avoid future errors
Theta_dist = MvNormal([LX/2, LY/2], [0.1, 0.1])
Theta_perturb = zeros(FT, nquads(s))
for ry in 1:nyqr(s), rx in 1:nxqr(s)
    q = coord_to_quad(s, rx + hx(s), ry + hy(s))
    dp = real_dual_point(s, rx, ry)
    Theta_perturb[q] = pdf(Theta_dist, [dp[1], dp[2]]) * 0.5
end

U_star_0 = to_device(zeros(FT, ne(s)))
rho_star_0 = to_device(inv_hdg_2_mat * ones(FT, nquads(s)))
Theta_star_0 = to_device(inv_hdg_2_mat * (fill(FT(300), nquads(s)) .+ Theta_perturb))

const u0 = ComponentVector(; U_star = U_star_0, rho_star = rho_star_0, Theta_star = Theta_star_0)

MPI.Barrier(cart_comm)

@inline function enforce_bc_U!(U::AbstractVector{FT}) where {FT}
    return U
end

@inline function enforce_bc_v!(v::AbstractVector{FT}) where {FT}
    return v
end

@inline function enforce_bc_V!(V::AbstractVector{FT}) where {FT}
    return V
end

# ── RHS ───────────────────────────────────────────────────────────────────────
function momentum_conservation(u, p)
    U   = hdg_1(u.U_star)
    rho = hdg_2(u.rho_star)
    Theta = hdg_2(u.Theta_star)

    u = wdg_dd_01(FT(1) ./ rho, U)
    v   = interp_dp_1(u)
    V   = interp_dp_1(U)

    enforce_bc_v!(v); enforce_bc_V!(V)

    div_term = wdg_dd_01(dcd_1(u), U)

    L_term   = dd0(hdg_2(wdg_11(v, inv_hdg_1(U)))) +
               hdg_1(wdg_01(inv_hdg_0(dd1(U) + d_beta(V)), v))

    energy   = FT(0.5) .* wdg_dd_01(rho, dd0(hdg_2(wdg_11(v, inv_hdg_1(u)))))

    diff_p   = dd0(pressure(Theta))

    viscous  = p.mu * (dlap_1(u) + dlap_1_v(v))

    # TODO: Will also need to body forces later 

    result = -inv_hdg_1(.-div_term .- L_term .+ energy .- diff_p .+ viscous)

    enforce_bc_U!(result)
    return result
end

function potential_temperature_continuity(u, p)
    U     = hdg_1(u.U_star)
    rho   = hdg_2(u.rho_star)
    Theta = hdg_2(u.Theta_star)

    u   = wdg_dd_01(FT(1) ./ rho, U)
    v     = interp_dp_1(u)
    theta = Theta ./ rho

    creation  = Theta .* dcd_1(u)

    advection = hdg_2(wdg_11(v, inv_hdg_1(dd0(Theta))))

    diffusion = p.kappa * dlap_0(theta)

    return inv_hdg_2(.-creation .- advection .+ diffusion)
end

function rhs!(du, u, p, t)
    du.U_star     .= momentum_conservation(u, p)
    du.Theta_star .= potential_temperature_continuity(u, p)
    du.rho_star   .= d1(u.U_star)
    return nothing
end

# ── Callbacks ─────────────────────────────────────────────────────────────────
const SAVEAT_STEPS = max(1, round(Int, SAVETIME / DT))
const PRINT_EVERY = 500

const N = 1
periodic_cb = DiscreteCallback(
    (u, t, integrator) -> integrator.iter > 0 && integrator.iter % N == 0,
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

smoothing_cb = DiscreteCallback(
    (u, t, integrator) -> integrator.iter > 0 && false, 
    integrator -> begin
        integrator.u.rho_star .= smooth_dual0_fused(rho_smooth_cache, integrator.u.rho_star)
        integrator.u.Theta_star .= smooth_dual0_fused(theta_smooth_cache, integrator.u.Theta_star)
        return nothing
    end; 
    save_positions = (false, false)
)

save_cb = FunctionCallingCallback((u, t, integrator) -> begin
    U_real = interior(Val(1), Array(hdg_1(u.U_star)), s)[:]
    rho_real = interior(Val(2), Array(hdg_2(u.rho_star)), s)[:]
    theta_real = interior(Val(2), Array(hdg_2(u.Theta_star)), s)[:]
    send_output!([rho_real, theta_real, U_real], io_stream, topo)
    cart_rank == 0 && @printf("  t = %.4f  (%.1f%%)\n", t, 100t / TE)
    flush(stdout)
end; funcat = collect(FT(0):SAVETIME:TE))

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
