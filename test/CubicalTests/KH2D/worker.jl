# kelvin_helmholtz_mpi/kh_worker.jl
using OrdinaryDiffEqSSPRK
using DiffEqCallbacks
using ComponentArrays

# ── Mesh & topology ───────────────────────────────────────────────────────────
const s = worker_mesh(topo, (LX, LY); halo = HALO)
const cart_comm = topo.cart_comm
const cart_rank = topo.cart_rank
const cart_coords = MPI.Cart_coords(cart_comm)

cart_rank == 0 && println("KH MPI worker grid: $(w_dims[1])×$(w_dims[2])")
println("Worker $cart_rank | coords=$cart_coords | mesh=$(nxr(s))×$(nyr(s)) real quads")

# ── DEC operators ─────────────────────────────────────────────────────────────
# ── DEC operators ─────────────────────────────────────────────────────────────
const cache = UniformDECCache(s)

const d1          = x -> exterior_derivative(Val(1), cache, x)
const dd0         = x -> no_flux_dual_derivative(Val(0), cache, x)
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

const rho_smooth_cache = SmoothingCache(s, FT(CONFIG["Smoothing"]["rho_smooth_constant"]))
const theta_smooth_cache = SmoothingCache(s, FT(CONFIG["Smoothing"]["theta_smooth_constant"]))

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
const ex_handler = ExchangeHandler(ex_stream, topo, s)

# ── IO stream (real-cell data only, sent to output processes) ─────────────────
# TODO: Add momentum data
const io_stream = DataStream(Datum[Datum{Quad,2}("rho", "fields", FT), Datum{Quad,2}("Theta", "fields", FT)])

# ── Initial conditions (San & Kara 2015) ─────────────────────────────────────
const alpha = FT(50)

function kh_density(y::FT) where {FT}
    return FT(1) + FT(0.5) * tanh(alpha * (y - FT(0.25))) - FT(0.5) * tanh(alpha * (y - FT(0.75)))
end

const ps = points(s)
const dps = dual_points(s)

const inv_hdg_2_mat = inv_hodge_star(Val(2), s)   # matrix backend for IC only

rho_star_0 = inv_hdg_2_mat * map(dps) do (x, y)
    return kh_density(FT(y))
end

U_star_0 = zeros(FT, ne(s))
for ex in 1:nxedges(s)
    v1 = src(s, ex)
    v2 = tgt(s, ex)
    xc = FT(0.5) * (ps[v1][1] + ps[v2][1])
    yc = FT(ps[v1][2])
    U_star_0[ex] = FT(0.01) * sin(FT(2π) * xc) * edge_len(s, X_ALIGN) * kh_density(yc)
end
for ey in (nxedges(s) + 1):ne(s)
    v1 = src(s, ey)
    v2 = tgt(s, ey)
    yc = FT(0.5) * (ps[v1][2] + ps[v2][2])
    flow = (FT(0.25) <= yc <= FT(0.75)) ? FT(0.5) : FT(-0.5)
    U_star_0[ey] = flow * edge_len(s, Y_ALIGN) * kh_density(yc)
end

Theta_star_0 = inv_hdg_2_mat * fill(FT(300), nquads(s))

const u0 = ComponentVector(; U_star = U_star_0, rho_star = rho_star_0, Theta_star = Theta_star_0)

# ── Per-rank initial condition plots ──────────────────────────────────────────
using CairoMakie

let
    rho_real = [hdg_2(rho_star_0)[coord_to_quad(s, x + hx(s), y + hy(s))]
                for x in 1:nxqr(s), y in 1:nyqr(s)]

    local_min = minimum(rho_real)
    local_max = maximum(rho_real)
    global_min = MPI.Allreduce(local_min, min, cart_comm)
    global_max = MPI.Allreduce(local_max, max, cart_comm)

    fig = Figure(; size = (600, 500))
    ax  = CairoMakie.Axis(fig[1, 1];
               title   = "Rank $cart_rank | coords=$cart_coords | ρ IC",
               xlabel  = "x",
               ylabel  = "y")
    hm  = heatmap!(ax, rho_real;
                   colorrange = (global_min, global_max),
                   colormap   = Makie.Reverse(:oslo))
    Colorbar(fig[1, 2], hm)
    fname = joinpath(IMGDIR, @sprintf("rank%03d_coords%d-%d_rho_IC.png",
                                      cart_rank, cart_coords[1], cart_coords[2]))
    save(fname, fig)
end

MPI.Barrier(cart_comm)

# ── No-op BC hooks expected by momentum_conservation [7] ─────────────────────
@inline enforce_bc_v!(v) = v
@inline enforce_bc_V!(V) = V
@inline enforce_bc_U!(U) = U

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

const N = 5
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
    (u, t, integrator) -> integrator.iter > 0, 
    integrator -> begin
        integrator.u.rho_star .= smooth_dual0_fused(rho_smooth_cache, integrator.u.rho_star)
        integrator.u.Theta_star .= smooth_dual0_fused(theta_smooth_cache, integrator.u.Theta_star)
        return nothing
    end; 
    save_positions = (false, false)
)

save_cb = FunctionCallingCallback((u, t, integrator) -> begin
    rho_real = interior(Val(2), hdg_2(u.rho_star), s)[:]
    theta_real = interior(Val(2), hdg_2(u.Theta_star), s)[:]
    send_output!([rho_real, theta_real], io_stream, topo)
    cart_rank == 0 && @printf("  t = %.4f  (%.1f%%)\n", t, 100t / TE)
    flush(stdout)
end; funcat = collect(FT(0):SAVETIME:TE))

cb = CallbackSet(periodic_cb, smoothing_cb, save_cb)

# ── Solve ─────────────────────────────────────────────────────────────────────
cart_rank == 0 && println("Warming up RHS...")
let _du = zero(u0)
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

worker_to_output(SIGNAL_DONE, topo)
MPI.Barrier(cart_comm)

close!(ex_handler)
MPI.Barrier(cart_comm)
