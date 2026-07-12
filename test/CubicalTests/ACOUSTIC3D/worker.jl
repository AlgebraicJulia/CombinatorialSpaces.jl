# kelvin_helmholtz_3d_mpi/worker.jl
using OrdinaryDiffEqSSPRK
using DiffEqCallbacks
using ComponentArrays
using Distributions
using KernelAbstractions

# ── Mesh & topology ───────────────────────────────────────────────────────────
const s           = worker_mesh(topo, (LX, LY, LZ); halo = HALO)
const cart_comm   = topo.cart_comm
const cart_rank   = topo.cart_rank
const cart_coords = MPI.Cart_coords(cart_comm)

if USE_AMDGPU
    device_id = cart_rank
    AMDGPU.device!(AMDGPU.devices()[device_id + 1])
    dev = AMDGPU.device()
    println("Rank $cart_rank | device: $(AMDGPU.device_id(dev))")
end

# ── to_device ─────────────────────────────────────────────────────────────────
function to_device(arr::AbstractVector{T}) where T
    USE_AMDGPU && return AMDGPU.ROCVector{T}(arr)
    return arr
end

function to_device(ca::ComponentVector)
    return ComponentArray(map(to_device, NamedTuple(ca)))
end

cart_rank == 0 && println("MPI worker grid: $(w_dims[1])×$(w_dims[2])×$(w_dims[3])")
println("Worker $cart_rank | coords=$cart_coords | mesh=$(nxr(s))×$(nyr(s))×$(nzr(s)) real boids")
flush(stdout)

# ── DEC operators (3D — no cache yet, dispatch directly on s) ─────────────────
# NOTE: UniformDECCache for 3D is not yet implemented; all operators take s directly.
const d0 = x -> exterior_derivative(Val(0), s, x)   # verts  → edges
const d1 = x -> exterior_derivative(Val(1), s, x)   # edges  → quads
const d2 = x -> exterior_derivative(Val(2), s, x)   # quads  → boids

const dd0 = x -> dual_derivative(Val(0), s, x)      # boids  → quads (dual 0→1)
const dd1 = x -> dual_derivative(Val(1), s, x)      # quads  → edges (dual 1→2)
const dd2 = x -> dual_derivative(Val(2), s, x)      # edges  → verts (dual 2→3)

# Primal Hodge stars
const hdg_0 = x -> hodge_star(Val(0), s, x)   # verts  → boids
const hdg_1 = x -> hodge_star(Val(1), s, x)   # edges  → quads
const hdg_2 = x -> hodge_star(Val(2), s, x)   # quads  → edges  (U_star → U, i.e. primal 2-form → dual 1-form)
const hdg_3 = x -> hodge_star(Val(3), s, x)   # boids  → verts

# Inverse Hodge stars
const inv_hdg_0 = x -> inv_hodge_star(Val(0), s, x)
const inv_hdg_1 = x -> inv_hodge_star(Val(1), s, x)
const inv_hdg_2 = x -> inv_hodge_star(Val(2), s, x)  # edges  → quads  (dual 1-form → primal 2-form)
const inv_hdg_3 = x -> inv_hodge_star(Val(3), s, x)  # verts  → boids

# Wedge products
const wdg_11     = (a, b) -> wedge_product(Val(1), Val(1), s, a, b)   # edge ∧ edge → quad
const wdg_12     = (a, b) -> wedge_product(Val(1), Val(2), s, a, b)   # edge ∧ quad → boid
const wdg_dd_01  = (f, a) -> wedge_product_dd(Val(0), Val(1), s, f, a) # dual-0 ∧ dual-1 → dual-1

const dcd_1 = x -> hdg_3(d2(inv_hdg_2(x))) # This is div
const dcd_2 = x-> hdg_2(d1(inv_hdg_1(x))) # This is curl

const dlap_0 = x -> dcd_1(dd0(x)) # This is scalar laplacian

# TODO: Should line up with the vector laplacian
# First term is gradient of divergence, second is curl of curl
const dlap_1 = x -> dd0(dcd_1(x)) .- dcd_2(dd1(x))

const interp_dp_1 = x -> interpolate_dp(Val(1), s, x)

# ── Physics ───────────────────────────────────────────────────────────────────
const p_phys = (mu = FT(1) / RE, kappa = FT(1) / (RE * PR))

const P₀    = FT(1e5)
const Cₚ    = FT(1006)
const R_gas = FT(287)

function pressure(Theta::AbstractVector{FT}) where {FT}
    R_Cₚ = R_gas / Cₚ
    return (Theta .* R_gas .* (P₀ .^ -R_Cₚ)) .^ (FT(1) / (FT(1) - R_Cₚ))
end

# ── ExchangeHandler ───────────────────────────────────────────────────────────
# U_star is a primal 2-form → lives on Quads.
# rho_star and Theta_star are dual 0-forms (primal 3-forms) → live on Boids.
const ex_stream  = DataStream(Datum[
    Datum{Quad,3}("U_star",     "fields", FT),
    Datum{Boid,3}("rho_star",   "fields", FT),
    Datum{Boid,3}("Theta_star", "fields", FT),
])
const ex_handler = ExchangeHandler(ex_stream, topo, s; backend = USE_AMDGPU ? ROCBackend() : CPU())

# ── IO stream ─────────────────────────────────────────────────────────────────
const io_stream = DataStream(Datum[
    Datum{Boid,3}("rho",   "fields", FT),
    Datum{Boid,3}("Theta", "fields", FT),
    Datum{Quad,3}("U",     "fields", FT),   # dual 1-form, quad-indexed
])

# ── Initial conditions ────────────────────────────────────────────────────────
Theta_dist    = MvNormal([LX/2, LY/2, LZ/2], [0.1, 0.1, 0.1])
Theta_perturb = zeros(FT, nboids(s))
for rz in 1:nzbr(s), ry in 1:nybr(s), rx in 1:nxbr(s)
    b  = coord_to_boid(s, rx + hx(s), ry + hy(s), rz + hz(s))
    dp = real_dual_point(s, rx, ry, rz)
    Theta_perturb[b] = pdf(Theta_dist, [dp[1], dp[2], dp[3]]) * 0.1
end

# U_star is a primal 2-form: zero initial velocity → nquads(s) zeros
U_star_0     = to_device(zeros(FT, nquads(s)))
rho_star_0   = to_device(inv_hdg_3(ones(FT, nboids(s))))
Theta_star_0 = to_device(inv_hdg_3(fill(FT(300), nboids(s))))

const u0 = ComponentVector(; U_star = U_star_0, rho_star = rho_star_0, Theta_star = Theta_star_0)

MPI.Barrier(cart_comm)

# ── No-op BC hooks ────────────────────────────────────────────────────────────
@inline enforce_bc_U!(U::AbstractVector{FT}) where FT <: AbstractFloat = U

# ── RHS ───────────────────────────────────────────────────────────────────────
function momentum_conservation(u, p)
    U     = hdg_2(u.U_star)
    rho   = hdg_3(u.rho_star)
    Theta = hdg_3(u.Theta_star)

    u = wdg_dd_01(FT(1) ./ rho, U)
    v = interp_dp_1(u)

    div_term = wdg_dd_01(dcd_1(u), U)

    adv_term = dd0(hdg_3(wdg_12(v, inv_hdg_2(U)))) + 
               hdg_2(wdg_11(v, inv_hdg_1(dd1(U)))) # TODO: This needs a closure term

    energy = FT(0.5) .* wdg_dd_01(rho, dd0(hdg_3(wdg_12(v, inv_hdg_2(u)))))

    diff_p = dd0(pressure(Theta))

    # TODO: This needs a closure term
    viscous = p.mu * dlap_1(u)

    result = inv_hdg_2(.-div_term .-adv_term .+ energy .- diff_p .+ viscous)
    enforce_bc_U!(result)
    return result
end

function potential_temperature_continuity(u, p)
    U     = hdg_2(u.U_star)
    rho   = hdg_3(u.rho_star)
    Theta = hdg_3(u.Theta_star)

    u = wdg_dd_01(FT(1) ./ rho, U)
    v = interp_dp_1(u)
    
    theta = Theta ./ rho

    creation  = Theta .* dcd_1(u)

    advection = hdg_3(wdg_12(v, inv_hdg_2(dd0(Theta))))

    # diffusion of specific theta
    diffusion = p.kappa * dlap_0(theta)

    return inv_hdg_3(.-creation .-advection .+ diffusion)
end

function rhs!(du, u, p, t)
    du.U_star     .= momentum_conservation(u, p)
    du.Theta_star .= potential_temperature_continuity(u, p)
    du.rho_star   .= -d2(u.U_star)
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

save_cb = FunctionCallingCallback((u, t, integrator) -> begin
    U_real     = interior(Val(2), Array(hdg_2(u.U_star)), s)[:]
    rho_real   = interior(Val(3), Array(hdg_3(u.rho_star)), s)[:]
    theta_real = interior(Val(3), Array(hdg_3(u.Theta_star)), s)[:]
    send_output!([rho_real, theta_real, U_real], io_stream, topo)
    cart_rank == 0 && @printf("  t = %.4f  (%.1f%%)\n", t, 100t / TE)
    flush(stdout)
end; funcat = collect(FT(0):SAVETIME:TE))

cb = CallbackSet(periodic_cb, save_cb)

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