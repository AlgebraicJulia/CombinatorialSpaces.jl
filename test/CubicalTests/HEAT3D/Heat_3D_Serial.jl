# Heat_Dual3D_Serial.jl
#
# Serial 3D heat equation on dual 0-forms (boids).
# Stripped of all MPI, topology, and output-process machinery.
#
# Physics pipeline (u = dual 0-form on boids):
#   grad_u      = dual_derivative(Val(0), s, u)        boid → quad
#   flux_dual   = -k * grad_u
#   flux_primal = inv_hodge_star(Val(2), s, flux_dual)  dual 2 → primal 2
#   div_flux    = exterior_derivative(Val(2), s, ...)   primal 2 → primal 3
#   laplacian_u = hodge_star(Val(3), s, div_flux)       primal 3 → dual 0
#
# Boundary conditions: no-flux (homogeneous Neumann).
# TODO: Replace the dual_derivative call with no_flux_dual_derivative! once
#       it is available for 3D. This will zero the flux through all boundary
#       quads and properly enforce zero-flux BCs, removing the need for any
#       periodicity. Until then, boundary quads will incorrectly see a
#       non-zero flux from the one-sided stencil — results near the boundary
#       should not be trusted.

using DiffEqCallbacks
using OrdinaryDiffEqTsit5
using Distributions
using KernelAbstractions
using CairoMakie
using Printf

const PROJECT_DIR = dirname(dirname(dirname(@__DIR__)))
include(joinpath(PROJECT_DIR, "src", "CubicalCode", "UniformMesh.jl"))
include(joinpath(PROJECT_DIR, "src", "CubicalCode", "UniformMesh3D.jl"))
include(joinpath(PROJECT_DIR, "src", "CubicalCode", "UniformKernelDEC3D.jl"))
include(joinpath(PROJECT_DIR, "src", "CubicalCode", "UniformPlotting.jl"))

# ── Problem parameters ────────────────────────────────────────────────────────

const NXB = 79
const NYB = 79
const NZB = 79
const LX  = 5.0
const LY  = 5.0
const LZ  = 5.0

const K_DIFFUSION       = 0.5
const T_START           = 0.0
const T_END             = 0.1
const DT                = 0.001
const SAVEAT            = 0.025
const PRINT_EVERY_N_STEPS = 50

const FT = Float64

const IMGDIR = joinpath(@__DIR__, "imgs", "serial")
mkpath(IMGDIR)

# ── Mesh ──────────────────────────────────────────────────────────────────────
# No halo needed — no exchange, no periodicity.

s = UniformCubicalComplex(NXB + 1, NYB + 1, NZB + 1, FT(LX), FT(LY), FT(LZ))

# ── Initial condition ─────────────────────────────────────────────────────────

u0 = zeros(FT, nboids(s))

center     = [LX / 2, LY / 2, LZ / 2]
covariance = [0.1, 0.3, 0.5]
dist       = MvNormal(center, covariance)

for rz in 1:nzbr(s), ry in 1:nybr(s), rx in 1:nxbr(s)
    i     = coord_to_boid(s, rx, ry, rz)
    dp    = real_dual_point(s, rx, ry, rz)
    u0[i] = pdf(dist, dp) * 10.0
end

mass_0 = sum(u0)
println("Initial mass: $mass_0")

# ── RHS ───────────────────────────────────────────────────────────────────────
function boundary_quads_3D(s::AbstractCubicalComplex3D)
    nxq_ = nxq(s); nyq_ = nyq(s); nzq_ = nzq(s)
    nx_  = nx(s);  ny_  = ny(s);  nz_  = nz(s)

    # ── Down / Up (Z-normal, XY quads, z = 1 and z = nz_) ────────────────
    down = Int32[coord_to_quad(s, x, y, 1,    Z_ALIGN) for x in 1:nxq_, y in 1:nyq_][:]
    up   = Int32[coord_to_quad(s, x, y, nz_,  Z_ALIGN) for x in 1:nxq_, y in 1:nyq_][:]

    # ── South / North (Y-normal, XZ quads, y = 1 and y = ny_) ────────────
    south = Int32[coord_to_quad(s, x, 1,    z, Y_ALIGN) for x in 1:nxq_, z in 1:nzq_][:]
    north = Int32[coord_to_quad(s, x, ny_,  z, Y_ALIGN) for x in 1:nxq_, z in 1:nzq_][:]

    # ── West / East (X-normal, YZ quads, x = 1 and x = nx_) ─────────────
    west  = Int32[coord_to_quad(s, 1,    y, z, X_ALIGN) for y in 1:nyq_, z in 1:nzq_][:]
    east  = Int32[coord_to_quad(s, nx_,  y, z, X_ALIGN) for y in 1:nyq_, z in 1:nzq_][:]

    return (down = down, up = up, south = south, north = north, west = west, east = east)
end

bdy = boundary_quads_3D(s)

function heat_rhs!(du, u, p, t)
    s, k = p
    # TODO: Replace dual_derivative with no_flux_dual_derivative! (3D) once
    #       implemented. That will zero flux through boundary quads and give
    #       correct homogeneous Neumann BCs without any periodic exchange.
    grad_u      = dual_derivative(Val(0), s, u)
    flux_dual   = k .* grad_u

    for idx in vcat(bdy.down, bdy.up, bdy.south, bdy.north, bdy.west, bdy.east)
        flux_dual[idx] = zero(FT)
    end

    flux_primal = inv_hodge_star(Val(2), s, flux_dual)
    div_flux    = exterior_derivative(Val(2), s, flux_primal)
    laplacian_u = hodge_star(Val(3), s, div_flux)
    return du .= laplacian_u
end

p = (s, K_DIFFUSION)

# ── Solve ─────────────────────────────────────────────────────────────────────

snapshots = Vector{Vector{FT}}()

progress_cb = FunctionCallingCallback(
    (u, t, integrator) -> begin
        step = integrator.stats.naccept
        if step % PRINT_EVERY_N_STEPS == 0
            pct = 100.0 * t / T_END
            @printf("step %d | t = %.4f (%.1f%%)\n", step, t, pct)
            flush(stdout)
        end
    end;
    func_everystep = true,
    func_start     = false,
)

save_cb = FunctionCallingCallback(
    (u, t, integrator) -> push!(snapshots, copy(u));
    funcat = collect(T_START:SAVEAT:T_END),
)

println("Solving...")
prob = ODEProblem(heat_rhs!, u0, (T_START, T_END), p)
sol  = solve(prob, Tsit5(); saveat = SAVEAT, adaptive = false, dt = DT,
             callback = CallbackSet(progress_cb, save_cb))
println("Solve complete.")

mass_f = sum(sol[end])
@printf("Final mass  : %.6f\n", mass_f)
@printf("Mass drift  : %.6f%%\n", 100.0 * (mass_f - mass_0) / mass_0)

# ── Plotting ──────────────────────────────────────────────────────────────────

z_mid = NZB ÷ 2
cr    = (0.0, maximum(u0) + eps())

for (idx, snap) in enumerate(snapshots)
    t_snap = T_START + (idx - 1) * SAVEAT
    data   = reshape(snap, NXB, NYB, NZB)

    slices = (
        (data[NXB ÷ 2, :, :], "x=$(NXB÷2)", "y", "z"),
        (data[:, NYB ÷ 2, :], "y=$(NYB÷2)", "x", "z"),
        (data[:, :, z_mid],   "z=$z_mid",   "x", "y"),
    )

    fig = Figure(; size = (600, 1800))
    for (row, (slice_data, plane_label, xlabel, ylabel)) in enumerate(slices)
        ax = Axis(fig[row, 1];
                  title  = "t=$(round(t_snap, digits=3)) | $plane_label slice",
                  xlabel = xlabel,
                  ylabel = ylabel)
        hm = heatmap!(ax, slice_data; colorrange = cr)
        Colorbar(fig[row, 2], hm)
    end

    fname = joinpath(IMGDIR, @sprintf("save%03d_t%.3f_allslices.png", idx, t_snap))
    save(fname, fig)
    println("Saved: $fname")
end