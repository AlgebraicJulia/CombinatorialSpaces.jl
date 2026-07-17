# Adv_Dual3D_Serial.jl
#
# Serial 3D constant-velocity advection on dual 0-forms (boids).
# Stripped of all MPI, topology, and output-process machinery.
#
# Physics pipeline (u = dual 0-form on boids, v = dual 1-form on quads):
#   w     = wedge_product_dd(Val(0), Val(1), s, u, v)   dual 0 ∧ dual 1 → dual 1 (quads)
#   du/dt = -dual_codifferential(Val(1), s, w)
#         = -(hs3 * d2 * ihs2) * w                      quads → boids
#
# Boundary conditions: no-flux (homogeneous Neumann).
# TODO: Zero the flux through all boundary quads in the velocity field v
#       (and in the wedge product output w) before each RHS evaluation.
#       This enforces zero-flux BCs and removes the need for periodicity.
#       Until then, boundary quads carry a spurious one-sided flux and
#       results near the boundary should not be trusted.

using OrdinaryDiffEqTsit5
using Distributions
using KernelAbstractions
using CairoMakie
using DiffEqCallbacks
using Printf

const PROJECT_DIR = dirname(dirname(dirname(@__DIR__)))
include(joinpath(PROJECT_DIR, "src", "CubicalCode", "UniformMesh.jl"))
include(joinpath(PROJECT_DIR, "src", "CubicalCode", "UniformMesh3D.jl"))
include(joinpath(PROJECT_DIR, "src", "CubicalCode", "UniformKernelDEC3D.jl"))

# ── Problem parameters ────────────────────────────────────────────────────────

const NXB = 79
const NYB = 79
const NZB = 79
const LX  = 5.0
const LY  = 5.0
const LZ  = 5.0

const V_X = 1.0
const V_Y = 0.0
const V_Z = 0.0

const T_START           = 0.0
const T_END             = 0.3 # 5.0   # one full period: LX / V_X = 5.0
const DT                = 0.001
const SAVEAT            = 0.1
const PRINT_EVERY_N_STEPS = 250

const FT = Float64

const IMGDIR = joinpath(@__DIR__, "imgs", "serial")
mkpath(IMGDIR)

# ── Mesh ──────────────────────────────────────────────────────────────────────
# No halo needed — no exchange, no periodicity.

s = UniformCubicalComplex(NXB + 1, NYB + 1, NZB + 1, FT(LX), FT(LY), FT(LZ))

# ── Constant velocity dual 1-form on quads ────────────────────────────────────
# v is a dual 1-form on quads. For uniform x-advection, only X_ALIGN (YZ) quads
# carry flux: V_X * dual_edge_len(s, x, y, z, X_ALIGN).
#
# TODO: After zeroing boundary quads (see note above), also zero v at all
#       boundary quad indices so no flux enters or leaves the domain.

v = zeros(FT, nquads(s))

for z in 1:nzb(s), y in 1:nyb(s), x in 1:nx(s)
    v[coord_to_quad(s, x, y, z, X_ALIGN)] = V_X * dual_edge_len(s, x, y, z, X_ALIGN)
end
for z in 1:nzb(s), y in 1:ny(s), x in 1:nxb(s)
    v[coord_to_quad(s, x, y, z, Y_ALIGN)] = V_Y * dual_edge_len(s, x, y, z, Y_ALIGN)
end
for z in 1:nz(s), y in 1:nyb(s), x in 1:nxb(s)
    v[coord_to_quad(s, x, y, z, Z_ALIGN)] = V_Z * dual_edge_len(s, x, y, z, Z_ALIGN)
end

# ── Initial condition ─────────────────────────────────────────────────────────

u0   = zeros(FT, nboids(s))
dist = MvNormal([LX / 2, LY / 2, LZ / 2], [0.3, 0.3, 0.3])

for rz in 1:nzbr(s), ry in 1:nybr(s), rx in 1:nxbr(s)
    i     = coord_to_boid(s, rx, ry, rz)
    dp    = real_dual_point(s, rx, ry, rz)
    u0[i] = pdf(dist, [dp[1], dp[2], dp[3]]) * 10.0
end

mass_0 = sum(u0) * boid_volume(s)
println("Initial mass: $mass_0")

# ── RHS ───────────────────────────────────────────────────────────────────────

ihs2 = f -> inv_hodge_star(Val(2), s, f)
d2   = f -> exterior_derivative(Val(2), s, f)
hs3  = f -> hodge_star(Val(3), s, f)

function advection_rhs!(du, u, p, t)
    s, v, ihs2, d2, hs3 = p
    w = wedge_product_dd(Val(0), Val(1), s, u, v)
    return du .= -(hs3(d2(ihs2(w))))
end

p = (s, v, ihs2, d2, hs3)

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
prob = ODEProblem(advection_rhs!, u0, (T_START, T_END), p)
sol  = solve(prob, Tsit5(); saveat = SAVEAT, adaptive = false, dt = DT,
             callback = CallbackSet(progress_cb, save_cb))
println("Solve complete.")

mass_f = sum(sol[end]) * boid_volume(s)
@printf("Final mass  : %.6f\n", mass_f)
@printf("Mass drift  : %.6f%%\n", 100.0 * (mass_f - mass_0) / mass_0)

# ── Plotting ──────────────────────────────────────────────────────────────────

cr = (0.0, maximum(u0) + eps())

for (idx, snap) in enumerate(snapshots)
    t_snap = T_START + (idx - 1) * SAVEAT
    data   = reshape(snap, NXB, NYB, NZB)

    slices = (
        (data[NXB ÷ 2, :, :], "x=$(NXB÷2)", "y", "z"),
        (data[:, NYB ÷ 2, :], "y=$(NYB÷2)", "x", "z"),
        (data[:, :, NZB ÷ 2], "z=$(NZB÷2)", "x", "y"),
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