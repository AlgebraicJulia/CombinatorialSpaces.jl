###############################################################################
#  track_rti_fronts_hdf5.jl
###############################################################################

using HDF5
using CairoMakie
using Printf

const SIM_NAME   = ARGS[1]
const SIM_NUM    = length(ARGS) >= 2 ? ARGS[2] : ""
const N_STEP     = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
const T_CUTOFF   = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : Inf   # cut off before RTI hits wall

const LOG_DIR = "/p/work2/grauta/3dlogs/$(SIM_NAME)test/$SIM_NUM"
const OUTFILE = joinpath(LOG_DIR, "output/acoustic.h5")
const IMGDIR  = joinpath(LOG_DIR, "imgs")
mkpath(IMGDIR)

const FT = Float64

# ── Physical parameters ───────────────────────────────────────────────────────
RHO_LIGHT = 0.0
RHO_HEAVY = 0.0
h5open(OUTFILE, "r") do h5
    rho_dset = h5["fields/rho"]
    rho_init = rho_dset[1, :, :, :]

    global RHO_LIGHT = minimum(rho_init)
    global RHO_HEAVY = maximum(rho_init)
end

const RHO_MID    = (RHO_LIGHT + RHO_HEAVY) / 2
const ATWOOD     = (RHO_HEAVY - RHO_LIGHT) / (RHO_HEAVY + RHO_LIGHT)
const GROWTH_RATE = sqrt(9.81 * ATWOOD * π)

println(@sprintf("Atwood number:   A = %.6f", ATWOOD))
println(@sprintf("Linear growth rate γ = %.6f", GROWTH_RATE))

# ── Read mesh parameters from HDF5 attributes ─────────────────────────────────
# const NXB, NYB, NZB, LX, LY, LZ, SAVETIME = h5open(OUTFILE, "r") do h5
#     attrs = attributes(h5)
#     Int(attrs["NXB"][]), Int(attrs["NYB"][]), Int(attrs["NZB"][]),
#     FT(attrs["LX"][]),   FT(attrs["LY"][]),   FT(attrs["LZ"][]),
#     FT(attrs["savetime"][])
# end

const NXB = 128
const NYB = 128
const NZB = 256

const LX = 2.0
const LY = 2.0
const LZ = 4.0

const SAVETIME = 0.01

const Y_MID   = LY / 2
const Z_MID   = LZ / 2
const y_mid_i = div(NYB, 2) + 1   # index of y-centerline slice [2]

# Physical x and z coordinates of cell centres on the xz slice
const xs = [(ix - 0.5) * (LX / NXB) for ix in 1:NXB]
const zs = [(iz - 0.5) * (LZ / NZB) for iz in 1:NZB]

# Column indices closest to x = LX/2 (bubble) and x = 0 (spike) [6]
const bubble_col = argmin(abs.(xs .- (LX / 2)))
const spike_col  = argmin(abs.(xs .- 0.0))
println("Bubble column: ix = $bubble_col  x = $(xs[bubble_col])")
println("Spike  column: ix = $spike_col   x = $(xs[spike_col])")

# Initial perturbation amplitude for linear theory reference [6]
const amp0 = 1e-4   # adjust if your IC uses a different amplitude
const t0   = 0.0

# ── Tip tracking ──────────────────────────────────────────────────────────────

# xz_slice is (NXB, NZB); z is the vertical axis [2]
function find_bubble_tip(xz_slice::Matrix{FT}, col::Int)
    col_rho = xz_slice[col, :]
    light_mask = col_rho .< RHO_MID
    any(light_mask) || return 0.0
    return maximum(zs[light_mask])
end

function find_spike_tip(xz_slice::Matrix{FT}, col::Int)
    col_rho = xz_slice[col, :]
    heavy_mask = col_rho .> RHO_MID
    any(heavy_mask) || return 0.0
    return minimum(zs[heavy_mask])
end

# ── Main loop ─────────────────────────────────────────────────────────────────
times       = FT[]
bubble_dist = FT[]
spike_dist  = FT[]

const ntimes = h5open(h5 -> size(h5["fields/rho"], 1), OUTFILE, "r")

println("Processing $ntimes timesteps (stride $N_STEP, cutoff t = $T_CUTOFF)...")

h5open(OUTFILE, "r") do h5
    rho_dset = h5["fields/rho"]

    for t in 1:N_STEP:ntimes
        sim_time = (t - 1) * SAVETIME
        sim_time > T_CUTOFF && break

        # xz slice at y-centerline [2]
        xz_slice = Matrix{FT}(rho_dset[t, :, y_mid_i, :])   # (NXB, NZB)

        bd = find_bubble_tip(xz_slice, bubble_col)
        sd = find_spike_tip(xz_slice, spike_col)

        push!(times,       sim_time)
        push!(bubble_dist, bd)
        push!(spike_dist,  sd)

        t % (25 * N_STEP) == 1 &&
            println(@sprintf("  t = %.4f  bubble = %.4f  spike = %.4f", sim_time, bd, sd))
    end
end

println("Processed $(length(times)) frames.")

separation = bubble_dist .- spike_dist

# ── Linear theory reference ───────────────────────────────────────────────────
t_ref_end = min(T_CUTOFF, maximum(times))
t_ref     = range(t0, t_ref_end; length = 500)
E_ref     = Z_MID .+ amp0 .* exp.(GROWTH_RATE .* (t_ref .- t0))   # bubble tip
E_ref_sp  = Z_MID .- amp0 .* exp.(GROWTH_RATE .* (t_ref .- t0))   # spike tip

# ── Plot ──────────────────────────────────────────────────────────────────────
x_major = collect(0.0:1.0:ceil(maximum(times)))
x_minor = collect(0.0:0.5:ceil(maximum(times)))
y_major = collect(0.0:1.0:ceil(LZ))
y_minor = collect(0.0:0.5:ceil(LZ))

fig = Figure(size = (900, 550))
ax  = Axis(fig[1, 1];
    xlabel             = "t",
    ylabel             = "z",
    title              = @sprintf("RTI Front Tracking  |  A = %.4f  |  γ = %.4f", ATWOOD, GROWTH_RATE),
    xticks             = x_major,
    xtickformat        = values -> [@sprintf("%.0f", v) for v in values],
    xminorticks        = x_minor,
    xminorticksvisible = true,
    xminorgridvisible  = true,
    yticks             = y_major,
    ytickformat        = values -> [@sprintf("%.0f", v) for v in values],
    yminorticks        = y_minor,
    yminorticksvisible = true,
    yminorgridvisible  = true,
    limits             = (0, T_CUTOFF == Inf ? maximum(times) : T_CUTOFF, 0, LZ),
)

lines!(ax, times, bubble_dist;
    label     = "Bubble (light rising, x = LX/2)",
    color     = :steelblue,
    linewidth = 2)
lines!(ax, times, spike_dist;
    label     = "Spike (heavy falling, x = 0)",
    color     = :tomato,
    linewidth = 2)
lines!(ax, times, separation;
    label     = "Bubble + spike separation",
    color     = :forestgreen,
    linewidth = 2,
    linestyle = :dash)
lines!(ax, collect(t_ref), E_ref;
    label     = @sprintf("Linear theory (bubble): exp(γ(t-t₀)),  γ = %.3f", GROWTH_RATE),
    color     = :steelblue,
    linewidth = 2,
    linestyle = :dot)
lines!(ax, collect(t_ref), E_ref_sp;
    label     = @sprintf("Linear theory (spike):  exp(γ(t-t₀)),  γ = %.3f", GROWTH_RATE),
    color     = :tomato,
    linewidth = 2,
    linestyle = :dot)

axislegend(ax; position = :lt)

outfile = joinpath(IMGDIR, "rti_front_tracking.png")
save(outfile, fig)
println("Saved: $outfile")