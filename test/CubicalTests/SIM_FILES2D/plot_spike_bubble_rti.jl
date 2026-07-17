###############################################################################
#  track_rti_fronts_hdf5_2d.jl
###############################################################################
using HDF5
using CairoMakie
using Printf

const SIM_NUM  = ARGS[1]
const N_STEP    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
const T_CUTOFF  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : Inf

const LOG_DIR = "/p/home/grauta/git/CombinatorialSpaces.jl/test/CubicalTests/SIM_FILES2D/logs/$SIM_NUM"
const OUTFILE = joinpath(LOG_DIR, "output/savedata.h5")
const IMGDIR  = joinpath(LOG_DIR, "imgs")
mkpath(IMGDIR)

const FT = Float64

# ── Physical parameters ───────────────────────────────────────────────────────
RHO_LIGHT = 0.0
RHO_HEAVY = 0.0
h5open(OUTFILE, "r") do h5
    rho_dset = h5["fields/rho"]
    rho_init = rho_dset[1, :, :]   # (NXB, NYB)

    global RHO_LIGHT = minimum(rho_init)
    global RHO_HEAVY = maximum(rho_init)
end

const RHO_MID     = (RHO_LIGHT + RHO_HEAVY) / 2
const ATWOOD      = (RHO_HEAVY - RHO_LIGHT) / (RHO_HEAVY + RHO_LIGHT)
const GROWTH_RATE = sqrt(9.81 * ATWOOD * π)

println(@sprintf("Atwood number:    A = %.6f", ATWOOD))
println(@sprintf("Linear growth rate γ = %.6f", GROWTH_RATE))

# ── Mesh parameters ───────────────────────────────────────────────────────────
const NXB = 128
const NYB = 1024   # y is the vertical axis in 2D

const LX = 2.0
const LY = 16.0

const SAVETIME = 0.01

const Y_MID = LY / 2

# Physical coordinates of cell centres [2]
const xs = [(ix - 0.5) * (LX / NXB) for ix in 1:NXB]
const ys = [(iy - 0.5) * (LY / NYB) for iy in 1:NYB]

# Column indices closest to x = LX/2 (bubble) and x = 0 (spike) [6]
const bubble_col = argmin(abs.(xs .- (LX / 2)))
const spike_col  = argmin(abs.(xs .- 0.0))
println("Bubble column: ix = $bubble_col  x = $(xs[bubble_col])")
println("Spike  column: ix = $spike_col   x = $(xs[spike_col])")

const amp0 = 1e-4
const t0   = 0.0

# ── Tip tracking ──────────────────────────────────────────────────────────────
# slice is (NXB, NYB); y is the vertical axis
function find_bubble_tip(slice::Matrix{FT}, col::Int)
    col_rho    = slice[col, :]
    light_mask = col_rho .< RHO_MID
    any(light_mask) || return 0.0
    return maximum(ys[light_mask])
end

function find_spike_tip(slice::Matrix{FT}, col::Int)
    col_rho    = slice[col, :]
    heavy_mask = col_rho .> RHO_MID
    any(heavy_mask) || return 0.0
    return minimum(ys[heavy_mask])
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

        slice = Matrix{FT}(rho_dset[t, :, :])   # (NXB, NYB)

        bd = find_bubble_tip(slice, bubble_col)
        sd = find_spike_tip(slice, spike_col)

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
E_ref     = Y_MID .+ amp0 .* exp.(GROWTH_RATE .* (t_ref .- t0))
E_ref_sp  = Y_MID .- amp0 .* exp.(GROWTH_RATE .* (t_ref .- t0))

# ── Plot ──────────────────────────────────────────────────────────────────────
x_major = collect(0.0:1.0:ceil(maximum(times)))
x_minor = collect(0.0:0.5:ceil(maximum(times)))
y_major = collect(0.0:1.0:ceil(LY))
y_minor = collect(0.0:0.5:ceil(LY))

fig = Figure(size = (900, 550))
ax  = Axis(fig[1, 1];
    xlabel             = "t",
    ylabel             = "y",
    title              = @sprintf("RTI Front Tracking (2D)  |  A = %.4f  |  γ = %.4f", ATWOOD, GROWTH_RATE),
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
    limits             = (0, T_CUTOFF == Inf ? maximum(times) : T_CUTOFF, 0, LY),
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

outfile = joinpath(IMGDIR, "rti_front_tracking_2d.png")
save(outfile, fig)
println("Saved: $outfile")