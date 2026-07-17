using HDF5, CairoMakie, TOML, FFTW

include("../../../src/CubicalCode/UniformMesh.jl")
include("../../../src/CubicalCode/UniformMesh3D.jl")
include("../../../src/CubicalCode/UniformKernelDEC.jl")
include("../../../src/CubicalCode/UniformKernelDEC3D.jl")

const SIM_NAME = ARGS[1]
const SIM_NUM  = length(ARGS) >= 2 ? ARGS[2] : ""
const N_STEP   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1

const LOG_DIR  = "/p/work2/grauta/3dlogs/$(SIM_NAME)test/$SIM_NUM"
const OUTFILE  = joinpath(LOG_DIR, "output/acoustic.h5")
const CONFIG   = TOML.parsefile(joinpath(@__DIR__, "Examples", "$SIM_NAME.toml"))
const IMGDIR   = joinpath(LOG_DIR, "imgs", "fft")

mkpath(IMGDIR)
println("Loading data from $OUTFILE...")

const FT = Float64
const NX = CONFIG["Mesh"]["nx"]
const NY = CONFIG["Mesh"]["ny"]
const NZ = CONFIG["Mesh"]["nz"]
const LX = FT(CONFIG["Mesh"]["lx"])
const LY = FT(CONFIG["Mesh"]["ly"])
const LZ = FT(CONFIG["Mesh"]["lz"])
const SAVETIME = FT(CONFIG["Simulation"]["savetime"])

const NXB = NX - 1
const NYB = NY - 1
const NZB = NZ - 1

const s_global = UniformCubicalComplex3D(NX, NY, NZ, LX, LY, LZ)

const y_mid = div(NYB, 2) + 1
const x_mid = div(NXB, 2) + 1
const z_mid = div(NZB, 2) + 1

const dx = LX / NXB
const dy = LY / NYB
const dz = LZ / NZB

# ── FFT helper ────────────────────────────────────────────────────────────────

function power_spectrum(signal::AbstractVector{<:Real}, dl::Real)
    n     = length(signal)
    F     = rfft(signal)
    power = abs2.(F) ./ n^2
    freqs = rfftfreq(n, 1 / dl)
    nondim_wavenumbers = 2π .* freqs * 0.5
    return nondim_wavenumbers, power
end

# ── MP4 builder using Observables ─────────────────────────────────────────────

function record_fft_video(h5, field_name, slc_label, dl, extractor, ntimes, outpath)

    # Pre-compute all frames to find stable axis limits
    all_k_star = nothing
    all_power = Vector{Vector{FT}}(undef, length(1:N_STEP:ntimes))

    for (i, t) in enumerate(1:N_STEP:ntimes)
        data = field_name == "pot_temp" ?
            h5["fields/Theta"][t, :, :, :] ./ h5["fields/rho"][t, :, :, :] :
            h5["fields/$field_name"][t, :, :, :]
        k_star, power = power_spectrum(extractor(data), dl)
        all_k_star    = k_star
        all_power[i] = power
    end

    idx  = all_k_star .> 0
    ymin = minimum(p -> minimum(p[idx]), all_power)
    ymax = maximum(p -> maximum(p[idx]), all_power)
    ymin = ymin > 0 ? ymin : 1e-20

    # ── Observables ───────────────────────────────────────────────────────────
    power_obs = Observable(all_power[1][idx])
    title_obs = Observable("$field_name $slc_label | FFT power | t = 0.0000")

    fig = Figure(; size = (800, 500))
    Label(fig[0, 1], title_obs; tellwidth = false)
    ax  = Axis(fig[1, 1];
               xlabel = "wavenumber k ($slc_label, cycles/m)",
               ylabel = "Power",
            #    xscale = log10,
               yscale = log10,
               limits = (nothing, (ymin, ymax)),
               xminorticksvisible = true,
               yminorticksvisible = true)

    lines!(ax, all_k_star[idx], power_obs)

    record(fig, outpath, enumerate(1:N_STEP:ntimes); framerate = 10) do (i, t)
        sim_time    = (t - 1) * SAVETIME
        power_obs[] = all_power[i][idx]
        title_obs[] = "$field_name $slc_label | FFT power | t = $(round(sim_time; digits=4))"
    end

    println("Saved $outpath")
end

# ── main ──────────────────────────────────────────────────────────────────────

h5open(OUTFILE, "r") do h5
    ntimes = size(h5["fields/rho"], 1)

    slices = (
        # ("z_line", dz, data -> data[x_mid, y_mid, :]),
        ("y_line", dy, data -> data[x_mid, :,   z_mid]),
        ("x_line", dx, data -> data[:,    y_mid, z_mid]),
    )

    for field in ("rho", "Theta", "pot_temp")
        for (slc_label, dl, extractor) in slices
            outpath = joinpath(IMGDIR, "$(field)_$(slc_label)_fft.mp4")
            record_fft_video(h5, field, slc_label, dl, extractor, ntimes, outpath)
        end
    end
end

println("Done. Videos saved to $IMGDIR")