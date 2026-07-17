using HDF5, CairoMakie, TOML

include("../../../src/CubicalCode/UniformMesh.jl")
include("../../../src/CubicalCode/UniformMesh3D.jl")
include("../../../src/CubicalCode/UniformKernelDEC.jl")
include("../../../src/CubicalCode/UniformKernelDEC3D.jl")

# const LOG_DIR  = "/p/work2/grauta/3dlogs/khtest/"
const SIM_NAME = ARGS[1]
const SIM_NUM = length(ARGS) == 2 ? ARGS[2] : ""
const LOG_DIR  = "/p/work2/grauta/3dlogs/$(SIM_NAME)test/$SIM_NUM"
const OUTFILE  = joinpath(LOG_DIR, "output/acoustic.h5")
const CONFIG   = TOML.parsefile(joinpath(@__DIR__, "Examples", "$SIM_NAME.toml"))
const IMGDIR   = joinpath(LOG_DIR, "imgs")

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
const s_global = UniformCubicalComplex3D(NX, NY, NZ, FT(LX), FT(LY), FT(LZ))

const y_mid = div(NYB, 2) + 1
const x_mid = div(NXB, 2) + 1
const z_mid = div(NZB, 2) + 1

# ── helper: save a plain heatmap ─────────────────────────────────────────────
function save_line(path, title, xlabel, ylabel, data; colorrange = nothing)
    fig = Figure(; size = (700, 600))
    ax  = Axis(fig[1, 1]; title, xlabel, ylabel)
    hm  = colorrange === nothing ? lines!(ax, data) : lines!(ax, data; colorrange)

    save(path, fig)
end

# ── helper: save a symmetric-diverging difference heatmap ────────────────────
function save_diff_line(path, title, xlabel, ylabel, diff)
    lim = max(abs(minimum(diff)), abs(maximum(diff)))
    lim = lim == 0 ? 1.0 : lim
    fig = Figure(; size = (700, 600))
    ax  = Axis(fig[1, 1]; title, xlabel, ylabel)
    hm  = lines!(ax, diff)

    save(path, fig)
end

h5open(OUTFILE, "r") do h5

    # ── scalar fields: rho and Theta ─────────────────────────────────────────
    for field in ("rho", "Theta")
        dset   = h5["fields/$field"]
        ntimes = HDF5.get_extent_dims(HDF5.dataspace(dset))[1][1]

        ic    = dset[1,      :, :, :]
        final = dset[ntimes, :, :, :]
        diff  = final .- ic

        for (slc, slc_label, xlabel, ylabel) in (
            (ic[x_mid,    y_mid, :], "z_line", "z", "theta"),
            (ic[x_mid, :,   z_mid], "y_line", "y", "theta"),
            (ic[:,    y_mid, z_mid], "x_line", "x", "theta"),
        )
            save_line(
                joinpath(IMGDIR, "$(field)_$(slc_label)_ic.png"),
                "$field $slc_label | IC", xlabel, ylabel, slc)
        end

        for (slc, slc_label, xlabel, ylabel) in (
            (final[x_mid,    y_mid, :], "z_line", "z", "theta"),
            (final[x_mid, :,   z_mid], "y_line", "y", "theta"),
            (final[:,    y_mid, z_mid], "x_line", "x", "theta"),
        )
            save_line(
                joinpath(IMGDIR, "$(field)_$(slc_label)_final.png"),
                "$field $slc_label | final", xlabel, ylabel, slc)
        end

        for (slc, slc_label, xlabel, ylabel) in (
            (diff[x_mid,    y_mid, :], "z_line", "z", "theta"),
            (diff[x_mid, :,   z_mid], "y_line", "y", "theta"),
            (diff[:,    y_mid, z_mid], "x_line", "x", "theta"),
        )
            save_diff_line(
                joinpath(IMGDIR, "$(field)_$(slc_label)_diff.png"),
                "$field $slc_label | final − IC", xlabel, ylabel, slc)
        end

    end

    # ── potential temperature: theta = Theta / rho ───────────────────────────
    ic_theta    = h5["fields/Theta"][1,   :, :, :] ./ h5["fields/rho"][1,   :, :, :]
    final_theta = h5["fields/Theta"][end, :, :, :] ./ h5["fields/rho"][end, :, :, :]
    diff_theta  = final_theta .- ic_theta

    for (data, label) in ((ic_theta, "ic"), (final_theta, "final"))
        for (slc, slc_label, xlabel, ylabel) in (
            (data[x_mid,    y_mid, :], "z_line", "z", "theta"),
            (data[x_mid, :,   z_mid], "y_line", "y", "theta"),
            (data[:,    y_mid, z_mid], "x_line", "x", "theta"),
        )
            save_line(
                joinpath(IMGDIR, "pot_temp_$(slc_label)_$(label).png"),
                "theta $slc_label | $label", xlabel, ylabel, slc)
        end
    end

    for (slc, slc_label, xlabel, ylabel) in (
        (diff_theta[x_mid,    y_mid, :], "z_line", "z", "theta"),
        (diff_theta[x_mid, :,   z_mid], "y_line", "y", "theta"),
        (diff_theta[:,    y_mid, z_mid], "x_line", "x", "theta"),
    )
        save_diff_line(
            joinpath(IMGDIR, "pot_temp_$(slc_label)_diff.png"),
            "theta $slc_label | final − IC", xlabel, ylabel, slc)
    end

    println("Output leader: plots saved to $IMGDIR")
end