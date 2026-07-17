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
function save_heatmap(path, title, xlabel, ylabel, data; colorrange = nothing)
    fig = Figure(; size = (700, 600))
    ax  = Axis(fig[1, 1]; title, xlabel, ylabel, aspect = DataAspect())
    hm  = colorrange === nothing ? heatmap!(ax, data) :
                                   heatmap!(ax, data; colorrange)
    Colorbar(fig[1, 2], hm)

    colsize!(fig.layout, 1, Aspect(1, 1.0))
    resize_to_layout!(fig)

    save(path, fig)
end

# ── helper: save a symmetric-diverging difference heatmap ────────────────────
function save_diff_heatmap(path, title, xlabel, ylabel, diff)
    lim = max(abs(minimum(diff)), abs(maximum(diff)))
    lim = lim == 0 ? 1.0 : lim
    fig = Figure(; size = (700, 600))
    ax  = Axis(fig[1, 1]; title, xlabel, ylabel, aspect = DataAspect())
    hm  = heatmap!(ax, diff; colormap = :RdBu, colorrange = (-lim, lim))
    Colorbar(fig[1, 2], hm)

    colsize!(fig.layout, 1, Aspect(1, 1.0))
    resize_to_layout!(fig)

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
            (ic[:,    y_mid, :], "y_slice", "x", "z"),
            (ic[x_mid, :,   :], "x_slice", "y", "z"),
            (ic[:,    :, z_mid], "z_slice", "x", "y"),
        )
            save_heatmap(
                joinpath(IMGDIR, "$(field)_$(slc_label)_ic.png"),
                "$field $slc_label | IC", xlabel, ylabel, slc)
        end

        for (slc, slc_label, xlabel, ylabel) in (
            (final[:,    y_mid, :], "y_slice", "x", "z"),
            (final[x_mid, :,   :], "x_slice", "y", "z"),
            (final[:,    :, z_mid], "z_slice", "x", "y"),
        )
            save_heatmap(
                joinpath(IMGDIR, "$(field)_$(slc_label)_final.png"),
                "$field $slc_label | final", xlabel, ylabel, slc)
        end

        for (slc, slc_label, xlabel, ylabel) in (
            (diff[:,    y_mid, :], "y_slice", "x", "z"),
            (diff[x_mid, :,   :], "x_slice", "y", "z"),
            (diff[:,    :, z_mid], "z_slice", "x", "y"),
        )
            save_diff_heatmap(
                joinpath(IMGDIR, "$(field)_$(slc_label)_diff.png"),
                "$field $slc_label | final − IC", xlabel, ylabel, slc)
        end

        for (slc, slc_label, xlabel, ylabel) in (
            (final[:,    1, :], "y_slice", "x", "z"),
            (final[1, :,   :], "x_slice", "y", "z"),
            (final[:,    :, 1], "z_slice", "x", "y"),
        )
            save_heatmap(
                joinpath(IMGDIR, "$(field)_$(slc_label)_low_bound.png"),
                "$field above low bound | final", xlabel, ylabel, slc)
        end

    end

    # ── potential temperature: theta = Theta / rho ───────────────────────────
    ic_theta    = h5["fields/Theta"][1,   :, :, :] ./ h5["fields/rho"][1,   :, :, :]
    final_theta = h5["fields/Theta"][end, :, :, :] ./ h5["fields/rho"][end, :, :, :]
    diff_theta  = final_theta .- ic_theta

    for (data, label) in ((ic_theta, "ic"), (final_theta, "final"))
        for (slc, slc_label, xlabel, ylabel) in (
            (data[:,    y_mid, :], "y_slice", "x", "z"),
            (data[x_mid, :,   :], "x_slice", "y", "z"),
            (data[:,    :, z_mid], "z_slice", "x", "y"),
        )
            save_heatmap(
                joinpath(IMGDIR, "pot_temp_$(slc_label)_$(label).png"),
                "theta $slc_label | $label", xlabel, ylabel, slc)
        end
    end

    for (slc, slc_label, xlabel, ylabel) in (
        (diff_theta[:,    y_mid, :], "y_slice", "x", "z"),
        (diff_theta[x_mid, :,   :], "x_slice", "y", "z"),
        (diff_theta[:,    :, z_mid], "z_slice", "x", "y"),
    )
        save_diff_heatmap(
            joinpath(IMGDIR, "pot_temp_$(slc_label)_diff.png"),
            "theta $slc_label | final − IC", xlabel, ylabel, slc)
    end

    # ── momentum components via sharp_dd ─────────────────────────────────────
    dset_ux = h5["fields/U_1"]
    dset_uy = h5["fields/U_2"]
    dset_uz = h5["fields/U_3"]
    ntimes  = HDF5.get_extent_dims(HDF5.dataspace(dset_ux))[1][1]

    function momentum_xyz(t::Int)
        u_flat = vcat(dset_ux[t, :, :, :][:],
                      dset_uy[t, :, :, :][:],
                      dset_uz[t, :, :, :][:])
        X, Y, Z = sharp_dd(s_global, u_flat)
        return (reshape(X, NXB, NYB, NZB),
                reshape(Y, NXB, NYB, NZB),
                reshape(Z, NXB, NYB, NZB))
    end

    ic_X,    ic_Y,    ic_Z    = momentum_xyz(1)
    final_X, final_Y, final_Z = momentum_xyz(ntimes)

    cr = (-1.0, 1.0)

    for (comp_label, ic_data, final_data) in (
        ("momentum_X", ic_X, final_X),
        ("momentum_Y", ic_Y, final_Y),
        ("momentum_Z", ic_Z, final_Z),
    )
        diff_data = final_data .- ic_data

        for (data, label) in ((ic_data, "ic"), (final_data, "final"))
            for (slc, slc_label, xlabel, ylabel) in (
                (data[:,    y_mid, :], "y_slice", "x", "z"),
                (data[x_mid, :,   :], "x_slice", "y", "z"),
                (data[:,    :, z_mid], "z_slice", "x", "y"),
            )
                save_heatmap(
                    joinpath(IMGDIR, "$(comp_label)_$(slc_label)_$(label).png"),
                    "$comp_label $slc_label | $label", xlabel, ylabel, slc;
                    colorrange = cr)
            end
        end

        for (slc, slc_label, xlabel, ylabel) in (
            (diff_data[:,    y_mid, :], "y_slice", "x", "z"),
            (diff_data[x_mid, :,   :], "x_slice", "y", "z"),
            (diff_data[:,    :, z_mid], "z_slice", "x", "y"),
        )
            save_diff_heatmap(
                joinpath(IMGDIR, "$(comp_label)_$(slc_label)_diff.png"),
                "$comp_label $slc_label | final − IC", xlabel, ylabel, slc)
        end
    end

    println("Output leader: plots saved to $IMGDIR")
end