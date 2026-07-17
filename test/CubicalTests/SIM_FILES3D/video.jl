using HDF5
using CairoMakie

const SIM_NAME  = ARGS[1]
const SIM_NUM   = length(ARGS) >= 2 ? ARGS[2] : ""
const N_STEP    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10

const LOG_DIR = "/p/work2/grauta/3dlogs/$(SIM_NAME)test/$SIM_NUM"
const OUTFILE = joinpath(LOG_DIR, "output/acoustic.h5")
const IMGDIR  = joinpath(LOG_DIR, "imgs")
mkpath(IMGDIR)

h5open(OUTFILE, "r") do h5
    rho    = h5["fields/rho"]
    ntimes = size(rho, 1)

    # Mid-slice indices for each axis [2]
    x_mid = div(size(rho, 2), 2) + 1
    y_mid = div(size(rho, 3), 2) + 1
    z_mid = div(size(rho, 4), 2) + 1

    slices = (
        ("xy", (t -> rho[t, :, :, z_mid]),  "z-midplane  (xy)"),
        ("xz", (t -> rho[t, :, y_mid, :]),  "y-midplane  (xz)"),
        ("yz", (t -> rho[t, x_mid, :, :]),  "x-midplane  (yz)"),
    )

    for (label, extractor, panel_title) in slices
        frame1 = extractor(1)

        frame_obs = Observable(frame1)
        title_obs = Observable("$panel_title  |  t = 1")

        fig = Figure(size = (500, 500))
        ax  = Axis(fig[1, 1];
                   title   = title_obs,
                   xlabel  = "dim 1",
                   ylabel  = "dim 2",
                   aspect  = DataAspect())
        hm  = image!(ax, (0, 1), (0, 1), frame_obs; colormap = Reverse(:oslo))
        Colorbar(fig[1, 2], hm)

        resize_to_layout!(fig)

        outfile = joinpath(IMGDIR, "rho_$(label).mp4")
        record(fig, outfile, 1:N_STEP:ntimes; framerate = 30) do t
            frame_obs[] = extractor(t)
            title_obs[] = "$panel_title  |  t = $t"
        end

        println("Saved -> $outfile")
    end
end