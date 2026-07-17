###############################################################################
#  plot_rho_3d.jl
###############################################################################

using HDF5
using GLMakie
using Printf

const SIM_NAME  = ARGS[1]
const SIM_NUM   = length(ARGS) >= 2 ? ARGS[2] : ""
const N_STEP    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10

const LOG_DIR = "/p/work2/grauta/3dlogs/$(SIM_NAME)test/$SIM_NUM"
const OUTFILE = joinpath(LOG_DIR, "output/acoustic.h5")
const IMGDIR  = joinpath(LOG_DIR, "imgs")
mkpath(IMGDIR)

const NXB = 128
const NYB = 128
const NZB = 256

const LX = 2.0
const LY = 2.0
const LZ = 4.0

const SAVETIME = 0.01

# ── Manual color limits — adjust as needed ────────────────────────────────────
const RHO_MIN    = 0.95
const RHO_MAX    = 1.15
const ABSORPTION = 100.0

h5open(OUTFILE, "r") do h5
    rho    = h5["fields/rho"]
    ntimes = size(rho, 1)

    frame1 = rho[1, :, :, :]

    rho_obs   = Observable(frame1)
    title_obs = Observable("ρ  |  t = 0.00")

    fig = Figure(size = (600, 600))

    ls = LScene(fig[1, 1]; show_axis = true)
    
    vol = volume!(ls, rho_obs;
        colormap   = Reverse(:oslo),
        colorrange = (minimum(frame1), maximum(frame1)),
        algorithm  = :absorption,
        absorption = ABSORPTION,
        transparency = true,
    )
    
    center!(ls.scene)
    zoom!(ls.scene, 3.0)
        
    # Manual axis labels via 3D text annotations
    text!(ls, "x"; position = Point3f(size(frame1, 1) / 2, -5, -5), fontsize = 14)
    text!(ls, "y"; position = Point3f(-5, size(frame1, 2) / 2, -5), fontsize = 14)
    text!(ls, "z"; position = Point3f(-5, -5, size(frame1, 3) / 2), fontsize = 14)
    
    Colorbar(fig[1, 2], vol;
        label         = "ρ",
        width         = 15,
        ticklabelsize = 11,
    )
    
    colsize!(fig.layout, 1, Relative(0.9))
    resize_to_layout!(fig)

    record(fig, joinpath(IMGDIR, "rho_3d.mp4"), 1:N_STEP:ntimes; framerate = 30) do t
        rho_obs[]   = rho[t, :, :, :]
        title_obs[] = @sprintf("ρ  |  t = %.2f", ((t-1) * SAVETIME))
    end
end

println("Saved -> $(joinpath(IMGDIR, "rho_3d.mp4"))")