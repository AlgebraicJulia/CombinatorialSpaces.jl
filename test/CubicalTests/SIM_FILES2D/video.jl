using HDF5
using CairoMakie
using Printf

const SIM_NUM  = ARGS[1]

const LOG_DIR = "/p/home/grauta/git/CombinatorialSpaces.jl/test/CubicalTests/SIM_FILES2D/logs/$SIM_NUM"
const OUTFILE = joinpath(LOG_DIR, "output/savedata.h5")
const IMGDIR  = joinpath(LOG_DIR, "imgs")

const SAVETIME = (parse(Float64, ARGS[2]))
const WIDTH = (parse(Float64, ARGS[3]))
const HEIGHT = (parse(Float64, ARGS[4]))
const NSTEP = (parse(Int64, ARGS[5]))
const ASPECT = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : nothing

h5open(OUTFILE, "r") do h5
    rho = h5["fields/rho"]
    ntimes = size(rho, 1)
    ndims_spatial = ndims(rho) - 1  # 2 or 3

    # For 3D, pick the z-midplane slice index
    z_mid = ndims_spatial == 3 ? div(size(rho, 4), 2) + 1 : nothing

    # Read first frame to set colour limits
    frame1 = ndims_spatial == 3 ? rho[1, :, :, z_mid] : rho[1, :, :]
    clims = (minimum(frame1), maximum(frame1))

    ax_aspect = isnothing(ASPECT) ? DataAspect() : AxisAspect(ASPECT)

    fig = Figure(size = (400, 1600))
    ax  = CairoMakie.Axis(fig[1, 1]; title = "rho - t = 0.00", xlabel = "x", ylabel = "y",
        aspect = DataAspect(), xticks=0.0:0.5:WIDTH, yticks=0.0:2:HEIGHT)
    hm  = heatmap!(ax, (0, WIDTH), (0, HEIGHT), frame1; colorrange = clims, colormap = Reverse(:oslo))
    Colorbar(fig[1, 2], hm)

    # colsize!(fig.layout, 1, Aspect(1, isnothing(ASPECT) ? 1.0 : ASPECT))
    resize_to_layout!(fig)
    
    record(fig, "rho.mp4", 1:NSTEP:ntimes; framerate = 30) do t
        frame = ndims_spatial == 3 ? rho[t, :, :, z_mid] : rho[t, :, :]
        hm[3][] = frame
        ax.title = @sprintf("rho - t = %.2f", ((t-1)*SAVETIME))
    end
end

println("Saved -> rho.mp4")