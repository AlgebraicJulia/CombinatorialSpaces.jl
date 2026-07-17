using HDF5
using CairoMakie
using Printf

include("../../../src/CubicalCode/UniformMesh.jl")
include("../../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../../src/CubicalCode/UniformKernelDEC.jl")

# const H5_PATH = ARGS[1]  # pass the .h5 file as a command-line argument
const ASPECT = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : nothing

const NXB = 512
const NYB = 512
const LX = 6.0
const LY = 6.0

const OMEGA_LIM =  10.0

const NX = NXB + 1
const NY = NYB + 1

const SAVETIME = 0.1

const s = UniformCubicalComplex2D(NX, NY, LX, LY)

const all_dps_q = dual_points(s)
const xs_q = map(p -> p[1], all_dps_q)
const ys_q = map(p -> p[2], all_dps_q)

const all_pts_v = points(s)
const xs_v = map(p -> p[1], all_pts_v)
const ys_v = map(p -> p[2], all_pts_v)

const dd1 = dual_derivative(Val(1), s)
const ihs_0 = inv_hodge_star(Val(0), s)

function compute_vorticity(h5, idx::Int)
    U_x = vec(h5["fields/U_1"][idx, :, :])
    U_y = vec(h5["fields/U_2"][idx, :, :])
    U = vcat(U_x, U_y)
    return reshape(ihs_0 * dd1 * U, NX, NY)
end

h5open(H5_PATH, "r") do h5
    rho = h5["fields/rho"]
    Theta = h5["fields/Theta"]

    ntimes = size(rho, 1)

    # Read first frame to set colour limits
    frame1 = Theta[1, :, :] ./ rho[1, :, :]
    clims = (minimum(frame1), maximum(frame1))

    ax_aspect = isnothing(ASPECT) ? DataAspect() : AxisAspect(ASPECT)

    time = Observable(1)

    title_obs = @lift(@sprintf("t = %.4f", ($time - 1) * SAVETIME))

    fig = Figure(size = (600, 950))
    Label(fig[0, :], title_obs; fontsize = 14)
    ax_theta  = CairoMakie.Axis(fig[1, 1]; 
        title = "Potential temperature (θ = Θ/ρ)", 
        xlabel = "x", 
        ylabel = "y",
        aspect = DataAspect(),
        )

    hm_theta  = image!(ax_theta, (0,6), (0,6), frame1; colorrange = clims, colormap = :inferno)
    Colorbar(fig[1, 2], hm_theta,
        width         = 15,
        ticklabelsize = 11,
        )

    frame2 = compute_vorticity(h5, 1)

    ax_omega = Axis(fig[2, 1];
        xlabel = "x", 
        ylabel = "y",
        title  = "Vorticity (ω)",
        aspect = DataAspect(),
    )
    hm_omega = image!(ax_omega, (0,6), (0,6), frame2;
        colormap   = :RdBu,
        colorrange = (-OMEGA_LIM, OMEGA_LIM)
    )
    Colorbar(fig[2, 2], hm_omega; label = "ω",
        width         = 15,
        ticklabelsize = 11,
        )

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 20)

    colsize!(fig.layout, 1, Aspect(1, isnothing(ASPECT) ? 1.0 : ASPECT))
    resize_to_layout!(fig)
    
    record(fig, "rho.mp4", 1:ntimes; framerate = 30) do t
        time[] = t
        hm_theta[3][] = Theta[t, :, :] ./ rho[t, :, :]
        hm_omega[3][] = compute_vorticity(h5, t)
    end
end

println("Saved -> rho.mp4")