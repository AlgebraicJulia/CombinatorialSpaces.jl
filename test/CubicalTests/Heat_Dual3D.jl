using OrdinaryDiffEqTsit5
using Distributions
using CairoMakie
using KernelAbstractions

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMesh3D.jl")
include("../../src/CubicalCode/UniformKernelDEC3D.jl")
include("../../src/CubicalCode/UniformPlotting.jl")

iter = 0

# --- 1. Right Hand Side for the Heat Equation ---
# Computes ∂u/∂t = k * Δu using DEC operators
function heat_rhs!(du, u, p, t)
    s, k_diffusion, boundary_idx = p
    
    # u is a dual 0-form (lives on boid centers).
    
    # 1. Gradient: Dual 0-form -> Dual 1-form
    grad_u = dual_derivative(Val(0), s, u)
    grad_u[boundary_idx] .= 0 # Flux conditions
    
    # Flux = -k * ∇u
    flux_dual = -k_diffusion .* grad_u
    
    # 2. Map to primal mesh: Dual 1-form -> Primal 2-form
    # In 3D, the inverse hodge star maps a dual 1-form to a primal 2-form (faces)
    flux_primal = inv_hodge_star(Val(2), s, flux_dual)
    
    # 3. Divergence: Primal 2-form -> Primal 3-form
    div_flux_primal = exterior_derivative(Val(2), s, flux_primal)
    
    # 4. Map back to dual points: Primal 3-form -> Dual 0-form
    laplacian_u = hodge_star(Val(3), s, div_flux_primal)
    
    # Update the derivative array
    du .= laplacian_u

    if ((global iter += 1) % 100 == 0)
      println("Current time in simulation is $(t)")
    end
end

function run_3d_heat_diffusion()
    println("Setting up 3D Mesh and Initial Conditions...")
    
    nx_, ny_, nz_ = 81, 81, 81
    lx_, ly_, lz_ = 5.0, 5.0, 5.0
    
    s = UniformCubicalComplex3D(nx_, ny_, nz_, lx_, ly_, lz_)
    FT = Float64
    
    u0 = zeros(FT, nboids(s))
    center = [lx_/2, ly_/2, lz_/2]
    covariance = [0.5, 0.5, 0.5]
    initial_dist = MvNormal(center, covariance)
    
    for i in 1:nboids(s)
        x, y, z = boid_to_coord(s, i)
        dp = dual_point(s, x, y, z)
        u0[i] = pdf(initial_dist, dp) * 10.0 
    end

    east, west = primal_boundary_quads(s, EASTWEST)
    north, south = primal_boundary_quads(s, NORTHSOUTH)
    up, down = primal_boundary_quads(s, UPDOWN)
    boundary_idx  = vcat(east, west, north, south, up, down)

    # --- Plot Slices of INITIAL Condition ---
    println("Plotting Initial Condition Slices...")
    z_slices = [div(nz_, 2) - 5, div(nz_, 2), div(nz_, 2) + 5]
    
    mkpath("imgs/Heat3D")
    
    for slice_idx in z_slices
        fig = plot_dual_zeroform_slice(
            s, 
            u0, 
            Z_ALIGN, 
            slice_idx; 
            figure_kwargs=(size=(600, 500),),
            heatmap_kwargs=(colorrange=(0.0, maximum(u0)),) # Lock to initial max
        )
        filename = "Heat3D_Initial_Slice_Z$(slice_idx).png"
        save(joinpath("imgs/Heat3D", filename), fig)
        println("  Saved $filename")
    end
    
    # --- ODE Problem Setup ---
    println("Configuring ODE Solver...")
    tspan = (0.0, 1.0)
    k_diffusion = 0.5
    p = (s, k_diffusion, boundary_idx)
    
    prob = ODEProblem(heat_rhs!, u0, tspan, p)
    
    # --- Solve ---
    println("Solving Heat Diffusion (this may take a moment)...")
    sol = solve(prob, Tsit5(), saveat=0.025, progress=true)
    
    u_final = sol[end]
    println("Simulation complete. Generating plots...")
    
    # --- Plotting Slices ---
    for slice_idx in z_slices
        fig = plot_dual_zeroform_slice(
            s, 
            u_final, 
            Z_ALIGN, 
            slice_idx; 
            figure_kwargs=(size=(600, 500),),
            heatmap_kwargs=(colorrange=(0.0, maximum(u_final)),) # Lock colorrange for comparison
        )
        
        filename = "Heat3D_Slice_Z$(slice_idx).png"
        save(joinpath("imgs/Heat3D", filename), fig)
        println("Saved slice plot to $filename")
    end
    
    println("All done!")
    return sol
end

sol = run_3d_heat_diffusion()