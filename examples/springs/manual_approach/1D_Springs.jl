using Plots

time_steps = 10000
num_oscillators = 5
k = 0.5
dt = 0.01
dxm = 2
dx0 = 1.5
m = range(1, stop=0.25, length=num_oscillators)

values = Dict{Symbol, Array{Real, 2}}()

# Initialize harmonic oscillators
values[:IP] = zeros((num_oscillators+1, time_steps+1)) # x
values[:TP] = zeros((num_oscillators+1, time_steps))   # v
values[:IL] = zeros((num_oscillators, time_steps+1))   # Δx
values[:TP2] = zeros((num_oscillators+1, time_steps))  # f
values[:TL2] = zeros((num_oscillators, time_steps))    # ∑f
values[:IL2] = zeros((num_oscillators, time_steps+1))  # p

# Initial conditions
values[:IP][:,1] = range(0, step=dx0, length=num_oscillators+1)

# Coboundary operator from 0-simplex to 1-simplex in primal complex
x_to_Δx!(dx, x) = begin
    for i in 2:length(x)
        dx[i-1] = x[i] - x[i-1]
    end
end

# Constitutive equation from primal to dual complex
Δx_to_f!(f, dx) = begin
    f[1:(end-1)] .= k*(dx .- dxm)
end

# Coboundary operator from 0-simplex to 1-simplex in dual complex
f_to_∑f!(sf, f) = begin
    for i in 2:length(f)
        sf[i-1] = f[i] - f[i-1]
    end
end

# Solved coboundary (?) operator from 1-simplex to 0-simplex in dual temporal complex
∑f_to_p!(p, p0, sf) = begin
    p .= p0 .+ sf * dt
end

# Constitutive equation from dual to primal complex
p_to_v!(v, p) = begin
    v[2:end] .= p ./ m
end

# Solved coboundary (?) operator from 1-simplex to 0-simplex in dual temporal complex
v_to_x!(x, x0, v) = begin
    x .= x0 .+ v * dt
end

# Perform simulation loop
for i in 1:time_steps
    x_to_Δx!(view(values[:IL], :, i), view(values[:IP], :, i))
    Δx_to_f!(view(values[:TP2], :, i), view(values[:IL], :, i))
    f_to_∑f!(view(values[:TL2], :, i), view(values[:TP2], :, i))
    ∑f_to_p!(view(values[:IL2], :, i+1), view(values[:IL2], :, i), view(values[:TL2], :, i))
    p_to_v!(view(values[:TP], :, i), view(values[:IL2], :, i))
    ∑f_to_p!(view(values[:IP], :, i+1), view(values[:IP], :, i), view(values[:TP], :, i))
end

# Create animation of dynamic system
anim = @animate for i ∈ 1:20:(time_steps÷5)
    plot(zeros(6),values[:IP][:,i], legend=:none, seriestype = :scatter, ylims=[0,15], xlims=[-0.1, 0.1])
end
gif(anim, fps = 15)
