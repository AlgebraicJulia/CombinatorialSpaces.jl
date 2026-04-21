const P₀ = 1e5 # Reference pressure for potential temperature
const Cₚ = 1006 # Specific heat
const R_gas = 287 # Specific gas constant
const ρᵣ = 1 # Reference density
const θᵣ = 300 # Reference potential temperature, temperature at P₀
const Pᵣ = ρᵣ * R_gas * θᵣ
const gₐ = -9.81 # Acceleration due to gravity

# From the ideal gas law: P = (Θ * R * (P₀^(-R/Cₚ)))^(Cₚ/(Cₚ-R))
function pressure(Theta)
  R_Cₚ = R_gas / Cₚ
  return (Theta .* R_gas .* (P₀ .^ -R_Cₚ)) .^ (1 / (1 - R_Cₚ))
end

# Assumes theta is constant in z
function hydrostatic_pressure(theta, h)
  return (1 / (Cₚ * theta) * (P₀)^(R_gas / Cₚ) * gₐ * h + Pᵣ^(R_gas / Cₚ))^(Cₚ / R_gas)
end

function hydrostatic_density(theta, h)
  Pₕ = hydrostatic_pressure(theta, h)
  return (Pₕ / (R_gas * theta)) * (P₀ / Pₕ)^(R_gas / Cₚ)
end

function pressure_same_theta(rho::AbstractVector{<:AbstractFloat})
  return R_gas * θᵣ * rho
end
