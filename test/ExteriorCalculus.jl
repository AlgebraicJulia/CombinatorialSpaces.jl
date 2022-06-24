module TestExteriorCalculus
using Test

using Catlab
using CombinatorialSpaces.ExteriorCalculus

############
# 1D Tests #
############

@present Diffusion1DQuantities(FreeExtCalc1D) begin
  X::Space
  C::Hom(munit(), Form0(X))     # concentration
  ϕ::Hom(munit(), DualForm0(X)) # negative diffusion flux
  k::Hom(Form1(X), Form1(X))    # diffusivity (usually scalar multiplication)
end

@present Diffusion1D <: Diffusion1DQuantities begin
  # Fick's first law
  ϕ == C ⋅ d₀(X) ⋅ k ⋅ ⋆₁(X)
  # Diffusion equation
  C ⋅ ∂ₜ(Form0(X)) == ϕ ⋅ dual_d₀(X) ⋅ ⋆₀⁻¹(X)
end

@present Diffusion1D′ <: Diffusion1DQuantities begin
  # Diffusion equation
  C ⋅ ∂ₜ(Form0(X)) == C ⋅ d₀(X) ⋅ k ⋅ δ₁(X)
end

X, C, ϕ = Diffusion1D[:X], Diffusion1D[:C], Diffusion1D[:ϕ]
@test codom(C) == Form0(X)
@test codom(ϕ) == DualForm0(X)

############
# 2D Tests #
############

@present Diffusion2DQuantities(FreeExtCalc2D) begin
  X::Space
  C::Hom(munit(), Form0(X))     # concentration
  ϕ::Hom(munit(), DualForm1(X)) # negative diffusion flux
  k::Hom(Form1(X), Form1(X))    # diffusivity (usually scalar multiplication)
end

@present Diffusion2D <: Diffusion2DQuantities begin
  # Fick's first law
  ϕ == C ⋅ d₀(X) ⋅ k ⋅ ⋆₁(X)
  # Diffusion equation
  C ⋅ ∂ₜ(Form0(X)) == ϕ ⋅ dual_d₁(X) ⋅ ⋆₀⁻¹(X)
end

@present Diffusion2D′ <: Diffusion2DQuantities begin
  # Diffusion equation
  C ⋅ ∂ₜ(Form0(X)) == C ⋅ d₀(X) ⋅ k ⋅ δ₁(X)
end

X, C, ϕ = Diffusion2D[:X], Diffusion2D[:C], Diffusion2D[:ϕ]
@test codom(C) == Form0(X)
@test codom(ϕ) == DualForm1(X)

end
