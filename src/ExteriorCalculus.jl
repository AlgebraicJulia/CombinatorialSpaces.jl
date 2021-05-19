module ExteriorCalculus
export Ob, Hom, dom, codom, compose, ⋅, id,
  otimes, ⊗, munit, braid, oplus, ⊕, mzero, swap,
  mcopy, Δ, delete, ◊, plus, +, zero, antipode,
  MetricFreeExtCalc1D, MetricFreeExtCalc2D, ExtCalc1D, ExtCalc2D, FreeExtCalc2D,
  Space, Chain0, Chain1, Chain2, Form0, Form1, Form2, ∂₁, ∂₂, d₀, d₁,
  DualForm0, DualForm1, DualForm2, dual_d₀, dual_d₁, δ₁, δ₂,
  ⋆₀, ⋆₁, ⋆₂, ⋆₀⁻¹, ⋆₁⁻¹, ⋆₂⁻¹

using Catlab, Catlab.Theories
import Catlab.Theories: Ob, Hom, dom, codom, compose, ⋅, id,
  otimes, ⊗, munit, braid, oplus, ⊕, mzero, swap,
  mcopy, Δ, delete, ◊, plus, +, zero, antipode

""" Theory of additive (symmetric) monoidal categories.

Additive monoidal categories are additive categories with a symmetric monoidal
product that is bi-additive, i.e., additive in each variable separately.
Sometimes these are called "tensor categories" but that term is ambiguous, as
noted at the nLab. Not to be confused with an SMC that happens to be written
additively (`SymmetricMonoidalCategoryAdditive` in Catlab).

TODO: Migrate to `Catlab.Theories.MonoidalMultiple`.
"""
@theory AdditiveMonoidalCategory{Ob,Hom} <: AdditiveCategory{Ob,Hom} begin
  # Tensor product.
  otimes(A::Ob, B::Ob)::Ob
  otimes(f::(A → B), g::(C → D))::((A ⊗ C) → (B ⊗ D)) ⊣
    (A::Ob, B::Ob, C::Ob, D::Ob)
  @op (⊗) := otimes
  munit()::Ob
  braid(A::Ob, B::Ob)::((A ⊗ B) → (B ⊗ A))
  @op (σ) := braid
  # TODO: Tensor product axioms should be inherited along with SMC operators.

  # Distributity of sums of over products.
  #
  # In the standard categories like R-Mod, these would be distributors
  # (distributivity natural isomorphisms) but we replace them with equalities
  # following our usual approach to coherence maps like associators and unitors.
  (A ⊕ B) ⊗ C == (A ⊗ C) ⊕ (B ⊗ C) ⊣ (A::Ob, B::Ob, C::Ob)
  mzero() ⊗ A == mzero() ⊣ (A::Ob)
  A ⊗ (B ⊕ C) == (A ⊗ B) ⊕ (A ⊗ C) ⊣ (A::Ob, B::Ob, C::Ob)
  A ⊗ mzero() == mzero() ⊣ (A::Ob)

  # Bi-additivity.
  (f + g) ⊗ h == (f ⊗ h) + (g ⊗ h) ⊣
    (A::Ob, B::Ob, C::Ob, D::Ob, f::(A → B), g::(A → B), h::(C → D))
  (delete(A) ⋅ zero(B)) ⊗ f == delete(A⊗C) ⋅ zero(B⊗D) ⊣
    (A::Ob, B::Ob, C::Ob, D::Ob, f::(C → D))
  f ⊗ (g + h) == (f ⊗ g) + (f ⊗ h) ⊣
    (A::Ob, B::Ob, C::Ob, D::Ob, f::(A → B), g::(C → D), h::(C → D))
  f ⊗ (delete(C) ⋅ zero(D)) == delete(A⊗C) ⋅ zero(B⊗D) ⊣
    (A::Ob, B::Ob, C::Ob, D::Ob, f::(A → B))
end

# Metric-free exterior calculus
###############################

""" Theory of exterior caclulus on 1D manifold-like spaces.
"""
@theory MetricFreeExtCalc1D{Ob,Hom,Space} <: AdditiveMonoidalCategory{Ob,Hom} begin
  Space::TYPE

  Chain0(X::Space)::Ob
  Chain1(X::Space)::Ob
  ∂₁(X::Space)::Hom(Chain1(X), Chain0(X))

  Form0(X::Space)::Ob
  Form1(X::Space)::Ob
  d₀(X::Space)::Hom(Form0(X), Form1(X))
end

""" Theory of exterior calculus on 2D manifold-like spaces.
"""
@theory MetricFreeExtCalc2D{Ob,Hom,Space} <: MetricFreeExtCalc1D{Ob,Hom,Space} begin
  Chain2(X::Space)::Ob
  ∂₂(X::Space)::Hom(Chain2(X), Chain1(X))
  ∂₂(X) ⋅ ∂₁(X) == zero(Chain2(X), Chain0(X)) ⊣ (X::Space)

  Form2(X::Space)::Ob
  d₁(X::Space)::Hom(Form1(X), Form2(X))
  d₀(X) ⋅ d₁(X) == zero(Form0(X), Form2(X)) ⊣ (X::Space)
end

# Exterior calculus (with metric)
#################################

""" Theory of exterior calculus on 1D Riemannian manifold-like space.
"""
@theory ExtCalc1D{Ob,Hom,Space} <: MetricFreeExtCalc1D{Ob,Hom,Space} begin
  DualForm0(X::Space)::Ob
  DualForm1(X::Space)::Ob
  dual_d₀(X::Space)::Hom(DualForm0(X), DualForm1(X))

  ⋆₀(X::Space)::Hom(Form0(X), DualForm1(X))
  ⋆₁(X::Space)::Hom(Form1(X), DualForm0(X))
  ⋆₀⁻¹(X::Space)::Hom(DualForm1(X), Form0(X))
  ⋆₁⁻¹(X::Space)::Hom(DualForm0(X), Form1(X))
  ⋆₀(X) ⋅ ⋆₀⁻¹(X) == id(X) ⊣ (X::Ob)
  ⋆₁(X) ⋅ ⋆₁⁻¹(X) == id(X) ⊣ (X::Ob)

  δ₁(X::Space)::Hom(Form1(X), Form0(X))
  δ₁(X) == ⋆₁(X) ⋅ dual_d₀(X) ⋅ ⋆₀⁻¹(X) ⋅ antipode(Form0(X))
end

""" Theory of exterior calculus on 2D Riemannian manifold-like space.
"""
@theory ExtCalc2D{Ob,Hom,Space} <: MetricFreeExtCalc2D{Ob,Hom,Space} begin
  DualForm0(X::Space)::Ob
  DualForm1(X::Space)::Ob
  DualForm2(X::Space)::Ob
  dual_d₀(X::Space)::Hom(DualForm0(X), DualForm1(X))
  dual_d₁(X::Space)::Hom(DualForm1(X), DualForm2(X))
  dual_d₀(X) ⋅ dual_d₁(X) == zero(DualForm0(X), DualForm2(X)) ⊣ (X::Space)

  ⋆₀(X::Space)::Hom(Form0(X), DualForm2(X))
  ⋆₁(X::Space)::Hom(Form1(X), DualForm1(X))
  ⋆₂(X::Space)::Hom(Form2(X), DualForm0(X))
  ⋆₀⁻¹(X::Space)::Hom(DualForm2(X), Form0(X))
  ⋆₁⁻¹(X::Space)::Hom(DualForm1(X), Form1(X))
  ⋆₂⁻¹(X::Space)::Hom(DualForm0(X), Form2(X))
  ⋆₀(X) ⋅ ⋆₀⁻¹(X) == id(Form0(X)) ⊣ (X::Ob)
  ⋆₁(X) ⋅ ⋆₁⁻¹(X) == antipode(Form1(X)) ⊣ (X::Ob)
  ⋆₂(X) ⋅ ⋆₂⁻¹(X) == id(Form1(X)) ⊣ (X::Ob)

  δ₁(X::Space)::Hom(Form1(X), Form0(X))
  δ₂(X::Space)::Hom(Form2(X), Form1(X))
  δ₁(X) == ⋆₁(X) ⋅ dual_d₁(X) ⋅ ⋆₀⁻¹(X) ⋅ antipode(Form0(X))
  δ₂(X) == ⋆₂(X) ⋅ dual_d₀(X) ⋅ ⋆₁⁻¹(X)
end

@syntax FreeExtCalc2D{ObExpr,HomExpr,GATExpr} ExtCalc2D begin
  compose(f::Hom, g::Hom) = associate_unit(new(f,g; strict=true), id)
  oplus(A::Ob, B::Ob) = associate_unit(new(A,B), mzero)
  oplus(f::Hom, g::Hom) = associate(new(f,g))
  otimes(A::Ob, B::Ob) = associate_unit(new(A,B), munit)
  otimes(f::Hom, g::Hom) = associate(new(f,g))
end

end
