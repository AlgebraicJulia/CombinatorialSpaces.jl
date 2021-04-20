using Catlab
using Catlab.Theories
using Catlab.LinearAlgebra
using Catlab.LinearAlgebra.GraphicalLinearAlgebra
# @theory DEC{ℕ, Space, Ob, Hom} begin
#     n::ℕ
#     α::Scalar

#     # Spaces
#     A,B,C::Space        # [Δ^op, Set]
#     ∅::Space            # empty space no points
#     •(n::ℕ)::Space      # singleton space with 1 n-simplex,
#     P = •(0)            # used as ℝ
#     A×B::Space          # product space
#     dual(A)::Space      # dual space
#     dim(A::Space)::ℕ    # dimension of the space

#     # Forms
#     F(A::Space, n::ℕ)::Ob # n-forms over A
#     F(A, n) ⊕ F(B, n) = F(A⊔B, n) # product of spaces is direct sum of forms
#     # is a biproduct for a fixed F(A,n)

#     # Chains
#     C(A::Space, n::ℕ)::(F(A,n)→ F(P, 0)) # chains map forms to scalars

#     # Topological Operators
#     d(A, n):: F(A, n)   → F(A, n+1)
#     ∂(A, n):: C(A, n+1) → C(A, n)
#     δ(A, n):: F(A, n+1) → F(A, n)
#     ⋆(A, k):: F(A, k)   → F(dual(A), dim(A)-k)

#     δ(A)::F(A) → F(A)

#     # axioms for dg
#     # d(A, n) ⋅ d(A,n-1)) = 0
#     # ∂(A, n) ⋅ ∂(A,n+1)) = 0
#     # δ(A, n) ⋅ δ(A,n-1)) = 0

#     munit(oplus) = F(∅,0) = I

#     # maps between forms
#     # F(A,n) → F(B,m) are arbitrary functions in Set
#     plus(X::Ob) :: X⊕X → X
#     +(X→Y), (X→Y)) :: X→Y # choose A = ∅ for pointwise addition of forms
#     + = copy(X) ⋅ f⊕g ⋅ plus(Y)

#     scale(α::Scalar, X::Ob)::X→X
#     plus(X)⋅scale(α, X) = scale(α)⊕scale(α) ⋅ plus(A,n) # linearity of addition

#     # F(A,n) is a vector space include all GAT data + axioms

#     x ↦ -k(x-x₀)^2
#     # TODO differential operator of sheaves
#     #
#     constitutive relations
#     f::Hom = g::Hom

#     grothendieck construction of functor that takes A to sheaves over A
# end


@theory DEC{Ob, Hom} <: LinearFunctions{Ob, Hom} begin
    ℝ()::Ob
    dual(X::Ob)::Ob

    # topological operators
    ∂(X::Ob)::(X → X)
    d(X::Ob)::(X → X)
    δ(X::Ob)::(X → X)
    hodge(X::Ob)::(X → dual(X))


    # axioms for differentially graded category
    ∂(X)⋅∂(X) == zero(X)
    d(X)⋅d(X) == zero(X)
    δ(X)⋅δ(X) == zero(X)

    # axioms for hodge star
    # TBD

    # chain complexes are homs into singleton complex
    chain(X::Ob)::Hom(X, ℝ())
    𝟏(X::Ob)::Hom(X, ℝ())
    𝟏(ℝ) == id(ℝ())

    # laplace de rahm operator
    # Δ was taken by copy/Δuplicate
    L(X::Ob)::(X→X)
    L(X) == copy(X) ⋅ (d(X)⋅δ(X) ⊕ δ(X)⋅d(X)) ⋅ plus(X)
end

@syntax FreeDEC{ObExpr, HomExpr} DEC begin
  oplus(A::Ob, B::Ob) = associate_unit(new(A,B), mzero)
  oplus(R::Hom, S::Hom) = associate(new(R,S))
  compose(R::Hom, S::Hom) = new(R,S; strict=true) # No normalization!
end


@present Laplace(FreeDEC) begin
    A::Ob #cochain complex over the mesh
    f::Hom(mzero(), A)
    g::Hom(mzero(), A)

    f⋅L(A)⋅antipode(A) == g
end

A = Ob(FreeDEC.Ob, :A)
B = Ob(FreeDEC.Ob, :B)
C = Ob(FreeDEC.Ob, :C)

I = mzero(FreeDEC.Ob)
f = Hom(:f, I, A)
g = Hom(:g, I, A)
h = Hom(:h, A, B)
k = Hom(:k, B, C)

((f⋅∂(A))⊕(g⋅δ(A)⋅antipode(A)))⋅plus(A) |> show_unicode
