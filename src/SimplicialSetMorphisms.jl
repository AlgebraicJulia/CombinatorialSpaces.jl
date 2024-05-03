module SimplicialSetMorphisms

using ..SimplicialSets
using StaticArrays: SMatrix
using Catlab

export ΔMap, pullback, simplicialMap
import AlgebraicInterfaces: dom, cod, compose

struct ΔMap{n,m,p,q,T<:Real}
  domain::Simplex{n}
  codomain::Simplex{m}

  ps::SMatrix{p,q,T}

  # https://stackoverflow.com/a/34075704
  # https://discourse.julialang.org/t/how-to-do-arithmetic-in-the-type-system/76565
  # https://math.stackexchange.com/questions/576638/how-to-calculate-the-pullback-of-a-k-form-explicitly
  function ΔMap(s::Simplex{n}, t::Simplex{m},ps::SMatrix{p,q,T}) where {n,m,p,q,T}
    p == n+1 && q == m+1 || throw(ArgumentError("ps must be of size $(n+1)×$(m+1)"))
    new{n,m,p,q,T}(s, t, ps)
  end
end

dom(x::ΔMap) = x.domain
cod(x::ΔMap) = x.codomain

struct simplicialMap 
  domain::HasDeltaSet
  codomain::HasDeltaSet
  map::Vector{Vector{ΔMap}} #bigger than usually needed XX
end





function compose(f::ΔMap{n,m,p,q,T}, g::ΔMap{m,l,r,s,T}) where {n,m,l,p,q,r,s,T}
  ΔMap(f.s, g.t, f.ps * g.ps)
end

representable(DeltaSet3D, SchDeltaSet3D, :Tet)

#function representableDeltaSet(::Val{3})
#    tiny = @acset DeltaSet3D() begin end
#    vs = add_vertices!(tiny,4)
#    add_tetrahedron!(tiny, vs...)
#    tiny
#end
#function representableDeltaSet(::Val{2})
#    tiny = @acset DeltaSet2D() begin end
#    vs = add_vertices!(tiny,3)
#    add_triangle!(tiny, vs...)
#    tiny
#end
#function representableDeltaSet(::Val{1})
#    tiny = @acset DeltaSet1D() begin end
#    vs = add_vertices!(tiny,2)
#    add_edge!(tiny, vs...)
#    tiny
#end
#function representableDeltaSet(::Val{0})
#    tiny = @acset DeltaSet0D() begin end
#    vs = add_vertices!(tiny,1)
#    tiny
#end

# TODO: Implement pullbacks for 0-forms
function pullback(Δdc::ΔMap{n,m,p,q,T}, ws::SimplexForm{0}) where {n,m,p,q,T<:Real} #w is an l-form on the codomain of Δdc
    vals = ws.data #values of the form
    SimplexForm{0}(Δdc.ps * vals)
end

end # SimplicialSetMorphisms
