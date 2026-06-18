# Shared GPU backend operator definitions.
#
# Including modules must define before including this file:
#   _BACKEND::Symbol            - backend name, e.g. :CUDA or :Metal
#   _DEFAULT_ARR_CONS           - default array constructor, e.g. CuArray or MtlArray
#   _backend_typed_sparse(::Type{T}, mat) where T  - wrap sparse matrix with element type T
#   _backend_sparse(mat)        - wrap sparse matrix preserving element type
#   _backend_dense(arr)         - wrap dense array/vector

const _BACKEND_VAL = Val{_BACKEND}

# Wedge Product
#--------------

function cache_wedge(::Val{m}, ::Val{n}, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p}, ::_BACKEND_VAL, arr_cons=_DEFAULT_ARR_CONS, cast_float=nothing) where {float_type,_p,m,n}
  cache_wedge(m, n, sd, float_type, arr_cons, cast_float)
end
function cache_wedge(::Val{m}, ::Val{n}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p}, ::_BACKEND_VAL, arr_cons=_DEFAULT_ARR_CONS, cast_float=nothing) where {float_type,_p,m,n}
  cache_wedge(m, n, sd, float_type, arr_cons, cast_float)
end

# Boundary and Co-boundary
#-------------------------

dec_boundary(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::_BACKEND_VAL) where float_type =
  _backend_typed_sparse(float_type, dec_boundary(n, sd))
dec_boundary(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::_BACKEND_VAL) where float_type =
  _backend_typed_sparse(float_type, dec_boundary(n, sd))

dec_dual_derivative(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::_BACKEND_VAL) where float_type =
  _backend_typed_sparse(float_type, dec_dual_derivative(n, sd))
dec_dual_derivative(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::_BACKEND_VAL) where float_type =
  _backend_typed_sparse(float_type, dec_dual_derivative(n, sd))

dec_differential(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::_BACKEND_VAL) where float_type =
  _backend_typed_sparse(float_type, dec_differential(n, sd))
dec_differential(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::_BACKEND_VAL) where float_type =
  _backend_typed_sparse(float_type, dec_differential(n, sd))

# Hodge Star
#-----------

dec_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::_BACKEND_VAL) =
  dec_hodge_star(Val(n), sd, h, Val(_BACKEND))

dec_hodge_star(::Val{n}, sd::HasDeltaSet, h::DiscreteHodge, ::_BACKEND_VAL) where n =
  _backend_dense(dec_hodge_star(Val(n), sd, h))

dec_hodge_star(::Val{1}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::_BACKEND_VAL) =
  _backend_sparse(dec_hodge_star(Val(1), sd, GeometricHodge()))

# Inverse Hodge Star
#-------------------

dec_inv_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::_BACKEND_VAL) =
  dec_inv_hodge_star(Val(n), sd, h, Val(_BACKEND))

dec_inv_hodge_star(::Val{n}, sd::HasDeltaSet, h::DiscreteHodge, ::_BACKEND_VAL) where n =
  _backend_dense(dec_inv_hodge_star(Val(n), sd, h))
