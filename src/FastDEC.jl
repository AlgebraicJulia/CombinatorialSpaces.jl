""" The discrete exterior calculus (DEC) with high performance in mind.

This module provides similar functionality to the DiscreteExteriorCalculus module
but uses assumptions about the ACSet mesh structure to greatly improve performance.
Some operators, like the exterior derivative are returned as sparse matrices while others,
like the wedge product, are instead returned as functions that will compute the product.
"""
module FastDEC

# TODO: Using Int32 for indices is valid for all but the largest meshes.
# This type should be user-settable, defaulting to Int32.

using ..SimplicialSets, ..DiscreteExteriorCalculus
using ..DiscreteExteriorCalculus: crossdot

using ACSets
using Base.Iterators
using KernelAbstractions
using LinearAlgebra: cross, dot, Diagonal, factorize, norm
using SparseArrays: sparse, spzeros, SparseMatrixCSC
using StaticArrays: SVector, MVector

import ..DiscreteExteriorCalculus: ∧
import ..SimplicialSets: numeric_sign

export dec_wedge_product, cache_wedge, dec_c_wedge_product, dec_c_wedge_product!,
  dec_boundary, dec_differential, dec_dual_derivative,
  dec_hodge_star, dec_inv_hodge_star,
  dec_wedge_product_pd, dec_wedge_product_dp, ∧,
  interior_product_dd, ℒ_dd,
  dec_wedge_product_dd,
  Δᵈ,
  avg₀₁, avg_01, avg₀₁_mat, avg_01_mat

# Wedge Product
#--------------

# Cache coefficients to be used by wedge product kernels.
function wedge_kernel_coeffs(::Type{Tuple{0,1}}, sd::Union{EmbeddedDeltaDualComplex1D, EmbeddedDeltaDualComplex2D})
  (hcat(convert(Vector{Int32}, sd[:∂v0])::Vector{Int32}, convert(Vector{Int32}, sd[:∂v1])::Vector{Int32}),
   ne(sd))
end

function wedge_kernel_coeffs(::Type{Tuple{0,2}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p}) where {float_type, _p}
  verts = Array{Int32}(undef, 6, ntriangles(sd))
  coeffs = Array{float_type}(undef, 6, ntriangles(sd))
  shift::Int = ntriangles(sd)
  @inbounds for t in triangles(sd)
    for dt in 1:6
      dt_real = t + (dt - 1) * shift
      verts[dt, t] = sd[sd[dt_real, :dual_∂e2], :dual_∂v1]
      coeffs[dt, t] = sd[dt_real, :dual_area] / sd[t, :area]
    end

    return wedge_terms
end

"""    dec_p_wedge_product(::Type{Tuple{0,2}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type

Precomputes values for the wedge product between a 0 and 2-form.
The values are to be fed into the wedge_terms parameter for the computational "c" varient.
This relies on the assumption of a well ordering of the dual space simplices.
Do NOT modify the mesh once it's dual mesh has been computed else this method may not function properly.
"""
function dec_p_wedge_product(::Type{Tuple{0,2}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type
    # XXX: This is assuming that meshes don't have too many entries
    # TODO: This type should be settable by the user and default set to Int32
    primal_vertices = Array{Int32}(undef, 6, ntriangles(sd))
    coeffs = Array{float_type}(undef, 6, ntriangles(sd))

    shift::Int = ntriangles(sd)

    @inbounds for primal_tri in triangles(sd)
      for dual_tri_idx in 1:6
        dual_tri_real = primal_tri + (dual_tri_idx - 1) * shift

        primal_vertices[dual_tri_idx, primal_tri] = sd[sd[dual_tri_real, :dual_∂e2], :dual_∂v1]
        coeffs[dual_tri_idx, primal_tri] = sd[dual_tri_real, :dual_area] / sd[primal_tri, :area]
      end
  end
  (verts, coeffs, ntriangles(sd))
end

function wedge_kernel_coeffs(::Type{Tuple{1,1}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p}) where {float_type, _p}
  coeffs = Array{float_type}(undef, 3, ntriangles(sd))
  shift = ntriangles(sd)
  @inbounds for i in 1:ntriangles(sd)
    area = 2 * sd[i, :area]
    coeffs[1, i] = (sd[i+0*shift, :dual_area] + sd[i+1*shift, :dual_area]) / area
    coeffs[2, i] = (sd[i+2*shift, :dual_area] + sd[i+3*shift, :dual_area]) / area
    coeffs[3, i] = (sd[i+4*shift, :dual_area] + sd[i+5*shift, :dual_area]) / area
  end
  e = Array{Int32}(undef, 3, ntriangles(sd))
  e[1, :], e[2, :], e[3, :] = ∂(2, 0, sd), ∂(2, 1, sd), ∂(2, 2, sd)
  (e, coeffs, ntriangles(sd))
end

# Grab the float type of the volumes of the complex.
function cache_wedge(::Type{Tuple{m,n}}, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p}, backend, arr_cons=identity, cast_float=nothing) where {float_type,_p,m,n}
  cache_wedge(m, n, sd, float_type, arr_cons, cast_float)
end
function cache_wedge(::Type{Tuple{m,n}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p}, backend, arr_cons=identity, cast_float=nothing) where {float_type,_p,m,n}
  cache_wedge(m, n, sd, float_type, arr_cons, cast_float)
end
# Grab wedge kernel coeffs and cast.
function cache_wedge(m::Int, n::Int, sd::HasDeltaSet1D, float_type::DataType, arr_cons, cast_float::Union{Nothing, DataType})
  ft = isnothing(cast_float) ? float_type : cast_float
  wc = wedge_kernel_coeffs(Tuple{m,n}, sd)
  if wc[2] isa Matrix
    (arr_cons(wc[1]), arr_cons(Matrix{ft}(wc[2])), wc[3])
  else
    (arr_cons.(wc[1:end-1])..., wc[end])
  end
end

# XXX: 0.5 implies the dual vertex on an edge is the midpoint.
# TODO: Add options to change 0.5 to a different value.
@kernel function wedge_kernel_01!(res, @Const(f), @Const(α), @Const(p), @Const(simples))
  @uniform half = eltype(f)(0.5)
  i = @index(Global)
  @inbounds res[i] = half * α[i] * (f[p[i, 1]] + f[p[i, 2]])
end

@kernel function wedge_kernel_02!(res, @Const(f), @Const(α), @Const(p), @Const(c))
  i = @index(Global)
  c1, c2, c3, c4, c5, c6 = c[Int32(1),i], c[Int32(2),i], c[Int32(3),i], c[Int32(4),i], c[Int32(5),i], c[Int32(6),i]
  p1, p2, p3, p4, p5, p6 = p[Int32(1),i], p[Int32(2),i], p[Int32(3),i], p[Int32(4),i], p[Int32(5),i], p[Int32(6),i]
  @inbounds res[i] = α[i] * (c1*f[p1] + c2*f[p2] + c3*f[p3] + c4*f[p4] + c5*f[p5] + c6*f[p6])
end

@kernel function wedge_kernel_11!(res, @Const(α), @Const(β), @Const(e), @Const(c))
  i = @index(Global)
  e0, e1, e2 = e[Int32(1), i], e[Int32(2), i], e[Int32(3), i]
  c1, c2, c3 = c[Int32(1), i], c[Int32(2), i], c[Int32(3), i]
  ae0, ae1, ae2 = α[e0], α[e1], α[e2]
  be0, be1, be2 = β[e0], β[e1], β[e2]
 @inbounds res[i] = (c1 * (ae2 * be1 - ae1 * be2) + c2 * (ae2 * be0 - ae0 * be2) + c3 * (ae1 * be0 - ae0 * be1))
end

function auto_select_backend(kernel_function, res, α, β, p, c)
  backend = get_backend(res)
  kernel = kernel_function(backend, backend == CPU() ? 64 : 256)
  kernel(res, α, β, p, c, ndrange=size(res))
  res
end

# Manually dispatch, since CUDA.jl kernels cannot.
# Alternatively, wrap each wedge_kernel separately.
function dec_c_wedge_product!(::Type{Tuple{j,k}}, res, α, β, p, c) where {j,k}
  kernel_function = if (j,k) == (0,1)
    wedge_kernel_01!
  elseif (j,k) == (0,2)
    wedge_kernel_02!
  elseif (j,k) == (1,1)
    wedge_kernel_11!
  else
    error("Unsupported combination of degrees $j and $k. Ensure that their sum is not greater than the degree of the complex, and the degree of the first is ≤ the degree of the second.")
  end
  auto_select_backend(kernel_function, res, α, β, p, c)
end

function dec_c_wedge_product(::Type{Tuple{m,n}}, α, β, wedge_cache) where {m,n}
  α_data = α isa SimplexForm ? α.data : α
  res = KernelAbstractions.zeros(get_backend(α_data), eltype(α_data), last(wedge_cache))
  dec_c_wedge_product!(Tuple{m,n}, res, α, β, wedge_cache[1], wedge_cache[2])
end

"""    dec_wedge_product(::Type{Tuple{m,n}}, sd::HasDeltaSet, backend=Val{:CPU}, arr_cons=identity, cast_float=nothing) where {m,n}

Return a function that computes the wedge product between a primal `m`-form and a primal `n`-form, assuming special properties of the mesh.

It is assumed...
... for the 0-1 wedge product, that the dual vertex on an edge is at the midpoint.
... for the 1-1 wedge product, that the dual mesh simplices are in the default order as returned by the dual complex constructor.

# Arguments:
`Tuple{m,n}`: the degrees of the differential forms.
`sd`: the simplicial complex.
`backend=Val{:CPU}`: a value-type to select special backend logic, if implemented.
`arr_cons=identity`: a constructor of the desired array type on the appropriate backend e.g. `MtlArray`.
`cast_float=nothing`: a specific Float type to use e.g. `Float32`. Otherwise, the type of the first differential form will be used.
"""
function dec_wedge_product(::Type{Tuple{m,n}}, sd::HasDeltaSet, backend=Val{:CPU}, arr_cons=identity, cast_float=nothing) where {m,n}
  error("Unsupported combination of degrees $m and $n. Ensure that their sum is not greater than the degree of the complex.")
end

dec_wedge_product(m::Int, n::Int, sd::HasDeltaSet) =
  dec_wedge_product(Tuple{m,n}, sd::HasDeltaSet)

function dec_wedge_product(::Type{Tuple{0,0}}, sd::HasDeltaSet, backend=Val{:CPU}, arr_cons=identity, cast_float=nothing)
  (f, g) -> f .* g
end

function dec_wedge_product(::Type{Tuple{k,0}}, sd::HasDeltaSet, backend=Val{:CPU}, arr_cons=identity, cast_float=nothing) where {k}
  wedge_cache = cache_wedge(Tuple{0,k}, sd, backend, arr_cons, cast_float)
  (α, β) -> dec_c_wedge_product(Tuple{0,k}, β, α, wedge_cache)
end

function dec_wedge_product(::Type{Tuple{0,k}}, sd::HasDeltaSet, backend=Val{:CPU}, arr_cons=identity, cast_float=nothing) where {k}
  wedge_cache = cache_wedge(Tuple{0,k}, sd, backend, arr_cons, cast_float)
  (α, β) -> dec_c_wedge_product(Tuple{0,k}, α, β, wedge_cache)
end

function dec_wedge_product(::Type{Tuple{1,1}}, sd::HasDeltaSet2D, backend=Val{:CPU}, arr_cons=identity, cast_float=nothing)
  wedge_cache = cache_wedge(Tuple{1,1}, sd, backend, arr_cons, cast_float)
  (α, β) -> dec_c_wedge_product(Tuple{1,1}, α, β, wedge_cache)
end

# Return a matrix that can be multiplied to a dual 0-form, before being
# elementwise-multiplied by a dual 1-form, encoding the wedge product.
function wedge_dd_01_mat(sd::HasDeltaSet)
  m = spzeros(ne(sd), ntriangles(sd))
  for e in edges(sd)
    des = elementary_duals(1,sd,e)
    dvs = sd[des, :dual_∂v0]
    tris = only.(incident(sd, dvs, :tri_center))
    ws = sd[des, :dual_length] ./ sum(sd[des, :dual_length])
    for (w,t) in zip(ws,tris)
      m[e,t] = w
    end
  end
  m
end

"""    dec_wedge_product_dd(::Type{Tuple{m,n}}, sd::HasDeltaSet) where {m,n}

Return a function that computes the wedge product between a dual `m`-form and a dual `n`-form.

The currently supported dual-dual wedges are 0-1 and 1-0.
"""
function dec_wedge_product_dd(::Type{Tuple{m,n}}, sd::HasDeltaSet) where {m,n}
  error("Unsupported combination of degrees $m and $n. Ensure that their sum is not greater than the degree of the complex. The currently supported dual-dual wedges are 0-1 and 1-0.")
end

function dec_wedge_product_dd(::Type{Tuple{0,1}}, sd::HasDeltaSet)
  m = wedge_dd_01_mat(sd)
  (f,g) -> (m * f) .* g
end

function dec_wedge_product_dd(::Type{Tuple{1,0}}, sd::HasDeltaSet)
  m = wedge_dd_01_mat(sd)
  (f,g) -> f .* (m * g)
end

# Return a matrix that can be multiplied to a primal 0-form, before being
# elementwise-multiplied by a dual 1-form, encoding the wedge product.
function wedge_pd_01_mat(sd::HasDeltaSet)
  m = spzeros(ne(sd), nv(sd))
  for e in edges(sd)
    α, β = edge_vertices(sd,e)
    des = elementary_duals(1,sd,e)
    dvs = sd[des, :dual_∂v0]
    tris = only.(incident(sd, dvs, :tri_center))
    γδ = map(tris) do t
      only(filter(x -> x ∉ [α,β], triangle_vertices(sd,t)))
    end
    ws = sd[des, :dual_length] ./ sum(sd[des, :dual_length])
    for (w,l) in zip(ws,γδ)
      m[e,α] += w*5/12
      m[e,β] += w*5/12
      m[e,l] += w*2/12
    end
  end
  m
end

"""    dec_wedge_product_dp(::Type{Tuple{m,n}}, sd::HasDeltaSet) where {m,n}

Return a function that computes the wedge product between a dual `m`-form and a primal `n`-form.

It is assumed...
... for the 1-0 and 0-1 wedge product, that means are barycentric, and performs bilinear interpolation. It is not known if this definition has appeared in the literature or any code.

The currently supported dual-primal wedges are 0-1, 1-0, and 1-1.
"""
function dec_wedge_product_dp(::Type{Tuple{m,n}}, sd::HasDeltaSet) where {m,n}
  error("Unsupported combination of degrees $m and $n. Ensure that their sum is not greater than the degree of the complex. The currently supported dual-primal wedges are 0-1, 1-0, and 1-1.")
end

"""    dec_wedge_product_pd(::Type{Tuple{m,n}}, sd::HasDeltaSet) where {m,n}

Return a function that computes the wedge product between a primal `m`-form and a dual `n`-form.

See [`dec_wedge_product_dp`](@ref) for assumptions.
"""
function dec_wedge_product_pd(::Type{Tuple{m,n}}, sd::HasDeltaSet) where {m,n}
  error("Unsupported combination of degrees $m and $n. Ensure that their sum is not greater than the degree of the complex. The currently supported primal-dual wedges are 0-1, 1-0, and 1-1.")
end

function dec_wedge_product_dp(::Type{Tuple{1,0}}, sd::HasDeltaSet)
  m = wedge_pd_01_mat(sd)
  (f,g) -> f .* (m * g)
end

function dec_wedge_product_pd(::Type{Tuple{0,1}}, sd::HasDeltaSet)
  m = wedge_pd_01_mat(sd)
  (g,f) -> (m * g) .* f
end

function dec_wedge_product_pd(::Type{Tuple{1,1}}, sd::HasDeltaSet)
  ♭♯_m = ♭♯_mat(sd)
  Λ_cached = dec_wedge_product(Tuple{1, 1}, sd)
  (f, g) -> Λ_cached(f, ♭♯_m * g)
end

function dec_wedge_product_dp(::Type{Tuple{1,1}}, sd::HasDeltaSet)
  ♭♯_m = ♭♯_mat(sd)
  Λ_cached = dec_wedge_product(Tuple{1, 1}, sd)
  (f, g) -> Λ_cached(♭♯_m * f, g)
end

"""    ∧(s::HasDeltaSet, α::SimplexForm{1}, β::DualForm{1})

Wedge product of a primal 1-form and a dual 1-form.

Chain the musical isomorphisms to interpolate the dual 1-form to a primal
1-form, using the linear least squares ♯. Then use the CombinatorialSpaces
version of the Hirani primal-primal weddge.
"""
∧(s::HasDeltaSet, α::SimplexForm{1}, β::DualForm{1}) =
  dec_wedge_product_pd(Tuple{1,1}, s)(α, β)

"""    ∧(s::HasDeltaSet, α::DualForm{1}, β::SimplexForm{1})

Wedge product of a dual 1-form and a primal 1-form.

Chain the musical isomorphisms to interpolate the dual 1-form to a primal
1-form. Then use the CombinatorialSpaces version of the Hirani primal-primal
weddge (without explicitly dividing by 2.)
"""
∧(s::HasDeltaSet, α::DualForm{1}, β::SimplexForm{1}) =
  dec_wedge_product_dp(Tuple{1,1}, s)(α, β)


# Boundary and Co-boundary
#-------------------------

"""    dec_boundary(n::Int, sd::HasDeltaSet)

Return the boundary operator (as a matrix) for `(n+1)`-simplices to `(n)`-simplices
"""
dec_boundary(n::Int, sd::HasDeltaSet) = sparse(dec_p_boundary(Val{n}, sd)...)

dec_p_boundary(::Type{Val{k}}, sd::HasDeltaSet; negate::Bool=false) where {k} =
  dec_p_derivbound(Val{k - 1}, sd, transpose=true, negate=negate)

"""    dec_dual_derivative(n::Int, sd::HasDeltaSet)

Return the dual exterior derivative (as a matrix) between dual `n`-simplices and dual `(n+1)`-simplices
"""
dec_dual_derivative(n::Int, sd::HasDeltaSet) = sparse(dec_p_dual_derivative(Val{n}, sd)...)

dec_p_dual_derivative(::Type{Val{0}}, sd::HasDeltaSet1D) =
  dec_p_boundary(Val{1}, sd, negate=true)

dec_p_dual_derivative(::Type{Val{0}}, sd::HasDeltaSet2D) =
  dec_p_boundary(Val{2}, sd)

dec_p_dual_derivative(::Type{Val{1}}, sd::HasDeltaSet2D) =
  dec_p_boundary(Val{1}, sd, negate=true)

"""    dec_differential(n::Int, sd::HasDeltaSet)

Return the exterior derivative (as a matrix) between `n`-simplices and `(n+1)`-simplices
"""
dec_differential(n::Int, sd::HasDeltaSet) = sparse(dec_p_derivbound(Val{n}, sd)...)

function dec_p_derivbound(::Type{Val{0}}, sd::HasDeltaSet; transpose::Bool=false, negate::Bool=false)
  vec_size = 2 * ne(sd)
  I = Vector{Int32}(undef, vec_size)
  J = Vector{Int32}(undef, vec_size)
  V = Vector{Int8}(undef, vec_size)
  for i in edges(sd)
    j = 2 * i - 1

    I[j], I[j+1] = i, i

    J[j], J[j+1] = sd[i, :∂v0], sd[i, :∂v1]

    sign_term = numeric_sign(sd[i, :edge_orientation]::Bool)

    V[j], V[j+1] = sign_term, -1*sign_term
  end
  if (transpose)
    I, J = J, I
  end
  if (negate)
    V .= -1 .* V
  end
  (I, J, V)
end

function dec_p_derivbound(::Type{Val{1}}, sd::HasDeltaSet; transpose::Bool=false, negate::Bool=false)
  vec_size = 3 * ntriangles(sd)
  I = Vector{Int32}(undef, vec_size)
  J = Vector{Int32}(undef, vec_size)
  V = Vector{Int8}(undef, vec_size)
  for i in triangles(sd)
    j = 3 * i - 2

    I[j], I[j+1], I[j+2] = i, i, i

    J[j], J[j+1], J[j+2] = sd[i, :∂e0], sd[i, :∂e1], sd[i, :∂e2]

    e0_sign = numeric_sign(sd[sd[i, :∂e0], :edge_orientation]::Bool)
    e1_sign = numeric_sign(sd[sd[i, :∂e1], :edge_orientation]::Bool)
    e2_sign = numeric_sign(sd[sd[i, :∂e2], :edge_orientation]::Bool)
    t_sign = numeric_sign(sd[i, :tri_orientation]::Bool)

    V[j] = e0_sign * t_sign
    V[j+1] = -1 * e1_sign * t_sign
    V[j+2] = e2_sign * t_sign
  end
  if (transpose)
      I, J = J, I
  end
  if (negate)
      V .= -1 .* V
  end
  (I, J, V)
end

# Diagonal Hodge Star
#--------------------

function dec_p_hodge_diag(::Type{Val{0}}, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p) where float_type
    num_v_sd = nv(sd)

    hodge_diag_0 = zeros(float_type, num_v_sd)

    for d_edge_idx in parts(sd, :DualE)
      v1 = sd[d_edge_idx, :dual_∂v1]
      if (1 <= v1 <= num_v_sd)
          hodge_diag_0[v1] += sd[d_edge_idx, :dual_length]
      end
    end
  end
  h_0
end

function dec_p_hodge_diag(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p) where float_type
  vols::Vector{float_type} = volume(Val{1}, sd, edges(sd))
  1 ./ vols
end


function dec_p_hodge_diag(::Type{Val{0}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type
    hodge_diag_0 = zeros(float_type, nv(sd))

    for dual_tri in parts(sd, :DualTri)
      v = sd[sd[dual_tri, :dual_∂e1], :dual_∂v1]
      hodge_diag_0[v] += sd[dual_tri, :dual_area]
    end
    return hodge_diag_0
end

function dec_p_hodge_diag(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type
    num_v_sd = nv(sd)
    num_e_sd = ne(sd)

    hodge_diag_1 = zeros(float_type, num_e_sd)

    for d_edge_idx in parts(sd, :DualE)
      v1_shift = sd[d_edge_idx, :dual_∂v1] - num_v_sd
      if (1 <= v1_shift <= num_e_sd)
          hodge_diag_1[v1_shift] += sd[d_edge_idx, :dual_length] / sd[v1_shift, :length]
      end
    end
  end
  h_1
end

function dec_p_hodge_diag(::Type{Val{2}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type
  signed_tri_areas::Vector{float_type} = sd[:area] .* sign(2,sd)
  1 ./ signed_tri_areas
end

"""    dec_hodge_star(n::Int, sd::HasDeltaSet; hodge=GeometricHodge())

Return the hodge matrix between `n`-simplices and dual 'n'-simplices.
"""
dec_hodge_star(n::Int, sd::HasDeltaSet; hodge=GeometricHodge()) =
  dec_hodge_star(Val{n}, sd, hodge)
dec_hodge_star(n::Int, sd::HasDeltaSet, ::DiagonalHodge) =
  dec_hodge_star(Val{n}, sd, DiagonalHodge())
dec_hodge_star(n::Int, sd::HasDeltaSet, ::GeometricHodge) =
  dec_hodge_star(Val{n}, sd, GeometricHodge())
dec_hodge_star(::Type{Val{k}}, sd::HasDeltaSet, ::DiagonalHodge) where {k} =
  Diagonal(dec_p_hodge_diag(Val{k}, sd))

# Geometric Hodge Star
#---------------------

# TODO: Still need better implementation for Hodge 1 in 2D
dec_hodge_star(::Type{Val{j}}, sd::EmbeddedDeltaDualComplex1D, ::GeometricHodge) where {j} =
  dec_hodge_star(Val{j}, sd, DiagonalHodge())

dec_hodge_star(::Type{Val{j}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge) where {j} =
  dec_hodge_star(Val{j}, sd, DiagonalHodge())

function dec_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, point_type}, ::GeometricHodge) where {float_type, point_type}
  I = Vector{Int32}(undef, ntriangles(sd) * 9)
  J = Vector{Int32}(undef, ntriangles(sd) * 9)
  V = Vector{float_type}(undef, ntriangles(sd) * 9)

  # Reversed by contruction
  tri_edges_1 = @view sd[:∂e2]
  tri_edges_2 = @view sd[:∂e1]
  tri_edges_3 = @view sd[:∂e0]
  tri_edges = [tri_edges_1, tri_edges_2, tri_edges_3]

  evt = MVector{3, point_type}(undef)
  dvt = MVector{3, point_type}(undef)

  idx = 0

  @inbounds for t in triangles(sd)
    dual_point_tct = sd[sd[t, :tri_center], :dual_point]
    for i in 1:3
      tri_edge = tri_edges[i][t]
      evt[i] = sd[sd[tri_edge, :∂v0], :dual_point] - sd[sd[tri_edge, :∂v1], :dual_point]
      dvt[i] = dual_point_tct - sd[sd[tri_edge, :edge_center], :dual_point]
    end
    dvt[2] *= -1

    # This relative orientation needs to be redefined for each triangle in the
    # case that the mesh has multiple independent connected components
    cross_ev_dv = cross(evt[1], dvt[1])
    rel_orient = (last(cross_ev_dv) == 0 ? 1 : sign(last(cross_ev_dv)))
    for i in 1:3
      diag_cross = crossdot(evt[i], dvt[i])
      if diag_cross != 0.0
        idx += 1
        I[idx] = tri_edges[i][t]
        J[idx] = tri_edges[i][t]
        V[idx] = rel_orient * diag_cross / dot(evt[i], evt[i])
      end
    end

    for p ∈ ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))
      diag_dot = dot(evt[p[1]], dvt[p[1]]) / dot(evt[p[1]], evt[p[1]])
      val = diag_dot * dot(evt[p[1]], evt[p[3]])
      if val != 0.0
        idx += 1
        I[idx] = tri_edges[p[1]][t]
        J[idx] = tri_edges[p[2]][t]
        V[idx] = rel_orient * val / crossdot(evt[p[2]], evt[p[3]])
      end
    end
  end

  view_I = @view I[1:idx]
  view_J = @view J[1:idx]
  view_V = @view V[1:idx]
  sparse(view_I, view_J, view_V)
end

# Inverse Hodge Star
#-------------------

"""    dec_inv_hodge_star(n::Int, sd::HasDeltaSet; hodge=GeometricHodge())

Return the inverse hodge matrix between dual `n`-simplices and 'n'-simplices.
"""
dec_inv_hodge_star(n::Int, sd::HasDeltaSet; hodge=GeometricHodge()) =
  dec_inv_hodge_star(Val{n}, sd, hodge)
dec_inv_hodge_star(n::Int, sd::HasDeltaSet, ::DiagonalHodge) =
  dec_inv_hodge_star(Val{n}, sd, DiagonalHodge())
dec_inv_hodge_star(n::Int, sd::HasDeltaSet, ::GeometricHodge) =
  dec_inv_hodge_star(Val{n}, sd, GeometricHodge())

function dec_inv_hodge_star(::Type{Val{k}}, sd::HasDeltaSet, ::DiagonalHodge) where {k}
  hdg = dec_p_hodge_diag(Val{k}, sd)
  mult_term = iseven(k * (ndims(sd) - k)) ? 1 : -1
  hdg .= (1 ./ hdg) .* mult_term
  Diagonal(hdg)
end

dec_inv_hodge_star(::Type{Val{j}}, sd::EmbeddedDeltaDualComplex1D, ::GeometricHodge) where {j} =
  dec_inv_hodge_star(Val{j}, sd, DiagonalHodge())

dec_inv_hodge_star(::Type{Val{j}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge) where {j} =
  dec_inv_hodge_star(Val{j}, sd, DiagonalHodge())

function dec_inv_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge)
  hdg_lu = factorize(-1 * dec_hodge_star(1, sd, GeometricHodge()))
  x -> hdg_lu \ x
end

# Interior Product and Lie Derivative
#------------------------------------

"""    function interior_product_dd(::Type{Tuple{1,1}}, s::SimplicialSets.HasDeltaSet)

Given a dual 1-form and a dual 1-form, return their interior product as a dual 0-form.
"""
function interior_product_dd(::Type{Tuple{1,1}}, s::SimplicialSets.HasDeltaSet)
  ihs1 = dec_inv_hodge_star(Val{1}, s, GeometricHodge())
  Λ11 = dec_wedge_product_pd(Tuple{1,1}, s)
  hs2 = dec_hodge_star(Val{2}, s, GeometricHodge())
  (f,g) -> hs2 * Λ11(ihs1(g), f)
end

"""    function interior_product_dd(::Type{Tuple{1,1}}, s::SimplicialSets.HasDeltaSet)

Given a dual 1-form and a dual 2-form, return their interior product as a dual 1-form.
"""
function interior_product_dd(::Type{Tuple{1,2}}, s::SimplicialSets.HasDeltaSet)
  ihs0 = dec_inv_hodge_star(Val{0}, s, GeometricHodge())
  hs1 = dec_hodge_star(Val{1}, s, GeometricHodge())
  ♭♯_m = ♭♯_mat(s)
  Λ01_m = wedge_pd_01_mat(s)
  (f,g) -> hs1 * ♭♯_m * ((Λ01_m * ihs0 * g) .* f)
end

"""    function ℒ_dd(::Type{Tuple{1,1}}, s::SimplicialSets.HasDeltaSet)

Given a dual 1-form and a dual 1-form, return their lie derivative as a dual 1-form.
"""
function ℒ_dd(::Type{Tuple{1,1}}, s::SimplicialSets.HasDeltaSet)
  # ℒ := -diuv - iduv
  d0 = dec_dual_derivative(0, s)
  d1 = dec_dual_derivative(1, s)
  i1 = interior_product_dd(Tuple{1,1}, s)
  i2 = interior_product_dd(Tuple{1,2}, s)
  (f,g) -> -(d0 * i1(f,g)) - i2(f,d1 * g)
end

const lie_derivative_dd = ℒ_dd

# Laplacian
#----------

"""    function Δᵈ_mat(::Type{Val{0}}, s::SimplicialSets.HasDeltaSet)

Return a function matrix encoding the dual 0-form Laplacian.
"""
function Δᵈ(::Type{Val{0}}, s::SimplicialSets.HasDeltaSet)
  dd0 = dec_dual_derivative(0, s);
  ihs1 = dec_inv_hodge_star(1, s, GeometricHodge());
  d1 = dec_differential(1,s);
  hs2 = dec_hodge_star(2, s, GeometricHodge());
  m = hs2 * d1
  x -> hs2 * d1 * ihs1(dd0 * x)
end

"""    function Δᵈ_mat(::Type{Val{2}}, s::SimplicialSets.HasDeltaSet)

Return a function matrix encoding the dual 1-form Laplacian.
"""
function Δᵈ(::Type{Val{1}}, s::SimplicialSets.HasDeltaSet)
  dd0 = dec_dual_derivative(0, s);
  ihs1 = dec_inv_hodge_star(1, s, GeometricHodge());
  d1 = dec_differential(1,s);
  hs2 = dec_hodge_star(2, s, GeometricHodge());
  dd1 = dec_dual_derivative(1, s);
  ihs0 = dec_inv_hodge_star(0, s, GeometricHodge());
  d0 = dec_differential(0,s);
  hs1 = dec_hodge_star(1, s, GeometricHodge());
  m = hs1 * d0 * ihs0 * dd1
  n = dd0 * hs2 * d1
  x -> begin
    m * x +
    n * ihs1(x)
  end
end

# Average Operator
#-----------------

function avg₀₁_mat(s::HasDeltaSet, float_type)
  d0 = dec_differential(0,s)
  avg_mat = SparseMatrixCSC{float_type, Int32}(d0)
  avg_mat.nzval .= 0.5
  avg_mat
end

"""    Averaging matrix from 0-forms to 1-forms.

Given a 0-form, this matrix computes a 1-form by taking the mean of value stored on the faces of each edge.

This matrix can be used to implement a wedge product: `(avg₀₁(s)*X) .* Y` where `X` is a 0-form and `Y` a 1-form, assuming the center of an edge is halfway between its endpoints.

See also [`avg₀₁`](@ref).
"""
avg₀₁_mat(s::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p) where float_type =
  avg₀₁_mat(s, float_type)
avg₀₁_mat(s::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p) where float_type =
  avg₀₁_mat(s, float_type)

"""    avg₀₁(s::HasDeltaSet, α::SimplexForm{0})

Turn a 0-form into a 1-form by averaging data stored on the face of an edge.

See also [`avg₀₁_mat`](@ref).
"""
avg₀₁(s::HasDeltaSet, α::SimplexForm{0}) = avg₀₁_mat(s) * α

"""    Alias for the averaging operator [`avg₀₁`](@ref).
"""
const avg_01 = avg₀₁

"""    Alias for the averaging matrix [`avg₀₁_mat`](@ref).
"""
const avg_01_mat = avg₀₁_mat

end
