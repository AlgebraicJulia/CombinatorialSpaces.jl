function exterior_derivative(::Val{0}, s::HasCubicalComplex)

  tot = 2 * ne(s)
  I, J = zeros(Float64, tot), zeros(Float64, tot)
  V = zeros(Float64, tot)

  for coord_e in edges(s)
    e = coord_to_edge(s, coord_e...)
    idx = 2 * e - 1
    v0, v1 = coord_to_vert(s, tgt(coord_e...)...), coord_to_vert(s, src(coord_e...)...)

    I[idx] = e
    I[idx + 1] = e

    J[idx] = v0
    J[idx + 1] = v1

    V[idx] = 1
    V[idx + 1] = -1
  end

  return sparse(I, J, V)
end

# TODO: Convert to new functions
function exterior_derivative(::Val{1}, s::EmbeddedCubicalComplex2D)

  tot = 4 * nquads(s)
  I, J = zeros(Float64, tot), zeros(Float64, tot)
  V = zeros(Float64, tot)

  orients = SVector(1,-1,-1,1)
  for coord_q in quadrilaterals(s)
    q = coord_to_quad(s, coord_q...)
    idx = 4 * q - 3
    for (i, e) in enumerate(quad_edges(coord_q...))
      j = idx + i - 1
      I[j] = q
      z = i == 1 || i == 2 ? 1 : 2
      J[j] = coord_to_edge(s, z, e...)
      V[j] = orients[i]
    end
  end

  return sparse(I, J, V)
end

dual_derivative(::Val{0}, s::HasCubicalComplex) = -transpose(exterior_derivative(Val(1), s))
dual_derivative(::Val{1}, s::HasCubicalComplex) = transpose(exterior_derivative(Val(0), s))

hodge_star(::Val{0}, s::HasCubicalComplex) = spdiagm(map(dq -> d_quad_area(s, dq...), vertices(s)))

function hodge_star(::Val{1}, s::HasCubicalComplex)
  e_lens = map(e -> edge_len(s, e...), edges(s))
  de_lens = map(de -> d_edge_len(s, de...), edges(s))
  return spdiagm(de_lens ./ e_lens)
end

hodge_star(::Val{2}, s::HasCubicalComplex) = spdiagm(1 ./ map(q -> quad_area(s, q...), quadrilaterals(s)))

inv_hodge_star(::Val{k}, s::HasCubicalComplex) where k = spdiagm(1 ./ diag(hodge_star(Val(k), s)))

codifferential(::Val{1}, s::HasCubicalComplex) = inv_hodge_star(Val(0), s) * dual_derivative(Val(1), s) * hodge_star(Val(1), s)
codifferential(::Val{2}, s::HasCubicalComplex) = inv_hodge_star(Val(1), s) * dual_derivative(Val(0), s) * hodge_star(Val(2), s)

laplacian(::Val{0}, s::HasCubicalComplex) = codifferential(Val(1), s) * exterior_derivative(Val(0), s)
laplacian(::Val{1}, s::HasCubicalComplex) = exterior_derivative(Val(0), s) * codifferential(Val(1), s) + codifferential(Val(2), s) * exterior_derivative(Val(1), s) 
laplacian(::Val{2}, s::HasCubicalComplex) = exterior_derivative(Val(1), s) * codifferential(Val(2), s)

# TODO: Convert these to kernels
# Take primal 1-form and output dual vector field
function sharp_pd(s::HasCubicalComplex, alpha)
  X, Y = zeros(nquads(s)), zeros(nquads(s))
  for q in quadrilaterals(s)
    eb, et, el, er = quad_edges(s, q)
    X[q] = 0.5 * (alpha[eb] + alpha[et]) / quad_width(s, q)
    Y[q] = 0.5 * (alpha[el] + alpha[er]) / quad_height(s, q)
  end

  return X, Y
end

# Take dual vector field and output dual 1-form
function flat_dd(s::HasCubicalComplex, X, Y)
  alpha = zeros(eltype(X), nde(s))

  for q in quadrilaterals(s)
    h = quad_height(q, s)
    w = quad_width(q, s)

    deb, det, del, der = quad_edges(s, q)
    alpha[deb] += 0.5 * h * Y[q]
    alpha[det] += 0.5 * h * Y[q]

    alpha[del] += 0.5 * w * X[q]
    alpha[der] += 0.5 * w * X[q]
  end

  return alpha
end