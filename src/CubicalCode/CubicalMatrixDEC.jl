function exterior_derivative(::Val{0}, s::HasCubicalComplex)

  tot = 2 * ne(s)
  I, J = zeros(Float64, tot), zeros(Float64, tot)
  V = zeros(Float64, tot)

  for e in edges(s)
    idx = 2 * e - 1
    v0, v1 = tgt(s, e), src(s, e)

    I[idx] = e
    I[idx + 1] = e

    J[idx] = v0
    J[idx + 1] = v1

    V[idx] = 1
    V[idx + 1] = -1
  end

  return sparse(I, J, V)
end

function exterior_derivative(::Val{1}, s::HasCubicalComplex)

  tot = 4 * nquads(s)
  I, J = zeros(Float64, tot), zeros(Float64, tot)
  V = zeros(Float64, tot)

  orients = SVector(1,-1,-1,1)
  for q in quadrilaterals(s)
    idx = 4 * q - 3
    for (i, e) in enumerate(quad_edges(s, q))
      j = idx + i - 1
      I[j] = q
      J[j] = e
      V[j] = orients[i]
    end
  end

  return sparse(I, J, V)
end

dual_derivative(::Val{0}, s::HasCubicalComplex) = -transpose(exterior_derivative(Val(1), s))
dual_derivative(::Val{1}, s::HasCubicalComplex) = transpose(exterior_derivative(Val(0), s))

function wedge_product(::Val{(0,1)}, s::HasCubicalComplex, f, alpha)
  res = similar(alpha)

  for e in edges(s)
    res[e] = 0.5 * (getindex(f, src(s, e)) + getindex(f, tgt(s, e))) * alpha[e]
  end

  return res
end

function wedge_product(::Val{(0,2)}, s::HasCubicalComplex, f, alpha)
  res = similar(alpha)

  for q in quadrilaterals(s)
    v1, v2, v3, v4 = quad_vertices(s, q)
    res[q] = 0.25 * (getindex(f, v1) + getindex(f, v2) + getindex(f, v3) + getindex(f, v4)) * alpha[q]
  end

  return res
end

function wedge_product(::Val{(1,1)}, s::HasCubicalComplex, alpha, beta)
  res = zeros(eltype(alpha), nquads(s))

  # a ∧ b = (a[x]b[y] - a[y]b[x])dx ∧ dy
  for q in quadrilaterals(s)
    e1, e2, e3, e4 = quad_edges(s, q)
    res[q] = 0.25 * ((alpha[e1] + alpha[e2]) * (beta[e3] + beta[e4]) - (alpha[e3] + alpha[e4]) * (beta[e1] + beta[e2]))
  end

  return res
end

hodge_star(::Val{0}, s::HasCubicalComplex) = spdiagm(map(dq -> dual_quad_area(s, dq), vertices(s)))

function hodge_star(::Val{1}, s::HasCubicalComplex)
  e_lens = map(e -> edge_length(s, e), edges(s))
  de_lens = map(de -> dual_edge_length(s, de), edges(s))
  return spdiagm(de_lens ./ e_lens)
end

hodge_star(::Val{2}, s::HasCubicalComplex) = spdiagm(1 ./ map(q -> quad_area(s, q), quadrilaterals(s)))

inv_hodge_star(::Val{k}, s::HasCubicalComplex) where k = spdiagm(1 ./ diag(hodge_star(Val(k), s)))

codifferential(::Val{1}, s::HasCubicalComplex) = inv_hodge_star(Val(0), s) * dual_derivative(Val(1), s) * hodge_star(Val(1), s)
codifferential(::Val{2}, s::HasCubicalComplex) = inv_hodge_star(Val(1), s) * dual_derivative(Val(0), s) * hodge_star(Val(2), s)

laplacian(::Val{0}, s::HasCubicalComplex) = codifferential(Val(1), s) * exterior_derivative(Val(0), s)
laplacian(::Val{1}, s::HasCubicalComplex) = exterior_derivative(Val(0), s) * codifferential(Val(1), s) + codifferential(Val(2), s) * exterior_derivative(Val(1), s) 
laplacian(::Val{2}, s::HasCubicalComplex) = exterior_derivative(Val(1), s) * codifferential(Val(2), s)

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

# Take dual 1-form and output dual vector field
function sharp_dd(s::HasCubicalComplex, alpha)
  X, Y = zeros(nquads(s)), zeros(nquads(s))
  for q in quadrilaterals(s)
    deb, det, del, der = quad_edges(s, q)

    X[q] = 0.5 * (alpha[del] / dual_edge_length(s, del) + alpha[der] / dual_edge_length(s, der))
    Y[q] = -0.5 * (alpha[deb] / dual_edge_length(s, deb) + alpha[det] / dual_edge_length(s, det))
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

# Take dual vector field and output primal 1-form
# TODO: Might need to weight by dimensions of quad
function flat_dp(s::HasCubicalComplex, X, Y)
  alpha = zeros(eltype(X), nde(s))

  for q in quadrilaterals(s)
    h = quad_height(s, q)
    w = quad_width(s, q)

    eb, et, el, er = quad_edges(s, q)
    alpha[el] += 0.5 * h * Y[q]
    alpha[er] += 0.5 * h * Y[q]

    alpha[eb] += 0.5 * w * X[q]
    alpha[et] += 0.5 * w * X[q]
  end

  return alpha
end