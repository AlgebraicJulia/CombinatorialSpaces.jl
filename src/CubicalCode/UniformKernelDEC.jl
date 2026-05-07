using KernelAbstractions

# Kernel for exterior derivative of 0-forms (primal 0-forms to primal 1-forms)
@kernel function kernel_exterior_derivative_zero!(res, s, @Const(f))
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  @inbounds res[idx] = f[tgt(s, x, y, align)] - f[src(s, x, y, align)]
end

# Kernel for exterior derivative of 1-forms (primal 1-forms to primal 2-forms)
# quad_edges returns (bottom-x, right-y, top-x, left-y); signs match the
# matrix orientation (+1, +1, -1, -1) = bottom + right - top - left.
@kernel function kernel_exterior_derivative_one!(res, s, @Const(f))
  idx = @index(Global)
  x, y = quad_to_coord(s, idx)

  e1, e2, e3, e4 = quad_edges(s, x, y)
  @inbounds res[idx] = f[e1] + f[e2] - f[e3] - f[e4]
end

# Main interface functions

function exterior_derivative!(res, ::Val{0}, s::UniformCubicalComplex2D, f)
  backend = get_backend(f)
  kernel = kernel_exterior_derivative_zero!(backend)
  kernel(res, s, f; ndrange = size(res))
  return res
end

function exterior_derivative!(res, ::Val{1}, s::UniformCubicalComplex2D, f)
  backend = get_backend(f)
  kernel = kernel_exterior_derivative_one!(backend)
  kernel(res, s, f; ndrange = size(res))
  return res
end

# Kernel for wedge 01, 11, and dual 01 products

@kernel function kernel_wedge_product_01(res, s, @Const(f), @Const(a))
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  # Get the values of the 0-form at the vertices of the edge
  v1 = f[src(s, x, y, align)]
  v2 = f[tgt(s, x, y, align)]

  @inbounds res[idx] = (v1 + v2) * a[idx] / 2
end

@kernel function kernel_wedge_product_11(res, s, @Const(a), @Const(b))
  idx = @index(Global)
  x, y = quad_to_coord(s, idx)

  es = quad_edges(s, x, y)

  @inbounds res[idx] = 0.25 * (a[es[1]] + a[es[3]]) * (b[es[2]] + b[es[4]]) -
    0.25 * (a[es[2]] + a[es[4]]) * (b[es[1]] + b[es[3]])
end

# TODO: Remove control flow to make this more efficient on the GPU
@kernel function kernel_wedge_product_dual_01(res, s, @Const(f), @Const(a))
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  dv1, dv2 = edge_quads(s, x, y, align)

  ae = a[idx]

  if is_left_edge(s, x, y, align)
    tmp = f[dv2] * ae
  elseif is_right_edge(s, x, y, align)
    tmp = f[dv1] * ae
  elseif is_top_edge(s, x, y, align)
    tmp = f[dv1] * ae
  elseif is_bottom_edge(s, x, y, align)
    tmp = f[dv2] * ae
  else
    tmp = 0.5 * (f[dv1] + f[dv2]) * ae
  end

  @inbounds res[idx] = tmp
end

function wedge_product(::Val{0}, ::Val{1}, s::UniformCubicalComplex2D, f::AbstractVector{FT}, a::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, Float64, ne(s))
  kernel = kernel_wedge_product_01(backend)

  kernel(res, s, f, a; ndrange = size(res))
  return res
end

wedge_product(::Val{1}, ::Val{0}, s::UniformCubicalComplex2D, a::AbstractVector{FT}, f::AbstractVector{FT}) where FT <: AbstractFloat =
  wedge_product(Val(0), Val(1), s, f, a)

function wedge_product(::Val{1}, ::Val{1}, s::UniformCubicalComplex2D, f1::AbstractVector{FT}, f2::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f1)
  res = KernelAbstractions.zeros(backend, Float64, nquads(s))
  kernel = kernel_wedge_product_11(backend)

  kernel(res, s, f1, f2; ndrange = size(res))
  return res
end

function wedge_product_dd(::Val{0}, ::Val{1}, s::UniformCubicalComplex2D, f::AbstractVector{FT}, a::AbstractVector{FT}) where FT <: AbstractFloat
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, Float64, ne(s))
  kernel = kernel_wedge_product_dual_01(backend)

  kernel(res, s, f, a; ndrange = size(res))
  return res
end

wedge_product_dd(::Val{1}, ::Val{0}, s::UniformCubicalComplex2D, a, f) = wedge_product_dd(Val(0), Val(1), s, f, a)


# Convert a dual 1-form to a vector field on the dual points
@kernel function kernel_sharp_dd(X, Y, s, @Const(a))
  idx = @index(Global)
  x, y = quad_to_coord(s, idx)
  e1, e2, e3, e4 = quad_edges(s, x, y) # Order is (x-, y-, x+, y+)

  le1 = dual_edge_len(s, e1)
  le2 = dual_edge_len(s, e2)
  le3 = dual_edge_len(s, e3)
  le4 = dual_edge_len(s, e4)

  # Remember that for an X-aligned primal edge, we have a Y-aligned dual edge
  X[idx] = -(a[e2]/le2 + a[e4]/le4) / 2
  Y[idx] = (a[e1]/le1 + a[e3]/le3) / 2
end

function sharp_dd(s::UniformCubicalComplex2D, a::AbstractVector{FT}) where FT <:AbstractFloat
  backend = get_backend(a)
  X = KernelAbstractions.zeros(backend, Float64, nquads(s))
  Y = KernelAbstractions.zeros(backend, Float64, nquads(s))
  kernel = kernel_sharp_dd(backend)

  kernel(X, Y, s, a; ndrange = size(X))
  return X, Y
end

# Flat, convert dual vector field to primal 1-form
@kernel function kernel_flat_dp(res, s, @Const(X), @Const(Y))
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)
  if align == X_ALIGN
    if y == 1
      res[idx] = X[coord_to_quad(s,x,y)] * edge_len(s, X_ALIGN)
    elseif y == ny(s)
      res[idx] = X[coord_to_quad(s,x,y-1)] * edge_len(s, X_ALIGN)
    else
      res[idx] = 0.5 * (X[coord_to_quad(s,x,y)] + X[coord_to_quad(s,x,y-1)]) * edge_len(s, X_ALIGN)
    end
  else
    if x == 1
      res[idx] = Y[coord_to_quad(s,x,y)] * edge_len(s, Y_ALIGN)
    elseif x == nx(s)
      res[idx] = Y[coord_to_quad(s,x-1,y)] * edge_len(s, Y_ALIGN)
    else
      res[idx] = 0.5 * (Y[coord_to_quad(s,x,y)] + Y[coord_to_quad(s,x-1,y)]) * edge_len(s, Y_ALIGN)
    end
  end
end

function flat_dp(s::UniformCubicalComplex2D, X::AbstractVector{FT}, Y::AbstractVector{FT}) where FT <:AbstractFloat
  backend = get_backend(X)
  res = KernelAbstractions.zeros(backend, Float64, ne(s))
  kernel = kernel_flat_dp(backend)

  kernel(res, s, X, Y; ndrange = size(res))
  return res
end

# Flat, convert dual vector field to dual 1-form
@kernel function kernel_flat_dd(res, s, @Const(X), @Const(Y))
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  if align == X_ALIGN
    if y == 1
      res[idx] = Y[coord_to_quad(s, x, y)] * dual_edge_len(s, idx)
    elseif y == ny(s)
      res[idx] = Y[coord_to_quad(s, x, y - 1)] * dual_edge_len(s, idx)
    else
      res[idx] = 0.5 * (Y[coord_to_quad(s, x, y)] + Y[coord_to_quad(s, x, y - 1)]) * dual_edge_len(s, idx)
    end
  else
    if x == 1
      res[idx] = -X[coord_to_quad(s, x, y)] * dual_edge_len(s, idx)
    elseif x == nx(s)
      res[idx] = -X[coord_to_quad(s, x - 1, y)] * dual_edge_len(s, idx)
    else
      res[idx] = -0.5 * (X[coord_to_quad(s, x, y)] + X[coord_to_quad(s, x - 1, y)]) * dual_edge_len(s, idx)
    end
  end
end

function flat_dd(s::UniformCubicalComplex2D, X::AbstractVector{FT}, Y::AbstractVector{FT}) where FT <:AbstractFloat
  backend = get_backend(X)
  res = KernelAbstractions.zeros(backend, Float64, ne(s))
  kernel = kernel_flat_dd(backend)

  kernel(res, s, X, Y; ndrange = size(res))
  return res
end

# TODO: Why is there scalar indexing in this function? Can we remove it to make it more efficient on the GPU?
function interpolate_dp(::Val{1}, s::UniformCubicalComplex2D, a)
  backend = get_backend(a)
  X, Y = sharp_dd(s, a)
  synchronize(backend)
  return flat_dp(s, X, Y)
end

# ═══════════════════════════════════════════════════════════════════════════
#  set_periodic! – KernelAbstractions versions
# ═══════════════════════════════════════════════════════════════════════════
#
# Each kernel iterates over the halo cells on one pair of opposing sides
# (east/west or north/south).  A single linear index maps to an (i, j)
# pair within the halo strip; each thread copies from the matching interior
# cell on the opposite side of the domain.

@enum GridSide EASTWEST NORTHSOUTH ALL

# ── Val{0}  (0-forms on vertices) ────────────────────────────────────────

@kernel function kernel_set_periodic_0_ew!(f, s)
  idx = @index(Global)
  hx_ = hx(s); nx_ = nx(s); ny_ = ny(s)

  j = div(idx - 1, hx_ + 1) + 1
  i = idx - (j - 1) * (hx_ + 1)

  # Right halo ← left interior
  @inbounds f[coord_to_vert(s, nx_ - hx_ + i - 1, j)] = f[coord_to_vert(s, hx_ + i, j)]

  # Left halo ← right interior
  @inbounds f[coord_to_vert(s, i, j)] = f[coord_to_vert(s, nx_ - 2hx_ + i - 1, j)]
end

@kernel function kernel_set_periodic_0_ns!(f, s)
  idx = @index(Global)
  hy_ = hy(s); nx_ = nx(s); ny_ = ny(s)
  j = div(idx - 1, nx_) + 1
  i = idx - (j - 1) * nx_

  # Top halo ← bottom interior
  # Correct, map first vertex to last vertex continue rightwards to fill halo
  @inbounds f[coord_to_vert(s, i, ny_ - hy_ + j - 1)] = f[coord_to_vert(s, i, hy_ + j)]

  # Bottom halo ← top interior
  @inbounds f[coord_to_vert(s, i, j)] = f[coord_to_vert(s, i, ny_ - 2hy_ + j - 1)]
end

function set_periodic!(f::AbstractVector{FT}, ::Val{0}, s::UniformCubicalComplex2D, side::GridSide) where FT <: AbstractFloat
  backend = get_backend(f)
  if side == EASTWEST || side == ALL
    kernel_set_periodic_0_ew!(backend)(f, s; ndrange = (hx(s) + 1) * ny(s))
  end
  if side == NORTHSOUTH || side == ALL
    kernel_set_periodic_0_ns!(backend)(f, s; ndrange = nx(s) * (hy(s) + 1))
  end
  return f
end

# ── Val{1}  (1-forms on edges) ───────────────────────────────────────────
#
# Edges come in two families (X-aligned and Y-aligned) with different grid
# dimensions, so the EW and NS kernels each handle both families in turn.

# East/West halo for x-edges:  strip is hx × ny(s)
@kernel function kernel_set_periodic_1_ew_x!(f, s)
  idx = @index(Global)
  hx_ = hx(s); nxe_ = nxe(s); nyv_ = ny(s)
  j = div(idx - 1, hx_) + 1
  i = idx - (j - 1) * hx_
  @inbounds f[coord_to_edge(s, nxe_ - hx_ + i, j, X_ALIGN)] = f[coord_to_edge(s, hx_ + i, j, X_ALIGN)]
  @inbounds f[coord_to_edge(s, i, j, X_ALIGN)] = f[coord_to_edge(s, nxe_ - 2hx_ + i, j, X_ALIGN)]
end

# East/West halo for y-edges:  strip is hx × nye(s)
@kernel function kernel_set_periodic_1_ew_y!(f, s)
  idx = @index(Global)
  hx_ = hx(s); nye_ = nye(s); nxv_ = nx(s)
  j = div(idx - 1, hx_ + 1) + 1
  i = idx - (j - 1) * (hx_ + 1)

  @inbounds f[coord_to_edge(s, nxv_ - hx_ + i - 1, j, Y_ALIGN)] = f[coord_to_edge(s, hx_ + i, j, Y_ALIGN)]
  @inbounds f[coord_to_edge(s, i, j, Y_ALIGN)] = f[coord_to_edge(s, nxv_ - 2hx_ + i - 1, j, Y_ALIGN)]
end

# North/South halo for x-edges:  strip is nxe(s) × hy
@kernel function kernel_set_periodic_1_ns_x!(f, s)
  idx = @index(Global)
  hy_ = hy(s); nxe_ = nxe(s); nyv_ = ny(s)
  j = div(idx - 1, nxe_) + 1
  i = idx - (j - 1) * nxe_
  @inbounds f[coord_to_edge(s, i, nyv_ - hy_ + j - 1, X_ALIGN)] = f[coord_to_edge(s, i, hy_ + j, X_ALIGN)]
  @inbounds f[coord_to_edge(s, i, j, X_ALIGN)] = f[coord_to_edge(s, i, nyv_ - 2hy_ + j - 1, X_ALIGN)]
end

# North/South halo for y-edges:  strip is nx(s) × hy
@kernel function kernel_set_periodic_1_ns_y!(f, s)
  idx = @index(Global)
  hy_ = hy(s); nxv_ = nx(s); nye_ = nye(s)
  j = div(idx - 1, nxv_) + 1
  i = idx - (j - 1) * nxv_
  @inbounds f[coord_to_edge(s, i, j, Y_ALIGN)] = f[coord_to_edge(s, i, nye_ - 2hy_ + j, Y_ALIGN)]
  @inbounds f[coord_to_edge(s, i, nye_ - hy_ + j, Y_ALIGN)] = f[coord_to_edge(s, i, hy_ + j, Y_ALIGN)]
end

function set_periodic!(f::AbstractVector{FT}, ::Val{1}, s::UniformCubicalComplex2D, side::GridSide) where FT <: AbstractFloat
  backend = get_backend(f)
  if side == EASTWEST || side == ALL
    kernel_set_periodic_1_ew_x!(backend)(f, s; ndrange = hx(s) * ny(s))
    kernel_set_periodic_1_ew_y!(backend)(f, s; ndrange = (hx(s) + 1) * nye(s))
  end
  if side == NORTHSOUTH || side == ALL
    kernel_set_periodic_1_ns_x!(backend)(f, s; ndrange = nxe(s) * (hy(s) + 1))
    kernel_set_periodic_1_ns_y!(backend)(f, s; ndrange = nx(s) * hy(s))
  end
  return f
end

# ── Val{2}  (2-forms on quads) ───────────────────────────────────────────

@kernel function kernel_set_periodic_2_ew!(f, s)
  idx = @index(Global)
  hx_ = hx(s); nqx_ = nxquads(s); nqy_ = nyquads(s)
  j = div(idx - 1, hx_) + 1
  i = idx - (j - 1) * hx_
  @inbounds f[coord_to_quad(s, i, j)] = f[coord_to_quad(s, nqx_ - 2hx_ + i, j)]
  @inbounds f[coord_to_quad(s, nqx_ - hx_ + i, j)] = f[coord_to_quad(s, hx_ + i, j)]
end

@kernel function kernel_set_periodic_2_ns!(f, s)
  idx = @index(Global)
  hy_ = hy(s); nqx_ = nxquads(s); nqy_ = nyquads(s)
  j = div(idx - 1, nqx_) + 1
  i = idx - (j - 1) * nqx_
  @inbounds f[coord_to_quad(s, i, j)] = f[coord_to_quad(s, i, nqy_ - 2hy_ + j)]
  @inbounds f[coord_to_quad(s, i, nqy_ - hy_ + j)] = f[coord_to_quad(s, i, hy_ + j)]
end

function set_periodic!(f::AbstractVector{FT}, ::Val{2}, s::UniformCubicalComplex2D, side::GridSide) where FT <: AbstractFloat
  backend = get_backend(f)
  if side == EASTWEST || side == ALL
    kernel_set_periodic_2_ew!(backend)(f, s; ndrange = hx(s) * nyquads(s))
  end
  if side == NORTHSOUTH || side == ALL
    kernel_set_periodic_2_ns!(backend)(f, s; ndrange = nxquads(s) * hy(s))
  end
  return f
end

# ═══════════════════════════════════════════════════════════════════════════
#  UniformDECCache – precomputed look-up tables for branchless, fast kernels
# ═══════════════════════════════════════════════════════════════════════════
#
# Building the cache from a mesh performs all integer-division-heavy
# coordinate arithmetic once, on the CPU.  The resulting index arrays are
# passed directly to kernels that need nothing but array reads — no
# branching, no `div`/`mod`, and no repeated floating-point reciprocals.
#
# GPU usage: adapt each field to the device before the first kernel call,
# e.g. via `Adapt.adapt(backend, cache)` with Adapt.jl + CUDA.jl.
#
# Memory note: Int32 indices halve the bandwidth cost of the look-up arrays
# on both CPU and GPU relative to the default Int64.

struct UniformDECCache{IT <: AbstractVector{Int32}, FT <: AbstractVector{Float64}}
  # ── Mesh dimensions (for allocation inside interface functions) ──────────
  nv_      :: Int
  ne_      :: Int
  nquads_  :: Int
  nxedges_ :: Int
  nyedges_ :: Int

  # ── d0 / wedge_01: src & tgt vertex for every edge (length ne) ──────────
  src_v :: IT
  tgt_v :: IT

  # ── d1 / wedge_11 / sharp_dd: 4 edge indices per quad (length nquads) ───
  # Order: (bottom-x, right-y, top-x, left-y) — matches quad_edges(s, x, y)
  q_e1 :: IT
  q_e2 :: IT
  q_e3 :: IT
  q_e4 :: IT

  # ── sharp_dd: 1/dual_edge_len for each quad's 4 edges (length nquads) ───
  inv_de1 :: FT
  inv_de2 :: FT
  inv_de3 :: FT
  inv_de4 :: FT

  # ── wedge_dd_01: two adjacent quads per edge (length ne) ────────────────
  # Boundary edges have e_q1[e] == e_q2[e] (the single valid adjacent quad
  # is stored twice), so the branchless formula
  #   0.5 * (f[e_q1[e]] + f[e_q2[e]]) * a[e]
  # correctly reduces to f[q] * a[e] for boundary edges.
  e_q1 :: IT
  e_q2 :: IT

  # ── flat_dp: adjacent quad pairs, split by edge family ──────────────────
  # X-aligned edges → interpolate X component; q1 == q2 for boundary edges.
  # Kernel: res[e] = (X[fp_xq1[e]] + X[fp_xq2[e]]) * fp_dx_half
  fp_xq1 :: IT   # length nxedges
  fp_xq2 :: IT
  # Y-aligned edges → interpolate Y component
  fp_yq1 :: IT   # length nyedges
  fp_yq2 :: IT
  fp_dx_half :: Float64   # = 0.5 * dx  (constant for uniform grid)
  fp_dy_half :: Float64   # = 0.5 * dy

  # ── flat_dd: adjacent quad pairs + per-edge signed scale ────────────────
  # X-aligned edges → use Y component with positive sign.
  # Kernel: res[e] = (Y[fd_xq1[e]] + Y[fd_xq2[e]]) * fd_xscale[e]
  # scale = 0.5 * dual_edge_len(e) for all edges — works uniformly for both
  # interior (dual_edge_len = full, q1 ≠ q2) and boundary (dual_edge_len
  # = half, q1 == q2, so the doubled sum is compensated by the halved scale).
  fd_xq1    :: IT   # length nxedges
  fd_xq2    :: IT
  fd_xscale :: FT
  # Y-aligned edges → use X component with negative sign (baked into scale).
  fd_yq1    :: IT   # length nyedges
  fd_yq2    :: IT
  fd_yscale :: FT   # negative values

  # ── hodge_star / inv_hodge_star: per-element scale vectors ──────────────
  # hs0_scale[v]  = dual_quad_area(v)          (interior: dx*dy, boundary: half/quarter)
  # hs1_scale[e]  = dual_edge_len(e)/edge_len(e)
  # hs2_val       = 1/(dx*dy)                  (uniform scalar)
  # ihs0_scale[v] = 1/dual_quad_area(v)
  # ihs1_scale[e] = -edge_len(e)/dual_edge_len(e)  (negative: matches matrix convention)
  # ihs2_val      = dx*dy                      (uniform scalar)
  hs0_scale  :: FT   # length nv
  hs1_scale  :: FT   # length ne
  hs2_val    :: Float64
  ihs0_scale :: FT   # length nv
  ihs1_scale     :: FT   # length ne  — values are negative
  ihs1_hs2_scale :: FT   # length ne  — ihs1_scale[e] * hs2_val (precomputed product)
  ihs2_val       :: Float64

  # ── dd0 = d1^T: for each edge e, res[e] = f[q_pos] - f[q_neg] ───────────
  # Positive/negative quads follow the orientation of d1:
  #   X-edge at (x,y): q_pos = quad(x,y) [above], q_neg = quad(x,y-1) [below]
  #   Y-edge at (x,y): q_pos = quad(x-1,y) [left], q_neg = quad(x,y) [right]
  # emask: bit 0 = positive quad exists, bit 1 = negative quad exists.
  # Boundary edges have one bit clear; the corresponding index is a dummy.
  # ifelse(Bool(bit), f[q_pos], 0.0) - ifelse(Bool(bit), f[q_neg], 0.0)
  dd0_qp    :: IT   # length ne
  dd0_qn    :: IT   # length ne
  dd0_emask :: IT   # length ne — Int32 bitmask

  # ── dd1 = -d0^T: for each vertex v, ────────────────────────────────────
  # res[v] = +a[x_src] + a[y_src] - a[x_tgt] - a[y_tgt]
  # where x_src is the X-edge starting at v (v = src), etc.
  # vmask: bit k = the k-th edge slot exists (0=vxs, 1=vys, 2=vxt, 3=vyt).
  # Boundary vertices have some bits clear; their index slots hold dummies.
  dd1_vxs   :: IT   # length nv — X-edge where v is src
  dd1_vys   :: IT   # length nv — Y-edge where v is src
  dd1_vxt   :: IT   # length nv — X-edge where v is tgt
  dd1_vyt   :: IT   # length nv — Y-edge where v is tgt
  dd1_vmask :: IT   # length nv — Int32 bitmask
end

function UniformDECCache(s::UniformCubicalComplex2D{FT}) where {FT <: AbstractFloat}
  ne_  = ne(s);     nq_  = nquads(s)
  nxe_ = nxedges(s); nye_ = nyedges(s)
  nx_  = nx(s);     ny_  = ny(s)
  nv_  = nv(s)

  # ── src / tgt vertices per edge ─────────────────────────────────────────
  src_v = Vector{Int32}(undef, ne_)
  tgt_v = Vector{Int32}(undef, ne_)
  for e in 1:ne_
    src_v[e] = src(s, e)
    tgt_v[e] = tgt(s, e)
  end

  # ── 4 edge indices + reciprocal dual-edge lengths per quad ──────────────
  q_e1 = Vector{Int32}(undef, nq_);  q_e2 = Vector{Int32}(undef, nq_)
  q_e3 = Vector{Int32}(undef, nq_);  q_e4 = Vector{Int32}(undef, nq_)
  inv_de1 = Vector{FT}(undef, nq_);  inv_de2 = Vector{FT}(undef, nq_)
  inv_de3 = Vector{FT}(undef, nq_);  inv_de4 = Vector{FT}(undef, nq_)
  for q in 1:nq_
    x, y = quad_to_coord(s, q)
    e1, e2, e3, e4 = quad_edges(s, x, y)
    q_e1[q] = e1;  q_e2[q] = e2;  q_e3[q] = e3;  q_e4[q] = e4
    inv_de1[q] = inv(dual_edge_len(s, e1));  inv_de2[q] = inv(dual_edge_len(s, e2))
    inv_de3[q] = inv(dual_edge_len(s, e3));  inv_de4[q] = inv(dual_edge_len(s, e4))
  end

  # ── two adjacent quads per edge (for wedge_dd_01) ───────────────────────
  e_q1 = Vector{Int32}(undef, ne_)
  e_q2 = Vector{Int32}(undef, ne_)
  for e in 1:ne_
    x, y, align = edge_to_coord(s, e)
    if align == X_ALIGN
      if y == 1
        e_q1[e] = e_q2[e] = coord_to_quad(s, x, y)
      elseif y == ny_
        e_q1[e] = e_q2[e] = coord_to_quad(s, x, y - 1)
      else
        e_q1[e] = coord_to_quad(s, x, y - 1)
        e_q2[e] = coord_to_quad(s, x, y)
      end
    else  # Y_ALIGN
      if x == 1
        e_q1[e] = e_q2[e] = coord_to_quad(s, x, y)
      elseif x == nx_
        e_q1[e] = e_q2[e] = coord_to_quad(s, x - 1, y)
      else
        e_q1[e] = coord_to_quad(s, x - 1, y)
        e_q2[e] = coord_to_quad(s, x, y)
      end
    end
  end

  # ── flat_dp quad pairs ───────────────────────────────────────────────────
  fp_xq1 = Vector{Int32}(undef, nxe_);  fp_xq2 = Vector{Int32}(undef, nxe_)
  for e in 1:nxe_
    x, y, _ = edge_to_coord(s, e)
    if y == 1
      fp_xq1[e] = fp_xq2[e] = coord_to_quad(s, x, y)
    elseif y == ny_
      fp_xq1[e] = fp_xq2[e] = coord_to_quad(s, x, y - 1)
    else
      fp_xq1[e] = coord_to_quad(s, x, y - 1)   # quad below
      fp_xq2[e] = coord_to_quad(s, x, y)        # quad above
    end
  end
  fp_yq1 = Vector{Int32}(undef, nye_);  fp_yq2 = Vector{Int32}(undef, nye_)
  for (i, e) in enumerate(nxe_ + 1 : ne_)
    x, y, _ = edge_to_coord(s, e)
    if x == 1
      fp_yq1[i] = fp_yq2[i] = coord_to_quad(s, x, y)
    elseif x == nx_
      fp_yq1[i] = fp_yq2[i] = coord_to_quad(s, x - 1, y)
    else
      fp_yq1[i] = coord_to_quad(s, x - 1, y)   # quad to left
      fp_yq2[i] = coord_to_quad(s, x, y)        # quad to right
    end
  end

  # ── flat_dd quad pairs + per-edge scale ─────────────────────────────────
  # scale = 0.5 * dual_edge_len(e) for every edge:
  #   interior: dual_edge_len = full, q1 ≠ q2 → (f[q1]+f[q2]) * 0.5*full ✓
  #   boundary: dual_edge_len = half, q1 == q2 → (f[q]+f[q]) * 0.5*half
  #                                             = f[q] * half ✓
  fd_xq1 = Vector{Int32}(undef, nxe_);  fd_xq2 = Vector{Int32}(undef, nxe_)
  fd_xscale = Vector{FT}(undef, nxe_)
  for e in 1:nxe_
    x, y, _ = edge_to_coord(s, e)
    fd_xscale[e] = 0.5 * dual_edge_len(s, e)
    if y == 1
      fd_xq1[e] = fd_xq2[e] = coord_to_quad(s, x, y)
    elseif y == ny_
      fd_xq1[e] = fd_xq2[e] = coord_to_quad(s, x, y - 1)
    else
      fd_xq1[e] = coord_to_quad(s, x, y - 1)
      fd_xq2[e] = coord_to_quad(s, x, y)
    end
  end
  fd_yq1 = Vector{Int32}(undef, nye_);  fd_yq2 = Vector{Int32}(undef, nye_)
  fd_yscale = Vector{FT}(undef, nye_)
  for (i, e) in enumerate(nxe_ + 1 : ne_)
    x, y, _ = edge_to_coord(s, e)
    fd_yscale[i] = -0.5 * dual_edge_len(s, e)   # negative: Y-edges use -X
    if x == 1
      fd_yq1[i] = fd_yq2[i] = coord_to_quad(s, x, y)
    elseif x == nx_
      fd_yq1[i] = fd_yq2[i] = coord_to_quad(s, x - 1, y)
    else
      fd_yq1[i] = coord_to_quad(s, x - 1, y)
      fd_yq2[i] = coord_to_quad(s, x, y)
    end
  end

  # ── hodge star scales ────────────────────────────────────────────────────
  hs0_scale  = Vector{FT}(undef, nv_)
  ihs0_scale = Vector{FT}(undef, nv_)
  for v in 1:nv_
    dqa = dual_quad_area(s, v)
    hs0_scale[v]  = dqa
    ihs0_scale[v] = inv(dqa)
  end

  hs1_scale  = Vector{FT}(undef, ne_)
  ihs1_scale = Vector{FT}(undef, ne_)
  for e in 1:ne_
    elen  = edge_len(s, e)
    delen = dual_edge_len(s, e)
    hs1_scale[e]  = delen / elen
    ihs1_scale[e] = -elen / delen   # negative: matches matrix inv_hodge_star(Val(1)) sign
  end

  hs2_val        = inv(quad_area(s))
  ihs2_val       = quad_area(s)
  ihs1_hs2_scale = ihs1_scale .* hs2_val

  # ── dd0 = d1^T: per-edge positive/negative quad + bitmask ───────────────
  # X-edge at (x, y): pos = quad(x, y) [above], neg = quad(x, y-1) [below]
  # Y-edge at (x, y): pos = quad(x-1, y) [left], neg = quad(x, y) [right]
  dd0_qp    = Vector{Int32}(undef, ne_)
  dd0_qn    = Vector{Int32}(undef, ne_)
  dd0_emask = Vector{Int32}(undef, ne_)
  for e in 1:ne_
    x, y, align = edge_to_coord(s, e)
    if align == X_ALIGN
      has_pos = (y <= ny_ - 1)
      has_neg = (y >= 2)
      dd0_qp[e] = has_pos ? coord_to_quad(s, x, y)     : Int32(1)
      dd0_qn[e] = has_neg ? coord_to_quad(s, x, y - 1) : Int32(1)
    else  # Y_ALIGN
      has_pos = (x >= 2)
      has_neg = (x <= nx_ - 1)
      dd0_qp[e] = has_pos ? coord_to_quad(s, x - 1, y) : Int32(1)
      dd0_qn[e] = has_neg ? coord_to_quad(s, x, y)     : Int32(1)
    end
    dd0_emask[e] = Int32(has_pos) | (Int32(has_neg) << 1)
  end

  # ── dd1 = -d0^T: per-vertex 4 incident edges + bitmask ──────────────────
  # For vertex v at (vx, vy):
  #   vxs = X-edge(vx, vy)   if vx < nx  (v is src → contributes +a[e])
  #   vys = Y-edge(vx, vy)   if vy < ny  (v is src → contributes +a[e])
  #   vxt = X-edge(vx-1, vy) if vx > 1   (v is tgt → contributes -a[e])
  #   vyt = Y-edge(vx, vy-1) if vy > 1   (v is tgt → contributes -a[e])
  dd1_vxs   = Vector{Int32}(undef, nv_)
  dd1_vys   = Vector{Int32}(undef, nv_)
  dd1_vxt   = Vector{Int32}(undef, nv_)
  dd1_vyt   = Vector{Int32}(undef, nv_)
  dd1_vmask = Vector{Int32}(undef, nv_)
  DUMMY_EDGE = Int32(1)
  for v in 1:nv_
    vx, vy = vert_to_coord(s, v)
    has_vxs = (vx <  nx_);  has_vys = (vy <  ny_)
    has_vxt = (vx >  1);    has_vyt = (vy >  1)
    dd1_vxs[v]   = has_vxs ? coord_to_edge(s, vx, vy, X_ALIGN)      : DUMMY_EDGE
    dd1_vys[v]   = has_vys ? coord_to_edge(s, vx, vy, Y_ALIGN)      : DUMMY_EDGE
    dd1_vxt[v]   = has_vxt ? coord_to_edge(s, vx - 1, vy, X_ALIGN)  : DUMMY_EDGE
    dd1_vyt[v]   = has_vyt ? coord_to_edge(s, vx, vy - 1, Y_ALIGN)  : DUMMY_EDGE
    dd1_vmask[v] = Int32(has_vxs)       | (Int32(has_vys) << 1) |
                   (Int32(has_vxt) << 2) | (Int32(has_vyt) << 3)
  end

  return UniformDECCache(
    nv_, ne_, nq_, nxe_, nye_,
    src_v, tgt_v,
    q_e1, q_e2, q_e3, q_e4,
    inv_de1, inv_de2, inv_de3, inv_de4,
    e_q1, e_q2,
    fp_xq1, fp_xq2, fp_yq1, fp_yq2, 0.5 * dx(s), 0.5 * dy(s),
    fd_xq1, fd_xq2, fd_xscale,
    fd_yq1, fd_yq2, fd_yscale,
    hs0_scale, hs1_scale, hs2_val,
    ihs0_scale, ihs1_scale, ihs1_hs2_scale, ihs2_val,
    dd0_qp, dd0_qn, dd0_emask,
    dd1_vxs, dd1_vys, dd1_vxt, dd1_vyt, dd1_vmask,
  )
end

# ═══════════════════════════════════════════════════════════════════════════
#  Cached kernels – no integer division, no control flow
# ═══════════════════════════════════════════════════════════════════════════

# ── d0 ───────────────────────────────────────────────────────────────────
@kernel function kernel_d0_cached!(res, @Const(src_v), @Const(tgt_v), @Const(f))
  e = @index(Global)
  @inbounds res[e] = f[tgt_v[e]] - f[src_v[e]]
end

# ── d1: signs (+1,+1,-1,-1) match quad_edges order (bottom,right,top,left)
@kernel function kernel_d1_cached!(res, @Const(q_e1), @Const(q_e2), @Const(q_e3), @Const(q_e4), @Const(f))
  q = @index(Global)
  @inbounds res[q] = f[q_e1[q]] + f[q_e2[q]] - f[q_e3[q]] - f[q_e4[q]]
end

# ── wedge 0∧1 ────────────────────────────────────────────────────────────
@kernel function kernel_wedge_01_cached!(res, @Const(src_v), @Const(tgt_v), @Const(f), @Const(a))
  e = @index(Global)
  @inbounds res[e] = (f[src_v[e]] + f[tgt_v[e]]) * a[e] * 0.5
end

# ── wedge 1∧1 ────────────────────────────────────────────────────────────
@kernel function kernel_wedge_11_cached!(res, @Const(q_e1), @Const(q_e2), @Const(q_e3), @Const(q_e4), @Const(a), @Const(b))
  q = @index(Global)
  @inbounds begin
    ae1 = a[q_e1[q]];  ae2 = a[q_e2[q]];  ae3 = a[q_e3[q]];  ae4 = a[q_e4[q]]
    be1 = b[q_e1[q]];  be2 = b[q_e2[q]];  be3 = b[q_e3[q]];  be4 = b[q_e4[q]]
    res[q] = ((ae1 + ae3) * (be2 + be4) - (ae2 + ae4) * (be1 + be3)) * 0.25
  end
end

# ── dual wedge 0∧1: boundary edges have q1==q2, so no branching needed ──
@kernel function kernel_wedge_dd_01_cached!(res, @Const(e_q1), @Const(e_q2), @Const(f), @Const(a))
  e = @index(Global)
  @inbounds res[e] = (f[e_q1[e]] + f[e_q2[e]]) * a[e] * 0.5
end

# ── sharp_dd: precomputed reciprocals replace per-thread division ─────────
@kernel function kernel_sharp_dd_cached!(X, Y, @Const(q_e1), @Const(q_e2), @Const(q_e3), @Const(q_e4),
                                         @Const(inv_de1), @Const(inv_de2), @Const(inv_de3), @Const(inv_de4), @Const(a))
  q = @index(Global)
  @inbounds begin
    X[q] = -(a[q_e2[q]] * inv_de2[q] + a[q_e4[q]] * inv_de4[q]) * 0.5
    Y[q] =  (a[q_e1[q]] * inv_de1[q] + a[q_e3[q]] * inv_de3[q]) * 0.5
  end
end

# ── flat_dp: split into two branchless launches (x-edges / y-edges) ───────
@kernel function kernel_flat_dp_x_cached!(res, @Const(fp_xq1), @Const(fp_xq2), @Const(X), dx_half)
  e = @index(Global)
  @inbounds res[e] = (X[fp_xq1[e]] + X[fp_xq2[e]]) * dx_half
end

@kernel function kernel_flat_dp_y_cached!(res, @Const(fp_yq1), @Const(fp_yq2), @Const(Y), dy_half, offset)
  i = @index(Global)
  @inbounds res[offset + i] = (Y[fp_yq1[i]] + Y[fp_yq2[i]]) * dy_half
end

# ── interpolate_dp = flat_dp ∘ sharp_dd: fused, no intermediate X/Y ─────
# Inlines the sharp_dd computation directly into flat_dp, avoiding:
#   • two intermediate nquads-length vectors (X, Y)
#   • a synchronize() barrier between the two kernel launches
# For an x-edge, X at each adjacent quad comes from the y-aligned dual edges
# (e2, e4) of that quad. For a y-edge, Y comes from x-aligned dual edges (e1, e3).
# Boundary sentinel: fp_xq1[e] == fp_xq2[e] for boundary x-edges (likewise y),
# so (X1+X2)*dx_half = X1*dx = X1*edge_len, matching the single-quad formula.
@kernel function kernel_interp_dp_x_cached!(res, @Const(fp_xq1), @Const(fp_xq2),
                                             @Const(q_e2), @Const(q_e4), @Const(inv_de2), @Const(inv_de4),
                                             interp_x_scale::Float64, @Const(a))
  e = @index(Global)
  @inbounds begin
    q1 = fp_xq1[e]; q2 = fp_xq2[e]
    res[e] = -(a[q_e2[q1]] * inv_de2[q1] + a[q_e4[q1]] * inv_de4[q1] +
               a[q_e2[q2]] * inv_de2[q2] + a[q_e4[q2]] * inv_de4[q2]) * interp_x_scale
  end
end

@kernel function kernel_interp_dp_y_cached!(res, @Const(fp_yq1), @Const(fp_yq2),
                                             @Const(q_e1), @Const(q_e3), @Const(inv_de1), @Const(inv_de3),
                                             interp_y_scale::Float64, offset, @Const(a))
  i = @index(Global)
  @inbounds begin
    q1 = fp_yq1[i]; q2 = fp_yq2[i]
    res[offset + i] = (a[q_e1[q1]] * inv_de1[q1] + a[q_e3[q1]] * inv_de3[q1] +
                       a[q_e1[q2]] * inv_de1[q2] + a[q_e3[q2]] * inv_de3[q2]) * interp_y_scale
  end
end

# ── flat_dd: split into two branchless launches (x-edges / y-edges) ───────
@kernel function kernel_flat_dd_x_cached!(res, @Const(fd_xq1), @Const(fd_xq2), @Const(fd_xscale), @Const(Y))
  e = @index(Global)
  @inbounds res[e] = (Y[fd_xq1[e]] + Y[fd_xq2[e]]) * fd_xscale[e]
end

@kernel function kernel_flat_dd_y_cached!(res, @Const(fd_yq1), @Const(fd_yq2), @Const(fd_yscale), @Const(X), offset)
  i = @index(Global)
  @inbounds res[offset + i] = (X[fd_yq1[i]] + X[fd_yq2[i]]) * fd_yscale[i]
end

# ── hodge_star: element-wise multiply by a precomputed scale vector ────────
@kernel function kernel_hodge_vec!(res, @Const(scale), @Const(f))
  i = @index(Global)
  @inbounds res[i] = scale[i] * f[i]
end

# ── hodge_star Val(2) / inv_hodge_star Val(2): uniform scalar multiply ─────
@kernel function kernel_hodge_scalar!(res, val::Float64, @Const(f))
  i = @index(Global)
  @inbounds res[i] = val * f[i]
end

# ── dd0 = d1^T: signed gather from two adjacent quads per edge ────────────
# emask bit 0 → positive quad contributes; bit 1 → negative quad contributes.
# ifelse is branchless on both CPU (CMOV) and GPU (predication).
@kernel function kernel_dd0_cached!(res, @Const(dd0_qp), @Const(dd0_qn), @Const(dd0_emask), @Const(f))
  e = @index(Global)
  @inbounds begin
    mask = dd0_emask[e]
    z = zero(eltype(f))
    pos  = ifelse(Bool(mask & Int32(1)),               f[dd0_qp[e]], z)
    neg  = ifelse(Bool((mask >> Int32(1)) & Int32(1)), f[dd0_qn[e]], z)
    res[e] = pos - neg
  end
end

# ── dd1 = -d0^T: signed gather from 4 incident edges per vertex ───────────
# vmask bits 0-3 → vxs, vys, vxt, vyt exist respectively.
@kernel function kernel_dd1_cached!(res, @Const(dd1_vxs), @Const(dd1_vys), @Const(dd1_vxt), @Const(dd1_vyt), @Const(dd1_vmask), @Const(a))
  v = @index(Global)
  @inbounds begin
    mask = dd1_vmask[v]
    z = zero(eltype(a))
    xs = ifelse(Bool(mask & Int32(1)),               a[dd1_vxs[v]], z)
    ys = ifelse(Bool((mask >> Int32(1)) & Int32(1)), a[dd1_vys[v]], z)
    xt = ifelse(Bool((mask >> Int32(2)) & Int32(1)), a[dd1_vxt[v]], z)
    yt = ifelse(Bool((mask >> Int32(3)) & Int32(1)), a[dd1_vyt[v]], z)
    res[v] = xs + ys - xt - yt
  end
end

# ═══════════════════════════════════════════════════════════════════════════
#  Cached interface functions
# ═══════════════════════════════════════════════════════════════════════════

function exterior_derivative!(res, ::Val{0}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_d0_cached!(backend)(res, cache.src_v, cache.tgt_v, f; ndrange = cache.ne_)
  return res
end

function exterior_derivative!(res, ::Val{1}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_d1_cached!(backend)(res, cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4, f;
                              ndrange = cache.nquads_)
  return res
end

function wedge_product(::Val{0}, ::Val{1}, cache::UniformDECCache,
                       f::AbstractVector{FT}, a::AbstractVector{FT}) where FT
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, FT, cache.ne_)
  kernel_wedge_01_cached!(backend)(res, cache.src_v, cache.tgt_v, f, a;
                                   ndrange = cache.ne_)
  return res
end

wedge_product(::Val{1}, ::Val{0}, cache::UniformDECCache, a, f) =
  wedge_product(Val(0), Val(1), cache, f, a)

function wedge_product(::Val{1}, ::Val{1}, cache::UniformDECCache,
                       f1::AbstractVector{FT}, f2::AbstractVector{FT}) where FT
  backend = get_backend(f1)
  res = KernelAbstractions.zeros(backend, FT, cache.nquads_)
  kernel_wedge_11_cached!(backend)(res, cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
                                   f1, f2; ndrange = cache.nquads_)
  return res
end

function wedge_product_dd(::Val{0}, ::Val{1}, cache::UniformDECCache,
                          f::AbstractVector{FT}, a::AbstractVector{FT}) where FT
  backend = get_backend(f)
  res = KernelAbstractions.zeros(backend, FT, cache.ne_)
  kernel_wedge_dd_01_cached!(backend)(res, cache.e_q1, cache.e_q2, f, a;
                                      ndrange = cache.ne_)
  return res
end

wedge_product_dd(::Val{1}, ::Val{0}, cache::UniformDECCache, a, f) =
  wedge_product_dd(Val(0), Val(1), cache, f, a)

function sharp_dd(cache::UniformDECCache, a::AbstractVector{FT}) where FT
  backend = get_backend(a)
  X = KernelAbstractions.zeros(backend, FT, cache.nquads_)
  Y = KernelAbstractions.zeros(backend, FT, cache.nquads_)
  kernel_sharp_dd_cached!(backend)(X, Y,
    cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
    cache.inv_de1, cache.inv_de2, cache.inv_de3, cache.inv_de4,
    a; ndrange = cache.nquads_)
  return X, Y
end

function flat_dp(cache::UniformDECCache, X::AbstractVector{FT}, Y::AbstractVector{FT}) where FT
  backend = get_backend(X)
  res = KernelAbstractions.zeros(backend, FT, cache.ne_)
  kernel_flat_dp_x_cached!(backend)(res, cache.fp_xq1, cache.fp_xq2, X, cache.fp_dx_half;
                                    ndrange = cache.nxedges_)
  kernel_flat_dp_y_cached!(backend)(res, cache.fp_yq1, cache.fp_yq2, Y, cache.fp_dy_half,
                                    cache.nxedges_; ndrange = cache.nyedges_)
  return res
end

function flat_dd(cache::UniformDECCache, X::AbstractVector{FT}, Y::AbstractVector{FT}) where FT
  backend = get_backend(X)
  res = KernelAbstractions.zeros(backend, FT, cache.ne_)
  kernel_flat_dd_x_cached!(backend)(res, cache.fd_xq1, cache.fd_xq2, cache.fd_xscale, Y;
                                    ndrange = cache.nxedges_)
  kernel_flat_dd_y_cached!(backend)(res, cache.fd_yq1, cache.fd_yq2, cache.fd_yscale, X,
                                    cache.nxedges_; ndrange = cache.nyedges_)
  return res
end

 function interpolate_dp!(res, ::Val{1}, cache::UniformDECCache, a)
  backend = get_backend(a)
  kernel_interp_dp_x_cached!(backend)(res,
    cache.fp_xq1, cache.fp_xq2,
    cache.q_e2, cache.q_e4, cache.inv_de2, cache.inv_de4,
    cache.fp_dx_half * Float64(0.5), a; ndrange = cache.nxedges_)
  kernel_interp_dp_y_cached!(backend)(res,
    cache.fp_yq1, cache.fp_yq2,
    cache.q_e1, cache.q_e3, cache.inv_de1, cache.inv_de3,
    cache.fp_dy_half * Float64(0.5), cache.nxedges_, a; ndrange = cache.nyedges_)
  return res
end

function interpolate_dp(::Val{1}, cache::UniformDECCache, a::AbstractVector{FT}) where FT
  backend = get_backend(a)
  res = KernelAbstractions.zeros(backend, FT, cache.ne_)
  return interpolate_dp!(res, Val(1), cache, a)
end

# ── hodge_star (cached) ───────────────────────────────────────────────────
function hodge_star!(res, ::Val{0}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_hodge_vec!(backend)(res, cache.hs0_scale, f; ndrange = cache.nv_)
  return res
end

function hodge_star!(res, ::Val{1}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_hodge_vec!(backend)(res, cache.hs1_scale, f; ndrange = cache.ne_)
  return res
end

function hodge_star!(res, ::Val{2}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_hodge_scalar!(backend)(res, cache.hs2_val, f; ndrange = cache.nquads_)
  return res
end

function hodge_star(::Val{k}, cache::UniformDECCache, f::AbstractVector{FT}) where {k, FT}
  backend = get_backend(f)
  n = k == 0 ? cache.nv_ : k == 1 ? cache.ne_ : cache.nquads_
  res = KernelAbstractions.zeros(backend, FT, n)
  return hodge_star!(res, Val(k), cache, f)
end

# ── inv_hodge_star (cached) ───────────────────────────────────────────────
function inv_hodge_star!(res, ::Val{0}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_hodge_vec!(backend)(res, cache.ihs0_scale, f; ndrange = cache.nv_)
  return res
end

function inv_hodge_star!(res, ::Val{1}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_hodge_vec!(backend)(res, cache.ihs1_scale, f; ndrange = cache.ne_)
  return res
end

function inv_hodge_star!(res, ::Val{2}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_hodge_scalar!(backend)(res, cache.ihs2_val, f; ndrange = cache.nquads_)
  return res
end

function inv_hodge_star(::Val{k}, cache::UniformDECCache, f::AbstractVector{FT}) where {k, FT}
  backend = get_backend(f)
  n = k == 0 ? cache.nv_ : k == 1 ? cache.ne_ : cache.nquads_
  res = KernelAbstractions.zeros(backend, FT, n)
  return inv_hodge_star!(res, Val(k), cache, f)
end

# ── dual_derivative (cached) ──────────────────────────────────────────────
function dual_derivative!(res, ::Val{0}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_dd0_cached!(backend)(res, cache.dd0_qp, cache.dd0_qn, cache.dd0_emask, f;
                               ndrange = cache.ne_)
  return res
end

function dual_derivative!(res, ::Val{1}, cache::UniformDECCache, a)
  backend = get_backend(a)
  kernel_dd1_cached!(backend)(res, cache.dd1_vxs, cache.dd1_vys,
                               cache.dd1_vxt, cache.dd1_vyt, cache.dd1_vmask, a;
                               ndrange = cache.nv_)
  return res
end

function dual_derivative(::Val{k}, cache::UniformDECCache, f::AbstractVector{FT}) where {k, FT}
  backend = get_backend(f)
  n = k == 0 ? cache.ne_ : cache.nv_
  res = KernelAbstractions.zeros(backend, FT, n)
  return dual_derivative!(res, Val(k), cache, f)
end

# ═══════════════════════════════════════════════════════════════════════════
#  Fused codifferential kernels
#
#  Each kernel fuses three operations into a single memory pass, eliminating
#  two intermediate buffers and roughly halving memory traffic vs separate
#  hodge/derivative steps:
#
#    codifferential(1)      = ihs0  ∘ dd1  ∘ hs1   (ne  → nv)
#    codifferential(2)      = ihs1  ∘ dd0  ∘ hs2   (nq  → ne)
#    dual_codifferential(1) = hs2   ∘ d1   ∘ ihs1  (ne  → nq)
#    dual_codifferential(2) = hs1   ∘ d0   ∘ ihs0  (nv  → ne)
# ═══════════════════════════════════════════════════════════════════════════

# ── codifferential(1): ihs0 * dd1 * hs1, thread per vertex ───────────────
# dd1 = -d0^T: edges where v=src contribute +, edges where v=tgt contribute -.
# Each edge contribution is pre-scaled by hs1[e] before the signed sum;
# the whole vertex result is then post-scaled by ihs0[v].
@kernel function kernel_codiff1_cached!(res, @Const(dd1_vxs), @Const(dd1_vys), @Const(dd1_vxt), @Const(dd1_vyt),
                                        @Const(dd1_vmask), @Const(ihs0_scale), @Const(hs1_scale), @Const(a))
  v = @index(Global)
  @inbounds begin
    mask = dd1_vmask[v]
    e_vxs = dd1_vxs[v]; e_vys = dd1_vys[v]; e_vxt = dd1_vxt[v]; e_vyt = dd1_vyt[v]
    xs = ifelse(Bool(mask & Int32(1)),               hs1_scale[e_vxs] * a[e_vxs], zero(eltype(a)))
    ys = ifelse(Bool((mask >> Int32(1)) & Int32(1)), hs1_scale[e_vys] * a[e_vys], zero(eltype(a)))
    xt = ifelse(Bool((mask >> Int32(2)) & Int32(1)), hs1_scale[e_vxt] * a[e_vxt], zero(eltype(a)))
    yt = ifelse(Bool((mask >> Int32(3)) & Int32(1)), hs1_scale[e_vyt] * a[e_vyt], zero(eltype(a)))
    res[v] = ihs0_scale[v] * (xs + ys - xt - yt)
  end
end

# ── codifferential(2): ihs1 * dd0 * hs2, thread per edge ─────────────────
# hs2 is a uniform scalar so it merges with ihs1 into a single per-edge scale.
# dd0 bitmask guards the two adjacent quad lookups.
@kernel function kernel_codiff2_cached!(res, @Const(dd0_qp), @Const(dd0_qn), @Const(dd0_emask),
                                        @Const(ihs1_hs2_scale), @Const(f))
  e = @index(Global)
  @inbounds begin
    mask = dd0_emask[e]
    pos  = ifelse(Bool(mask & Int32(1)),               f[dd0_qp[e]], zero(eltype(f)))
    neg  = ifelse(Bool((mask >> Int32(1)) & Int32(1)), f[dd0_qn[e]], zero(eltype(f)))
    res[e] = ihs1_hs2_scale[e] * (pos - neg)
  end
end

# ── dual_codifferential(1): hs2 * d1 * ihs1, thread per quad ─────────────
# d1 signs: (+1,+1,-1,-1) for (bottom-x, right-y, top-x, left-y).
# ihs1[e] pre-scales each edge contribution; hs2_val is a uniform post-scalar.
@kernel function kernel_dual_codiff1_cached!(res, @Const(q_e1), @Const(q_e2), @Const(q_e3), @Const(q_e4),
                                             @Const(ihs1_hs2_scale), @Const(a))
  q = @index(Global)
  @inbounds begin
    res[q] = ihs1_hs2_scale[q_e1[q]] * a[q_e1[q]] +
             ihs1_hs2_scale[q_e2[q]] * a[q_e2[q]] -
             ihs1_hs2_scale[q_e3[q]] * a[q_e3[q]] -
             ihs1_hs2_scale[q_e4[q]] * a[q_e4[q]]
  end
end

# ── dual_codifferential(2): hs1 * d0 * ihs0, thread per edge ─────────────
# d0: res[e] = ihs0[tgt]*f[tgt] - ihs0[src]*f[src]; post-scaled by hs1[e].
@kernel function kernel_dual_codiff2_cached!(res, @Const(src_v), @Const(tgt_v),
                                             @Const(hs1_scale), @Const(ihs0_scale), @Const(f))
  e = @index(Global)
  @inbounds res[e] = hs1_scale[e] * (ihs0_scale[tgt_v[e]] * f[tgt_v[e]] -
                                      ihs0_scale[src_v[e]] * f[src_v[e]])
end

# ── Codifferential interface functions ────────────────────────────────────
function codifferential!(res, ::Val{1}, cache::UniformDECCache, a)
  backend = get_backend(a)
  kernel_codiff1_cached!(backend)(res,
    cache.dd1_vxs, cache.dd1_vys, cache.dd1_vxt, cache.dd1_vyt, cache.dd1_vmask,
    cache.ihs0_scale, cache.hs1_scale, a; ndrange = cache.nv_)
  return res
end

function codifferential!(res, ::Val{2}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_codiff2_cached!(backend)(res,
    cache.dd0_qp, cache.dd0_qn, cache.dd0_emask,
    cache.ihs1_hs2_scale, f; ndrange = cache.ne_)
  return res
end

function codifferential(::Val{k}, cache::UniformDECCache, f::AbstractVector{FT}) where {k, FT}
  backend = get_backend(f)
  n = k == 1 ? cache.nv_ : cache.ne_
  res = KernelAbstractions.zeros(backend, FT, n)
  return codifferential!(res, Val(k), cache, f)
end

# ── Dual codifferential interface functions ───────────────────────────────
function dual_codifferential!(res, ::Val{1}, cache::UniformDECCache, a)
  backend = get_backend(a)
  kernel_dual_codiff1_cached!(backend)(res,
    cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
    cache.ihs1_hs2_scale, a; ndrange = cache.nquads_)
  return res
end

function dual_codifferential!(res, ::Val{2}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_dual_codiff2_cached!(backend)(res,
    cache.src_v, cache.tgt_v, cache.hs1_scale, cache.ihs0_scale, f;
    ndrange = cache.ne_)
  return res
end

function dual_codifferential(::Val{k}, cache::UniformDECCache, f::AbstractVector{FT}) where {k, FT}
  backend = get_backend(f)
  n = k == 1 ? cache.nquads_ : cache.ne_
  res = KernelAbstractions.zeros(backend, FT, n)
  return dual_codifferential!(res, Val(k), cache, f)
end

# ═══════════════════════════════════════════════════════════════════════════
#  Fused primal Laplacian kernels
#
#  Each kernel is a single-pass fusion of the full Laplacian chain:
#
#    laplacian(0) = codiff1 ∘ d0   = (ihs0 ∘ dd1 ∘ hs1) ∘ d0   (nv → nv)
#    laplacian(1) = d0 ∘ codiff1 + codiff2 ∘ d1                  (ne → ne)
#    laplacian(2) = d1 ∘ codiff2   = d1 ∘ (ihs1 ∘ dd0 ∘ hs2)   (nq → nq)
#
#  Memory saved vs chaining: 2 temporary buffers eliminated per operator.
# ═══════════════════════════════════════════════════════════════════════════

# ── laplacian(0): (ihs0 * dd1 * hs1) * d0, thread per vertex v ───────────
# d0 at the edge incident to v flips sign depending on whether v is src or tgt.
# For all 4 slots the combined result is hs1[e] * (f[neighbor] - f[v]):
#   vxs/vys (v=src): dd1 sign +1, d0 = f[tgt] - f[v]  → +hs1*(f[tgt]-f[v])
#   vxt/vyt (v=tgt): dd1 sign -1, d0 = f[v] - f[src]  → -hs1*(f[v]-f[src]) = +hs1*(f[src]-f[v])
@kernel function kernel_laplacian0_cached!(res, @Const(dd1_vxs), @Const(dd1_vys), @Const(dd1_vxt), @Const(dd1_vyt),
                                           @Const(dd1_vmask), @Const(ihs0_scale), @Const(hs1_scale), @Const(src_v), @Const(tgt_v), @Const(f))
  v = @index(Global)
  @inbounds begin
    mask = dd1_vmask[v]
    vxs = dd1_vxs[v]; vys = dd1_vys[v]; vxt = dd1_vxt[v]; vyt = dd1_vyt[v]
    xs = ifelse(Bool(mask & Int32(1)),             hs1_scale[vxs] * (f[tgt_v[vxs]] - f[v]), zero(eltype(f)))
    ys = ifelse(Bool((mask >> Int32(1)) & Int32(1)), hs1_scale[vys] * (f[tgt_v[vys]] - f[v]), zero(eltype(f)))
    xt = ifelse(Bool((mask >> Int32(2)) & Int32(1)), hs1_scale[vxt] * (f[src_v[vxt]] - f[v]), zero(eltype(f)))
    yt = ifelse(Bool((mask >> Int32(3)) & Int32(1)), hs1_scale[vyt] * (f[src_v[vyt]] - f[v]), zero(eltype(f)))
    res[v] = ihs0_scale[v] * (xs + ys + xt + yt)
  end
end

# ── laplacian(2): d1 * (ihs1 * dd0 * hs2), thread per quad q ─────────────
# For each of q's 4 boundary edges, codiff2[e] = ihs1[e]*hs2*(f[qp]-f[qn]).
# d1 signs (+1,+1,-1,-1) are applied to the 4 codiff2 values.
@kernel function kernel_laplacian2_cached!(res, @Const(q_e1), @Const(q_e2), @Const(q_e3), @Const(q_e4),
                                           @Const(dd0_qp), @Const(dd0_qn), @Const(dd0_emask),
                                           @Const(ihs1_hs2_scale), @Const(f))
  q = @index(Global)
  @inbounds begin
    e1 = q_e1[q]; e2 = q_e2[q]; e3 = q_e3[q]; e4 = q_e4[q]
    m1 = dd0_emask[e1]; m2 = dd0_emask[e2]; m3 = dd0_emask[e3]; m4 = dd0_emask[e4]
    c1 = ihs1_hs2_scale[e1] *
         (ifelse(Bool(m1 & Int32(1)),               f[dd0_qp[e1]], zero(eltype(f))) -
          ifelse(Bool((m1 >> Int32(1)) & Int32(1)), f[dd0_qn[e1]], zero(eltype(f))))
    c2 = ihs1_hs2_scale[e2] *
         (ifelse(Bool(m2 & Int32(1)),               f[dd0_qp[e2]], zero(eltype(f))) -
          ifelse(Bool((m2 >> Int32(1)) & Int32(1)), f[dd0_qn[e2]], zero(eltype(f))))
    c3 = ihs1_hs2_scale[e3] *
         (ifelse(Bool(m3 & Int32(1)),               f[dd0_qp[e3]], zero(eltype(f))) -
          ifelse(Bool((m3 >> Int32(1)) & Int32(1)), f[dd0_qn[e3]], zero(eltype(f))))
    c4 = ihs1_hs2_scale[e4] *
         (ifelse(Bool(m4 & Int32(1)),               f[dd0_qp[e4]], zero(eltype(f))) -
          ifelse(Bool((m4 >> Int32(1)) & Int32(1)), f[dd0_qn[e4]], zero(eltype(f))))
    res[q] = c1 + c2 - c3 - c4
  end
end

# ── laplacian(1): d0*codiff1 + codiff2*d1, thread per edge e ─────────────
# Term 1 — d0 * codiff1 = d0 * (ihs0 * dd1 * hs1):
#   codiff1[v] = ihs0[v] * (hs1[vxs]*f[vxs] + hs1[vys]*f[vys]
#                           - hs1[vxt]*f[vxt] - hs1[vyt]*f[vyt])
#   term1 = codiff1[tgt(e)] - codiff1[src(e)]
# Term 2 — codiff2 * d1 = (ihs1 * dd0 * hs2) * d1:
#   (d1*f)[q] = f[q_e1]+f[q_e2]-f[q_e3]-f[q_e4]
#   term2 = ihs1[e] * hs2 * ((d1*f)[qp(e)] - (d1*f)[qn(e)])
# ifelse is branchless; dummy index 1 is a safe fallback for missing elements.
@kernel function kernel_laplacian1_cached!(res, @Const(src_v), @Const(tgt_v),
                                           @Const(dd1_vxs), @Const(dd1_vys), @Const(dd1_vxt), @Const(dd1_vyt), @Const(dd1_vmask),
                                           @Const(q_e1), @Const(q_e2), @Const(q_e3), @Const(q_e4),
                                           @Const(dd0_qp), @Const(dd0_qn), @Const(dd0_emask),
                                           @Const(ihs0_scale), @Const(hs1_scale), @Const(ihs1_hs2_scale), @Const(f))
  e = @index(Global)
  @inbounds begin
    # ── Term 1 ─────────────────────────────────────────────────────────────
    sv = src_v[e]; tv = tgt_v[e]

    mask_sv = dd1_vmask[sv]
    svxs = dd1_vxs[sv]; svys = dd1_vys[sv]; svxt = dd1_vxt[sv]; svyt = dd1_vyt[sv]
    codiff1_sv = ihs0_scale[sv] * (
      ifelse(Bool(mask_sv & Int32(1)),             hs1_scale[svxs] * f[svxs], zero(eltype(f))) +
      ifelse(Bool((mask_sv >> Int32(1)) & Int32(1)), hs1_scale[svys] * f[svys], zero(eltype(f))) -
      ifelse(Bool((mask_sv >> Int32(2)) & Int32(1)), hs1_scale[svxt] * f[svxt], zero(eltype(f))) -
      ifelse(Bool((mask_sv >> Int32(3)) & Int32(1)), hs1_scale[svyt] * f[svyt], zero(eltype(f))))

    mask_tv = dd1_vmask[tv]
    tvxs = dd1_vxs[tv]; tvys = dd1_vys[tv]; tvxt = dd1_vxt[tv]; tvyt = dd1_vyt[tv]
    codiff1_tv = ihs0_scale[tv] * (
      ifelse(Bool(mask_tv & Int32(1)),             hs1_scale[tvxs] * f[tvxs], zero(eltype(f))) +
      ifelse(Bool((mask_tv >> Int32(1)) & Int32(1)), hs1_scale[tvys] * f[tvys], zero(eltype(f))) -
      ifelse(Bool((mask_tv >> Int32(2)) & Int32(1)), hs1_scale[tvxt] * f[tvxt], zero(eltype(f))) -
      ifelse(Bool((mask_tv >> Int32(3)) & Int32(1)), hs1_scale[tvyt] * f[tvyt], zero(eltype(f))))

    term1 = codiff1_tv - codiff1_sv

    # ── Term 2 ─────────────────────────────────────────────────────────────
    mask_e = dd0_emask[e]
    qp = dd0_qp[e]; qn = dd0_qn[e]
    d1_qp = ifelse(Bool(mask_e & Int32(1)),
                   f[q_e1[qp]] + f[q_e2[qp]] - f[q_e3[qp]] - f[q_e4[qp]],
                   zero(eltype(f)))
    d1_qn = ifelse(Bool((mask_e >> Int32(1)) & Int32(1)),
                   f[q_e1[qn]] + f[q_e2[qn]] - f[q_e3[qn]] - f[q_e4[qn]],
                   zero(eltype(f)))
    term2 = ihs1_hs2_scale[e] * (d1_qp - d1_qn)

    res[e] = term1 + term2
  end
end

# ── Primal Laplacian interface functions ──────────────────────────────────
function laplacian!(res, ::Val{0}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_laplacian0_cached!(backend)(res,
    cache.dd1_vxs, cache.dd1_vys, cache.dd1_vxt, cache.dd1_vyt, cache.dd1_vmask,
    cache.ihs0_scale, cache.hs1_scale, cache.src_v, cache.tgt_v, f;
    ndrange = cache.nv_)
  return res
end

function laplacian!(res, ::Val{1}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_laplacian1_cached!(backend)(res,
    cache.src_v, cache.tgt_v,
    cache.dd1_vxs, cache.dd1_vys, cache.dd1_vxt, cache.dd1_vyt, cache.dd1_vmask,
    cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
    cache.dd0_qp, cache.dd0_qn, cache.dd0_emask,
    cache.ihs0_scale, cache.hs1_scale, cache.ihs1_hs2_scale, f;
    ndrange = cache.ne_)
  return res
end

function laplacian!(res, ::Val{2}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_laplacian2_cached!(backend)(res,
    cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
    cache.dd0_qp, cache.dd0_qn, cache.dd0_emask,
    cache.ihs1_hs2_scale, f;
    ndrange = cache.nquads_)
  return res
end

function laplacian(::Val{k}, cache::UniformDECCache, f::AbstractVector{FT}) where {k, FT}
  backend = get_backend(f)
  n = k == 0 ? cache.nv_ : k == 1 ? cache.ne_ : cache.nquads_
  res = KernelAbstractions.zeros(backend, FT, n)
  return laplacian!(res, Val(k), cache, f)
end

# ═══════════════════════════════════════════════════════════════════════════
#  Fused dual Laplacian kernels
#
#    dual_laplacian(0) = dcd1 ∘ dd0   = (hs2 ∘ d1 ∘ ihs1) ∘ dd0   (nq → nq)
#    dual_laplacian(1) = dcd2 ∘ dd1 + dd0 ∘ dcd1                   (ne → ne)
#    dual_laplacian(2) = dd1 ∘ dcd2   = (-d0ᵀ) ∘ (hs1 ∘ d0 ∘ ihs0) (nv → nv)
#
#  Note: dual_laplacian(0) expands to hs2·d1·ihs1·dd0·f which is identical to
#  laplacian(2) = d1·ihs1·dd0·hs2·f since hs2 is a uniform scalar.
#  The same kernel is therefore reused.
# ═══════════════════════════════════════════════════════════════════════════

# ── dual_laplacian(2): dd1 ∘ dcd2, thread per vertex v ───────────────────
# dcd2 maps 0-forms to 1-forms: (dcd2*f)[e] = hs1[e]*(ihs0[tgt(e)]*f[tgt(e)] - ihs0[src(e)]*f[src(e)])
# dd1 then gathers with signs (+,+,-,-) for the 4 edges incident to v.
# For slot vxs (v=src): tgt neighbor = tgt_v[vxs]; for vxt (v=tgt): src neighbor = src_v[vxt].
@kernel function kernel_dual_laplacian2_cached!(res, @Const(dd1_vxs), @Const(dd1_vys), @Const(dd1_vxt), @Const(dd1_vyt),
                                                @Const(dd1_vmask), @Const(hs1_scale), @Const(ihs0_scale), @Const(src_v), @Const(tgt_v), @Const(f))
  v = @index(Global)
  @inbounds begin
    mask = dd1_vmask[v]
    vxs = dd1_vxs[v]; vys = dd1_vys[v]; vxt = dd1_vxt[v]; vyt = dd1_vyt[v]
    # For src-slots: (dcd2*f)[e] = hs1[e]*(ihs0[tgt(e)]*f[tgt(e)] - ihs0[v]*f[v])
    xs = ifelse(Bool(mask & Int32(1)),
                hs1_scale[vxs] * (ihs0_scale[tgt_v[vxs]] * f[tgt_v[vxs]] - ihs0_scale[v] * f[v]),
                zero(eltype(f)))
    ys = ifelse(Bool((mask >> Int32(1)) & Int32(1)),
                hs1_scale[vys] * (ihs0_scale[tgt_v[vys]] * f[tgt_v[vys]] - ihs0_scale[v] * f[v]),
                zero(eltype(f)))
    # For tgt-slots: (dcd2*f)[e] = hs1[e]*(ihs0[v]*f[v] - ihs0[src(e)]*f[src(e)])
    # dd1 sign is -1 for tgt-slots → net: -(hs1*(ihs0_v*f_v - ihs0_src*f_src))
    xt = ifelse(Bool((mask >> Int32(2)) & Int32(1)),
                hs1_scale[vxt] * (ihs0_scale[v] * f[v] - ihs0_scale[src_v[vxt]] * f[src_v[vxt]]),
                zero(eltype(f)))
    yt = ifelse(Bool((mask >> Int32(3)) & Int32(1)),
                hs1_scale[vyt] * (ihs0_scale[v] * f[v] - ihs0_scale[src_v[vyt]] * f[src_v[vyt]]),
                zero(eltype(f)))
    res[v] = xs + ys - xt - yt
  end
end

# ── dual_laplacian(1): dcd2*dd1 + dd0*dcd1, thread per edge e ────────────
# Term 1: dcd2(dd1*a)[e] = hs1[e]*(ihs0[tv]*(dd1*a)[tv] - ihs0[sv]*(dd1*a)[sv])
#   (dd1*a)[v] is the bitmask-guarded sum of a at the 4 incident edges of v.
# Term 2: dd0(dcd1*a)[e] = (dcd1*a)[qp] - (dcd1*a)[qn]
#   (dcd1*a)[q] = hs2 * (ihs1[e1]*a[e1]+ihs1[e2]*a[e2]-ihs1[e3]*a[e3]-ihs1[e4]*a[e4])
@kernel function kernel_dual_laplacian1_cached!(res, @Const(src_v), @Const(tgt_v),
                                                @Const(dd1_vxs), @Const(dd1_vys), @Const(dd1_vxt), @Const(dd1_vyt), @Const(dd1_vmask),
                                                @Const(q_e1), @Const(q_e2), @Const(q_e3), @Const(q_e4),
                                                @Const(dd0_qp), @Const(dd0_qn), @Const(dd0_emask),
                                                @Const(hs1_scale), @Const(ihs0_scale), @Const(ihs1_hs2_scale), @Const(a))
  e = @index(Global)
  @inbounds begin
    sv = src_v[e]; tv = tgt_v[e]

    # ── Term 1: dcd2 * dd1 ─────────────────────────────────────────────────
    mask_sv = dd1_vmask[sv]
    svxs = dd1_vxs[sv]; svys = dd1_vys[sv]; svxt = dd1_vxt[sv]; svyt = dd1_vyt[sv]
    dd1_sv = (ifelse(Bool(mask_sv & Int32(1)),             a[svxs], zero(eltype(a))) +
              ifelse(Bool((mask_sv >> Int32(1)) & Int32(1)), a[svys], zero(eltype(a))) -
              ifelse(Bool((mask_sv >> Int32(2)) & Int32(1)), a[svxt], zero(eltype(a))) -
              ifelse(Bool((mask_sv >> Int32(3)) & Int32(1)), a[svyt], zero(eltype(a))))

    mask_tv = dd1_vmask[tv]
    tvxs = dd1_vxs[tv]; tvys = dd1_vys[tv]; tvxt = dd1_vxt[tv]; tvyt = dd1_vyt[tv]
    dd1_tv = (ifelse(Bool(mask_tv & Int32(1)),             a[tvxs], zero(eltype(a))) +
              ifelse(Bool((mask_tv >> Int32(1)) & Int32(1)), a[tvys], zero(eltype(a))) -
              ifelse(Bool((mask_tv >> Int32(2)) & Int32(1)), a[tvxt], zero(eltype(a))) -
              ifelse(Bool((mask_tv >> Int32(3)) & Int32(1)), a[tvyt], zero(eltype(a))))

    term1 = hs1_scale[e] * (ihs0_scale[tv] * dd1_tv - ihs0_scale[sv] * dd1_sv)

    # ── Term 2: dd0 * dcd1 ─────────────────────────────────────────────────
    mask_e = dd0_emask[e]
    qp = dd0_qp[e]; qn = dd0_qn[e]
    dcd1_qp = ifelse(Bool(mask_e & Int32(1)),
                     ihs1_hs2_scale[q_e1[qp]] * a[q_e1[qp]] +
                     ihs1_hs2_scale[q_e2[qp]] * a[q_e2[qp]] -
                     ihs1_hs2_scale[q_e3[qp]] * a[q_e3[qp]] -
                     ihs1_hs2_scale[q_e4[qp]] * a[q_e4[qp]],
                     zero(eltype(a)))
    dcd1_qn = ifelse(Bool((mask_e >> Int32(1)) & Int32(1)),
                     ihs1_hs2_scale[q_e1[qn]] * a[q_e1[qn]] +
                     ihs1_hs2_scale[q_e2[qn]] * a[q_e2[qn]] -
                     ihs1_hs2_scale[q_e3[qn]] * a[q_e3[qn]] -
                     ihs1_hs2_scale[q_e4[qn]] * a[q_e4[qn]],
                     zero(eltype(a)))
    term2 = dcd1_qp - dcd1_qn

    res[e] = term1 + term2
  end
end

# ── Dual Laplacian interface functions ────────────────────────────────────
# dual_laplacian(0) = dcd1 ∘ dd0 = hs2·d1·ihs1·dd0·f
# Since hs2 is a uniform scalar this is algebraically identical to laplacian(2).
function dual_laplacian!(res, ::Val{0}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_laplacian2_cached!(backend)(res,
    cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
    cache.dd0_qp, cache.dd0_qn, cache.dd0_emask,
    cache.ihs1_hs2_scale, f;
    ndrange = cache.nquads_)
  return res
end

function dual_laplacian!(res, ::Val{1}, cache::UniformDECCache, a)
  backend = get_backend(a)
  kernel_dual_laplacian1_cached!(backend)(res,
    cache.src_v, cache.tgt_v,
    cache.dd1_vxs, cache.dd1_vys, cache.dd1_vxt, cache.dd1_vyt, cache.dd1_vmask,
    cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
    cache.dd0_qp, cache.dd0_qn, cache.dd0_emask,
    cache.hs1_scale, cache.ihs0_scale, cache.ihs1_hs2_scale, a;
    ndrange = cache.ne_)
  return res
end

function dual_laplacian!(res, ::Val{2}, cache::UniformDECCache, f)
  backend = get_backend(f)
  kernel_dual_laplacian2_cached!(backend)(res,
    cache.dd1_vxs, cache.dd1_vys, cache.dd1_vxt, cache.dd1_vyt, cache.dd1_vmask,
    cache.hs1_scale, cache.ihs0_scale, cache.src_v, cache.tgt_v, f;
    ndrange = cache.nv_)
  return res
end

function dual_laplacian(::Val{k}, cache::UniformDECCache, f::AbstractVector{FT}) where {k, FT}
  backend = get_backend(f)
  n = k == 0 ? cache.nquads_ : k == 1 ? cache.ne_ : cache.nv_
  res = KernelAbstractions.zeros(backend, FT, n)
  return dual_laplacian!(res, Val(k), cache, f)
end
