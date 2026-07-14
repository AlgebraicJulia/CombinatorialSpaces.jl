using KernelAbstractions
using Adapt

abstract type AdvectionScheme end
struct Upwind <: AdvectionScheme end
struct WENO5   <: AdvectionScheme end

abstract type FluxLimiter end

"""Minmod limiter – most dissipative TVD limiter (first-order near extrema)."""
struct MinmodLimiter    <: FluxLimiter end

"""Van Leer limiter – smooth, second-order TVD limiter."""
struct VanLeerLimiter   <: FluxLimiter end

"""Superbee limiter – most compressive TVD limiter (Roe 1985)."""
struct SuperbeeLimiter  <: FluxLimiter end

"""Monotonized Central (MC) limiter – between minmod and superbee."""
struct MCLimiter        <: FluxLimiter end

"""Van Albada limiter – smooth and differentiable TVD limiter."""
struct VanAlbadaLimiter <: FluxLimiter end

@inline apply_limiter(::MinmodLimiter,    r::FT) where FT <: AbstractFloat =
  max(zero(FT), min(one(FT), r))

@inline apply_limiter(::VanLeerLimiter,   r::FT) where FT <: AbstractFloat =
  (r + abs(r)) / (one(FT) + abs(r))

@inline apply_limiter(::SuperbeeLimiter,  r::FT) where FT <: AbstractFloat =
  max(zero(FT), min(2 * r, one(FT)), min(r, 2 * one(FT)))

@inline apply_limiter(::MCLimiter,        r::FT) where FT <: AbstractFloat =
  max(zero(FT), min(min(2 * r, (one(FT) + r) / 2), 2 * one(FT)))

@inline apply_limiter(::VanAlbadaLimiter, r::FT) where FT <: AbstractFloat =
  (r * r + r) / (r * r + one(FT))

@kernel function wedge_product_01_upwind!(res, s, f0, f1)
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  f0_src = f0[src(s, x, y, align)]
  f0_tgt = f0[tgt(s, x, y, align)]
  f1_val = f1[idx]

  @inbounds res[idx] = f0_src * max(f1_val, 0) + f0_tgt * min(f1_val, 0)
end

function wedge_product_01(sch::AdvectionScheme, s::UniformCubicalComplex2D, f0, f1)
  res = KernelAbstractions.zeros(get_backend(f0), eltype(f0), ne(s))
  return wedge_product_01!(res, sch, s, f0, f1)
end

function wedge_product_01!(res, sch::Upwind, s::UniformCubicalComplex2D, f0, f1)
  backend = get_backend(f0)
  kernel =  wedge_product_01_upwind!(backend)
  kernel(res, s, f0, f1; ndrange = ne(s))
  return res
end

@kernel function wedge_product_11_upwind!(res, s, f1a, f1b)
  idx = @index(Global)
  x, y = quad_to_coord(s, idx)

  # Edges of the quad, ordered as: bottom, right, top, left
  e1, e2, e3, e4 = quad_edges(s, x, y)

  f1a_e1 = f1a[e1]; f1a_e2 = f1a[e2]; f1a_e3 = f1a[e3]; f1a_e4 = f1a[e4]
  f1b_e1 = f1b[e1]; f1b_e2 = f1b[e2]; f1b_e3 = f1b[e3]; f1b_e4 = f1b[e4]

  avg_x_flow = (f1a_e2 + f1a_e4) / 2
  avg_y_flow = (f1a_e1 + f1a_e3) / 2

  # Upwind selection for each f1b edge value
  x_upwind = avg_x_flow >= 0 ? f1b_e1 : f1b_e3
  y_upwind = avg_y_flow >= 0 ? f1b_e4 : f1b_e2

  @inbounds res[idx] = y_upwind * avg_y_flow - x_upwind * avg_x_flow
end

@kernel function wedge_product_11_WENO5!(res, s, f1a, f1b, eps)
  idx = @index(Global)
  x, y = quad_to_coord(s, idx)

  # Edges of the quad, ordered as: bottom, right, top, left
  e1, e2, e3, e4 = quad_edges(s, x, y)

  f1a_e1 = f1a[e1]; f1a_e2 = f1a[e2]; f1a_e3 = f1a[e3]; f1a_e4 = f1a[e4]

  f1b_x = f1b[e1]; f1b_yp1 = f1b[e2]; f1b_xp1 = f1b[e3]; f1b_y = f1b[e4]

  avg_x_flow = (f1a_e2 + f1a_e4) / 2
  avg_y_flow = (f1a_e1 + f1a_e3) / 2

  # WENO5 reconstruction for each direction
  # If too close to the boundary, fall back to upwind selection

  # Need room for fm2 and fp3 stencils in both directions.
  if x <= 2 || x >= nx(s) - 2 || y <= 2 || y >= ny(s) - 2
    # Near boundaries, use upwind selection
     x_upwind = avg_x_flow >= 0 ? f1b_x : f1b_xp1
     y_upwind = avg_y_flow >= 0 ? f1b_y : f1b_yp1
  else
    f1b_xm1 = f1b[quad_edge_offset(s, x, y, X_ALIGN, -1)]; f1b_xp2 = f1b[quad_edge_offset(s, x, y, X_ALIGN, 2)]
    f1b_xm2 = f1b[quad_edge_offset(s, x, y, X_ALIGN, -2)]; f1b_xp3 = f1b[quad_edge_offset(s, x, y, X_ALIGN, 3)]

    f1b_ym1 = f1b[quad_edge_offset(s, x, y, Y_ALIGN, -1)]; f1b_yp2 = f1b[quad_edge_offset(s, x, y, Y_ALIGN, 2)]
    f1b_ym2 = f1b[quad_edge_offset(s, x, y, Y_ALIGN, -2)]; f1b_yp3 = f1b[quad_edge_offset(s, x, y, Y_ALIGN, 3)]

    # WENO5 reconstruction for x-direction
    x_upwind = if avg_x_flow >= 0
      weno5_point(f1b_xm2, f1b_xm1, f1b_x, f1b_xp1, f1b_xp2, eps)
    else
      weno5_point(f1b_xp3, f1b_xp2, f1b_xp1, f1b_x, f1b_xm1, eps)
    end

    y_upwind = if avg_y_flow >= 0
      weno5_point(f1b_ym2, f1b_ym1, f1b_y, f1b_yp1, f1b_yp2, eps)
    else
      weno5_point(f1b_yp3, f1b_yp2, f1b_yp1, f1b_y, f1b_ym1, eps)
    end
  end

  @inbounds res[idx] = y_upwind * avg_y_flow - x_upwind * avg_x_flow
end

function wedge_product_11(sch::AdvectionScheme, s::UniformCubicalComplex2D, f1a, f1b)
  res = KernelAbstractions.zeros(get_backend(f1a), eltype(f1a), nquads(s))
  return wedge_product_11!(res, sch, s, f1a, f1b)
end

function wedge_product_11!(res, sch::Upwind, s::UniformCubicalComplex2D, a, b)
  backend = get_backend(a)
  kernel = wedge_product_11_upwind!(backend)
  kernel(res, s, a, b; ndrange = nquads(s))
  return res
end

function wedge_product_11!(res, sch::WENO5, s::UniformCubicalComplex2D, a, b, eps = nothing)
  backend = get_backend(a)
  FT = eltype(a)
  eps_T = eps === nothing ? FT(1e-6) : FT(eps)
  kernel = wedge_product_11_WENO5!(backend)
  kernel(res, s, a, b, eps_T; ndrange = nquads(s))
  return res
end

function wedge_product(::Val{i}, ::Val{j}, sch::AdvectionScheme, s::UniformCubicalComplex2D, a, b) where {i, j}
  if i == 0 && j == 1
    return wedge_product_01(sch, s, a, b)
  elseif i == 1 && j == 0
    return wedge_product_01(sch, s, b, a)
  elseif i == 1 && j == 1
    return wedge_product_11(sch, s, a, b)
  else
    error("Wedge product not implemented for forms of degree ($i, $j) with scheme $(typeof(sch)).")
  end
end

# ═══════════════════════════════════════════════════════════════════════════
#  Advection-scheme caches
#
#  Each cache stores exactly the precomputed edge indices needed for its
#  scheme, eliminating per-thread coordinate arithmetic and integer division.
#  Scheme dispatch at construction time controls the memory footprint:
#    AdvectionCache(Upwind(), s)  → UpwindCache  (4 × nquads Int32 arrays)
#    AdvectionCache(WENO5(),  s)  → WENO5Cache   (13 × nquads Int32 arrays)
#
#  GPU usage: Adapt.adapt(backend, cache) after construction.
# ═══════════════════════════════════════════════════════════════════════════

abstract type AbstractAdvectionCache end

# ── UpwindCache ───────────────────────────────────────────────────────────────────
struct UpwindCache{IT <: AbstractVector{Int32}} <: AbstractAdvectionCache
  nquads_ :: Int
  q_e1 :: IT;  q_e2 :: IT;  q_e3 :: IT;  q_e4 :: IT
end
Adapt.@adapt_structure UpwindCache

function UpwindCache(s::UniformCubicalComplex2D)
  nq_ = nquads(s)
  q_e1 = Vector{Int32}(undef, nq_);  q_e2 = Vector{Int32}(undef, nq_)
  q_e3 = Vector{Int32}(undef, nq_);  q_e4 = Vector{Int32}(undef, nq_)
  for q in 1:nq_
    x, y           = quad_to_coord(s, q)
    e1, e2, e3, e4 = quad_edges(s, x, y)
    q_e1[q] = e1;  q_e2[q] = e2;  q_e3[q] = e3;  q_e4[q] = e4
  end
  return UpwindCache(nq_, q_e1, q_e2, q_e3, q_e4)
end

  # ── WENO5Cache (extended) ─────────────────────────────────────────────────────
struct WENO5Cache{IT <: AbstractVector{Int32}, MT <: AbstractVector{Int8}} <: AbstractAdvectionCache
  # ── wedge 1∧1 fields (quad-indexed) ──────────────────────────────────
  nquads_ :: Int
  q_e1 :: IT;  q_e2 :: IT;  q_e3 :: IT;  q_e4 :: IT

  # Extended stencil — x-aligned edges at y-offsets {-2,-1,+2,+3} from each quad
  q_wxm2 :: IT;  q_wxm1 :: IT;  q_wxp2 :: IT;  q_wxp3 :: IT

  # Extended stencil — y-aligned edges at x-offsets {-2,-1,+2,+3} from each quad
  q_wym2 :: IT;  q_wym1 :: IT;  q_wyp2 :: IT;  q_wyp3 :: IT

  # 1 = full WENO5 stencil fits, 0 = boundary quad (falls back to upwinding)
  q_weno_interior :: MT

  # ── wedge 0∧1 fields (edge-indexed) ──────────────────────────────────
  nedges_ :: Int
  e_src :: IT;  e_tgt :: IT   # src/tgt vertex indices per edge

  # Extended stencil vertices along the edge's own axis at offsets
  # {-2,-1,+2,+3} relative to src.  Dummy = 1 at boundaries.
  e_wm2 :: IT;  e_wm1 :: IT;  e_wp2 :: IT;  e_wp3 :: IT

  # 1 = full 6-point stencil fits, 0 = boundary edge (falls back to average)
  e_weno_interior :: MT
end
Adapt.@adapt_structure WENO5Cache

function WENO5Cache(s::UniformCubicalComplex2D)
  nq_  = nquads(s);  nx_ = nx(s);  ny_ = ny(s)
  ne_  = ne(s)

  # ── quad-indexed arrays (unchanged from before) ───────────────────────
  q_e1 = Vector{Int32}(undef, nq_);  q_e2 = Vector{Int32}(undef, nq_)
  q_e3 = Vector{Int32}(undef, nq_);  q_e4 = Vector{Int32}(undef, nq_)
  q_wxm2 = Vector{Int32}(undef, nq_);  q_wxm1 = Vector{Int32}(undef, nq_)
  q_wxp2 = Vector{Int32}(undef, nq_);  q_wxp3 = Vector{Int32}(undef, nq_)
  q_wym2 = Vector{Int32}(undef, nq_);  q_wym1 = Vector{Int32}(undef, nq_)
  q_wyp2 = Vector{Int32}(undef, nq_);  q_wyp3 = Vector{Int32}(undef, nq_)
  q_weno_interior = Vector{Int8}(undef, nq_)

  for q in 1:nq_
      x, y           = quad_to_coord(s, q)
      e1, e2, e3, e4 = quad_edges(s, x, y)
      q_e1[q] = e1;  q_e2[q] = e2;  q_e3[q] = e3;  q_e4[q] = e4
      #TODO: Check this interior
      # interior = (x > 2) & (x < nx_ - 2) & (y > 2) & (y < ny_ - 2)
      interior = (x > 3) & (x < nx_ - 3) & (y > 3) & (y < ny_ - 3)
      q_weno_interior[q] = Int8(interior)
      if interior
          q_wxm2[q] = Int32(quad_edge_offset(s, x, y, X_ALIGN, -2))
          q_wxm1[q] = Int32(quad_edge_offset(s, x, y, X_ALIGN, -1))
          q_wxp2[q] = Int32(quad_edge_offset(s, x, y, X_ALIGN,  2))
          q_wxp3[q] = Int32(quad_edge_offset(s, x, y, X_ALIGN,  3))
          q_wym2[q] = Int32(quad_edge_offset(s, x, y, Y_ALIGN, -2))
          q_wym1[q] = Int32(quad_edge_offset(s, x, y, Y_ALIGN, -1))
          q_wyp2[q] = Int32(quad_edge_offset(s, x, y, Y_ALIGN,  2))
          q_wyp3[q] = Int32(quad_edge_offset(s, x, y, Y_ALIGN,  3))
      else
          q_wxm2[q] = q_wxm1[q] = q_wxp2[q] = q_wxp3[q] = Int32(1)
          q_wym2[q] = q_wym1[q] = q_wyp2[q] = q_wyp3[q] = Int32(1)
      end
  end

  # ── edge-indexed arrays (new) ─────────────────────────────────────────
  e_src = Vector{Int32}(undef, ne_);  e_tgt = Vector{Int32}(undef, ne_)
  e_wm2 = Vector{Int32}(undef, ne_);  e_wm1 = Vector{Int32}(undef, ne_)
  e_wp2 = Vector{Int32}(undef, ne_);  e_wp3 = Vector{Int32}(undef, ne_)
  e_weno_interior = Vector{Int8}(undef, ne_)

  for e in 1:ne_
      x, y, align = edge_to_coord(s, e)
      e_src[e] = Int32(src(s, x, y, align))
      e_tgt[e] = Int32(tgt(s, x, y, align))

      # Stencil fits when there is room for offsets -2 and +3 along the
      # edge's own axis (identical radius logic to the quad WENO5 check).
      #TODO: Check this interior
      # interior = if align == X_ALIGN
      #     (x > 2) & (x <= nx_ - 2)
      # else  # Y_ALIGN
      #     (y > 2) & (y <= ny_ - 2)
      # end
      interior = if align == X_ALIGN
          (x > 3) & (x <= nx_ - 3)
      else  # Y_ALIGN
          (y > 3) & (y <= ny_ - 3)
      end

      e_weno_interior[e] = Int8(interior)

      if interior
          # Vertex stencil along the edge axis:
          #   src is offset 0, tgt is offset +1.
          #   We need offsets -2, -1 (behind src) and +2, +3 (beyond tgt).
          e_wm2[e] = Int32(edge_vertex_offset(s, x, y, align, -2))
          e_wm1[e] = Int32(edge_vertex_offset(s, x, y, align, -1))
          e_wp2[e] = Int32(edge_vertex_offset(s, x, y, align,  2))
          e_wp3[e] = Int32(edge_vertex_offset(s, x, y, align,  3))
      else
          e_wm2[e] = e_wm1[e] = e_wp2[e] = e_wp3[e] = Int32(1)
      end
  end

  return WENO5Cache(nq_, q_e1, q_e2, q_e3, q_e4,
                    q_wxm2, q_wxm1, q_wxp2, q_wxp3,
                    q_wym2, q_wym1, q_wyp2, q_wyp3,
                    q_weno_interior,
                    ne_, e_src, e_tgt,
                    e_wm2, e_wm1, e_wp2, e_wp3,
                    e_weno_interior)
end

# ── Factory: construct the appropriate cache for the given scheme ─────────
AdvectionCache(::Upwind, s) = UpwindCache(s)
AdvectionCache(::WENO5,  s) = WENO5Cache(s)

# ── Cached upwinding kernel (branchless) ──────────────────────────────────────
# Flow sign selection uses ifelse (GPU predication / CMOV).
@kernel function kernel_wedge_11_upwind_cached!(res,
                                                @Const(q_e1), @Const(q_e2),
                                                @Const(q_e3), @Const(q_e4),
                                                @Const(f1a), @Const(f1b))
  q = @index(Global)
  @inbounds begin
    avg_x_flow = (f1a[q_e2[q]] + f1a[q_e4[q]]) * 0.5
    avg_y_flow = (f1a[q_e1[q]] + f1a[q_e3[q]]) * 0.5
    x_upwind   = ifelse(avg_x_flow >= 0, f1b[q_e1[q]], f1b[q_e3[q]])
    y_upwind   = ifelse(avg_y_flow >= 0, f1b[q_e4[q]], f1b[q_e2[q]])
    res[q]     = y_upwind * avg_y_flow - x_upwind * avg_x_flow
  end
end

# ── direction-split cached WENO5 wedge 1∧1 ───────────────────────────────
#
# The original single-kernel evaluated both x and y WENO5 reconstructions
# per thread, driving register count to ~70-80 and severely limiting GPU
# occupancy (~12-16%).  Splitting into three passes halves register pressure
# per pass and roughly doubles occupancy:
#
#   Pass 1 (nquads threads): x-direction reconstruction → tmp_x[q]
#   Pass 2 (nquads threads): y-direction reconstruction → tmp_y[q]
#   Pass 3 (nquads threads): combine → res[q] = tmp_y*avg_y - tmp_x*avg_x
#
# Interior quads use full WENO5; boundary quads fall back to first-order
# upwinding.  Stencil arrays always hold valid indices (dummy=1 at boundaries)
# so there are no out-of-bounds accesses on either path.
#
# ── FURTHER PERFORMANCE IDEAS (not yet implemented) ──────────────────────
#
# 1. FUSED TWO-PASS VARIANT (eliminate tmp_x, tmp_y intermediates entirely)
#    The current design requires two nquads-sized temporaries and an extra
#    round-trip through global memory.  On large grids (500×500 = 250k quads)
#    that is ~2 MB of intermediate traffic at Float32.  An alternative is to
#    merge passes 1+2 back into a single kernel but store only 6 registers per
#    direction instead of 12 (use inlined scalars, not arrays).  This requires
#    careful register-pressure accounting with `@ptx_occupancy` or Nsight;
#    the break-even depends on whether the memory-bandwidth savings outweigh
#    any additional register spilling.  Worth profiling specifically on the
#    target arch (sm_86 / sm_89) since register file size differs.
#
# 2. SHARED MEMORY TILE FOR STENCIL READS
#    Each quad reads 6 f1b edge values from potentially scattered addresses.
#    On a regular nx×ny grid the x-stencil is stride-1 in memory (all x-edges
#    are contiguous in a row), but the y-stencil has stride ~nx.  Tiling a
#    (BX+4)×(BY+4) block of f1b into shared memory and computing from there
#    could cut global memory transactions by ~4x for the stencil reads.
#    KernelAbstractions supports @localmem and @synchronize; this would require
#    restructuring the kernel launch to use 2D ndrange with a tile workgroup
#    and replacing the q_w* index arrays with tile-relative offsets.
#    Expected benefit: largest for memory-bandwidth-bound regimes (small-to-mid
#    grids where L2 hit rate is low), minimal for very large grids that already
#    stream efficiently.
#
# 3. WARP-LEVEL PREDICATION TO REMOVE BRANCH DIVERGENCE
#    The `if Bool(q_weno_interior[q])` branch causes warp divergence on the
#    ~4*(nx+ny) boundary quads per interior ring.  For typical 500×500 grids
#    only ~0.3% of quads are boundary, so the divergent warp cost is low.
#    However, at small grid sizes (e.g. 64×64) the fraction is ~6%.  An
#    alternative is to sort/partition quads into two contiguous index ranges
#    (interior and boundary) at cache-build time and launch two separate
#    ndranges: one pure WENO5 kernel (no branch) and one pure upwind kernel
#    (also no branch).  This eliminates divergence entirely and makes each
#    kernel simpler.  Cache construction cost is one extra sort; no runtime
#    overhead.
#
# 4. INT8 FOR q_weno_interior — DONE
#    q_weno_interior was changed from Int32 (4 bytes/quad) to Int8
#    (1 byte/quad).  On 500×500 this saves ~1 MB of cache memory and
#    improves L1/L2 locality for this flag load.  The kernel usage
#    (Bool(q_weno_interior[q])) requires no change since Bool(Int8) works
#    identically.  A BitArray-style bitmask (1 bit/quad) was not pursued
#    since it would require bit-extraction logic in the kernel.
#
# 5. WENO5 WEIGHTS AS COMPILE-TIME CONSTANTS (avoid eps broadcast)
#    Currently `eps` is a scalar kernel argument that is broadcast to every
#    thread.  The Julia/LLVM pipeline typically hoists it to a register, but
#    marking it as a `@Const` literal (or passing it via a Val{eps} type
#    parameter) removes any residual broadcast overhead and lets the compiler
#    fold it into the constant pool.  Low-risk, minimal gain but free.
#
# 6. FP16 / BF16 STENCIL ACCUMULATION (experimental)
#    If the simulation can tolerate reduced stencil precision, computing
#    beta3 in Float16 and accumulating in Float32 could nearly halve register
#    pressure for the smoothness indicator computation.  Requires Julia CUDA.jl
#    Float16 arithmetic support and careful verification that the WENO5 weight
#    normalization remains numerically stable.  Not recommended without a
#    reference-accuracy suite.
#
# 7. REPLACE STENCIL INDEX ARRAYS WITH STRIDE ARITHMETIC
#    The q_w* arrays (8 × nquads Int32) exist to avoid integer division per
#    thread.  An alternative is to pack the (x, y) quad coordinate into a
#    single Int32 stored in q_xy and compute stencil offsets as simple
#    additions (e.g. x-stencil offset = ±1 edge-index stride).  If the stride
#    between consecutive x-edges is constant (it is: stride = 1 for x-aligned
#    edges, nx+1 for y-aligned edges), each stencil lookup becomes a base
#    pointer + small integer offset — no extra cache array needed.  This would
#    cut the WENO5Cache footprint from 13 arrays to 5 and improve cache
#    prefetch coherence.  The trade-off is one integer multiply per stencil
#    read vs. one indexed load.  On modern CUDA cores integer MADs are free
#    in the shadow of memory latency, so this is likely a net win.
#
# 8. ASYNC MEMORY COPY / PREFETCH WITH cp.async (sm_80+)
#    On Ampere and later, CUDA's cp.async instruction can overlap stencil data
#    movement with computation.  KernelAbstractions does not currently expose
#    this directly, but a hand-written CUDA.jl kernel using
#    CUDA.@async_copy could be used as a specialised backend override.
#    Only relevant for the memory-bandwidth-bound tile variant (idea 2).
#
# ─────────────────────────────────────────────────────────────────────────

# Pass 1 — x-direction upwind reconstruction
@kernel function kernel_weno5_upwind_x_cached!(x_upwind,
                                               @Const(q_e1), @Const(q_e2),
                                               @Const(q_e3), @Const(q_e4),
                                               @Const(q_wxm2), @Const(q_wxm1),
                                               @Const(q_wxp2), @Const(q_wxp3),
                                               @Const(q_weno_interior),
                                               eps, @Const(f1a), @Const(f1b))
  q = @index(Global)
  @inbounds begin
    avg_x_flow = (f1a[q_e2[q]] + f1a[q_e4[q]]) * 0.5
    if Bool(q_weno_interior[q])
      f1b_xm2 = f1b[q_wxm2[q]];  f1b_xm1 = f1b[q_wxm1[q]]
      f1b_x   = f1b[q_e1[q]];    f1b_xp1 = f1b[q_e3[q]]
      f1b_xp2 = f1b[q_wxp2[q]];  f1b_xp3 = f1b[q_wxp3[q]]
      x_upwind[q] = if avg_x_flow >= 0
        weno5_point(f1b_xm2, f1b_xm1, f1b_x,   f1b_xp1, f1b_xp2, eps)
      else
        weno5_point(f1b_xp3, f1b_xp2, f1b_xp1, f1b_x,   f1b_xm1, eps)
      end
    else
      x_upwind[q] = ifelse(avg_x_flow >= 0, f1b[q_e1[q]], f1b[q_e3[q]])
    end
  end
end

# Pass 2 — y-direction upwind reconstruction
@kernel function kernel_weno5_upwind_y_cached!(y_upwind,
                                               @Const(q_e1), @Const(q_e2),
                                               @Const(q_e3), @Const(q_e4),
                                               @Const(q_wym2), @Const(q_wym1),
                                               @Const(q_wyp2), @Const(q_wyp3),
                                               @Const(q_weno_interior),
                                               eps, @Const(f1a), @Const(f1b))
  q = @index(Global)
  @inbounds begin
    avg_y_flow = (f1a[q_e1[q]] + f1a[q_e3[q]]) * 0.5
    if Bool(q_weno_interior[q])
      f1b_ym2 = f1b[q_wym2[q]];  f1b_ym1 = f1b[q_wym1[q]]
      f1b_y   = f1b[q_e4[q]];    f1b_yp1 = f1b[q_e2[q]]
      f1b_yp2 = f1b[q_wyp2[q]];  f1b_yp3 = f1b[q_wyp3[q]]
      y_upwind[q] = if avg_y_flow >= 0
        weno5_point(f1b_ym2, f1b_ym1, f1b_y,   f1b_yp1, f1b_yp2, eps)
      else
        weno5_point(f1b_yp3, f1b_yp2, f1b_yp1, f1b_y,   f1b_ym1, eps)
      end
    else
      y_upwind[q] = ifelse(avg_y_flow >= 0, f1b[q_e4[q]], f1b[q_e2[q]])
    end
  end
end

# Pass 3 — combine: re-reads avg flows from f1a (4 loads) + 2 scalar reads
@kernel function kernel_weno5_combine!(res,
                                       @Const(q_e1), @Const(q_e2),
                                       @Const(q_e3), @Const(q_e4),
                                       @Const(x_upwind), @Const(y_upwind),
                                       @Const(f1a))
  q = @index(Global)
  @inbounds begin
    avg_x_flow = (f1a[q_e2[q]] + f1a[q_e4[q]]) * 0.5
    avg_y_flow = (f1a[q_e1[q]] + f1a[q_e3[q]]) * 0.5
    res[q] = y_upwind[q] * avg_y_flow - x_upwind[q] * avg_x_flow
  end
end

# ── Cached interface functions ─────────────────────────────────────────────

# Upwinding works with either UpwindCache or UniformDECCache (which also
# stores q_e1..q_e4 and nquads_).
function wedge_product_11!(res, ::Upwind, cache::Union{UpwindCache, UniformDECCache}, f1a, f1b)
  backend = get_backend(f1a)
  kernel_wedge_11_upwind_cached!(backend)(res,
    cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
    f1a, f1b; ndrange = cache.nquads_)
  return res
end

function wedge_product_11!(res, tmp_x, tmp_y, ::WENO5, cache::WENO5Cache, f1a, f1b; eps = nothing)
  backend = get_backend(f1a)
  FT      = eltype(f1a)
  eps_T   = eps === nothing ? FT(1e-6) : FT(eps)
  kernel_weno5_upwind_x_cached!(backend)(tmp_x,
    cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
    cache.q_wxm2, cache.q_wxm1, cache.q_wxp2, cache.q_wxp3,
    cache.q_weno_interior, eps_T, f1a, f1b; ndrange = cache.nquads_)
  kernel_weno5_upwind_y_cached!(backend)(tmp_y,
    cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
    cache.q_wym2, cache.q_wym1, cache.q_wyp2, cache.q_wyp3,
    cache.q_weno_interior, eps_T, f1a, f1b; ndrange = cache.nquads_)
  kernel_weno5_combine!(backend)(res,
    cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
    tmp_x, tmp_y, f1a; ndrange = cache.nquads_)
  return res
end

function wedge_product_11!(res, ::WENO5, cache::WENO5Cache, f1a, f1b; eps = nothing)
  backend = get_backend(f1a)
  FT      = eltype(f1a)
  tmp_x   = KernelAbstractions.zeros(backend, FT, cache.nquads_)
  tmp_y   = KernelAbstractions.zeros(backend, FT, cache.nquads_)
  return wedge_product_11!(res, tmp_x, tmp_y, WENO5(), cache, f1a, f1b; eps)
end

# Allocating wrappers
function wedge_product_11(sch::AdvectionScheme, cache::AbstractAdvectionCache,
                          f1a::AbstractVector{FT}, f1b::AbstractVector{FT}) where FT
  backend = get_backend(f1a)
  res = KernelAbstractions.zeros(backend, FT, cache.nquads_)
  return wedge_product_11!(res, sch, cache, f1a, f1b)
end

# Upwinding via UniformDECCache (backward compatibility)
function wedge_product_11(sch::Upwind, cache::UniformDECCache,
                          f1a::AbstractVector{FT}, f1b::AbstractVector{FT}) where FT
  backend = get_backend(f1a)
  res = KernelAbstractions.zeros(backend, FT, cache.nquads_)
  return wedge_product_11!(res, sch, cache, f1a, f1b)
end

# Val dispatch
wedge_product(::Val{1}, ::Val{1}, sch::AdvectionScheme, cache::AbstractAdvectionCache, f1a, f1b) =
  wedge_product_11(sch, cache, f1a, f1b)

wedge_product(::Val{1}, ::Val{1}, sch::Upwind, cache::UniformDECCache, f1a, f1b) =
  wedge_product_11(sch, cache, f1a, f1b)

# ── kernel ────────────────────────────────────────────────────────────────────
#
# Single pass over ne(s) edges.  For interior edges, WENO5 reconstructs f0
# at the edge midpoint from the 6 vertices along the edge's own axis.
# The stencil is oriented upwind: positive f1 flows from src→tgt (positive
# axis direction), so positive flow uses the stencil rooted behind src;
# negative flow reverses it.
#
# Boundary edges (stencil doesn't fit) fall back to the simple average
# (f0[src] + f0[tgt]) / 2, which is the uncached wedge_product_01 behaviour.
@kernel function kernel_weno5_wedge_01_cached!(res,
                                             @Const(e_src), @Const(e_tgt),
                                             @Const(e_wm2), @Const(e_wm1),
                                             @Const(e_wp2), @Const(e_wp3),
                                             @Const(e_weno_interior),
                                             eps, @Const(f0), @Const(f1))
  e = @index(Global)
  @inbounds begin
      f1_val  = f1[e]
      f0_recon = if Bool(e_weno_interior[e])
          vm2 = f0[e_wm2[e]];  vm1 = f0[e_wm1[e]]
          v0  = f0[e_src[e]];  vp1 = f0[e_tgt[e]]
          vp2 = f0[e_wp2[e]];  vp3 = f0[e_wp3[e]]
          if f1_val >= 0
              weno5_point(vm2, vm1, v0,  vp1, vp2, eps)
          else
              weno5_point(vp3, vp2, vp1, v0,  vm1, eps)
          end
      else
          (f0[e_src[e]] + f0[e_tgt[e]]) * eltype(f1)(0.5)
      end
      res[e] = f0_recon * f1_val
  end
end

# ── in-place interface ────────────────────────────────────────────────────────
function wedge_product_01!(res::AbstractVector{FT}, ::WENO5, cache::WENO5Cache, 
                          f0::AbstractVector{FT}, f1::AbstractVector{FT}; eps::FT = FT(1e-6)) where FT <: AbstractFloat
  backend = get_backend(f1)
  kernel_weno5_wedge_01_cached!(backend)(res,
      cache.e_src, cache.e_tgt,
      cache.e_wm2, cache.e_wm1, cache.e_wp2, cache.e_wp3,
      cache.e_weno_interior,
      eps, f0, f1; ndrange = cache.nedges_)
  return res
end

# ── allocating wrapper ────────────────────────────────────────────────────────
function wedge_product_01(::WENO5, cache::WENO5Cache,
                        f0::AbstractVector{FT},
                        f1::AbstractVector{FT}) where FT <: AbstractFloat
  res = KernelAbstractions.zeros(get_backend(f1), FT, cache.nedges_)
  return wedge_product_01!(res, WENO5(), cache, f0, f1)
end

# ── Val dispatch (matches existing wedge_product_11 pattern) ─────────────────
wedge_product(::Val{0}, ::Val{1}, ::WENO5, cache::WENO5Cache, f0, f1) =
  wedge_product_01(WENO5(), cache, f0, f1)

wedge_product(::Val{1}, ::Val{0}, ::WENO5, cache::WENO5Cache, f1, f0) =
  wedge_product_01(WENO5(), cache, f0, f1)

# ── WENO5Cache3D ──────────────────────────────────────────────────────────────
#
# Three sections, one per wedge product:
#
#   wedge_01 (edge-indexed):
#     e_src, e_tgt         — vertex indices at src/tgt of each edge
#     e_wm2..e_wp3         — vertex stencil along edge's own axis
#     e_weno_interior      — Int8 flag: 1 = full 6-point stencil fits
#
#   wedge_11 (quad-indexed):
#     q_e1..q_e4           — the 4 edge indices of each quad
#     q_wam2..q_wap3       — stencil edges for axis A reconstruction
#     q_wbm2..q_wbp3       — stencil edges for axis B reconstruction
#     q_weno_interior      — Int8 flag
#
#     Axis A / Axis B per quad type:
#       Z_ALIGN: A = X_ALIGN edges (x-offsets), B = Y_ALIGN edges (y-offsets)
#       Y_ALIGN: A = X_ALIGN edges (x-offsets), B = Z_ALIGN edges (z-offsets)
#       X_ALIGN: A = Y_ALIGN edges (y-offsets), B = Z_ALIGN edges (z-offsets)
#
#   wedge_12 (boid-indexed):
#     b_ex1..b_ex4         — 4 x-aligned primal edges (velocity average)
#     b_ey1..b_ey4         — 4 y-aligned primal edges
#     b_ez1..b_ez4         — 4 z-aligned primal edges
#     b_wxm2..b_wxp3       — x-axis YZ-quad stencil (offsets -2,-1,+2,+3)
#     b_wym2..b_wyp3       — y-axis XZ-quad stencil
#     b_wzm2..b_wzp3       — z-axis XY-quad stencil
#     b_weno_interior      — Int8 flag
#
struct WENO5Cache3D{IT <: AbstractVector{Int32},
                    MT <: AbstractVector{Int8}} <: AbstractAdvectionCache

    # ── wedge_01 (edge-indexed) ───────────────────────────────────────────
    nedges_ :: Int
    e_src  :: IT;  e_tgt  :: IT
    e_wm2  :: IT;  e_wm1  :: IT;  e_wp2  :: IT;  e_wp3  :: IT
    e_weno_interior :: MT

    # ── wedge_11 (quad-indexed) ───────────────────────────────────────────
    nquads_ :: Int
    q_e1 :: IT;  q_e2 :: IT;  q_e3 :: IT;  q_e4 :: IT
    # axis-A stencil (X-edges for Z/Y quads; Y-edges for X quads)
    q_wam2 :: IT;  q_wam1 :: IT;  q_wap2 :: IT;  q_wap3 :: IT
    # axis-B stencil (Y-edges for Z quads; Z-edges for Y/X quads)
    q_wbm2 :: IT;  q_wbm1 :: IT;  q_wbp2 :: IT;  q_wbp3 :: IT
    q_weno_interior :: MT

    # ── wedge_12 (boid-indexed) ───────────────────────────────────────────
    nboids_ :: Int
    # velocity edges (primal 1-form averaged over 4 parallel edges per axis)
    b_ex1 :: IT;  b_ex2 :: IT;  b_ex3 :: IT;  b_ex4 :: IT
    b_ey1 :: IT;  b_ey2 :: IT;  b_ey3 :: IT;  b_ey4 :: IT
    b_ez1 :: IT;  b_ez2 :: IT;  b_ez3 :: IT;  b_ez4 :: IT
    # x-axis stencil: X_ALIGN (YZ) quads at x+{-2,-1,+2,+3}
    b_wxm2 :: IT;  b_wxm1 :: IT;  b_wxp2 :: IT;  b_wxp3 :: IT
    # y-axis stencil: Y_ALIGN (XZ) quads at y+{-2,-1,+2,+3}
    b_wym2 :: IT;  b_wym1 :: IT;  b_wyp2 :: IT;  b_wyp3 :: IT
    # z-axis stencil: Z_ALIGN (XY) quads at z+{-2,-1,+2,+3}
    b_wzm2 :: IT;  b_wzm1 :: IT;  b_wzp2 :: IT;  b_wzp3 :: IT

    b_wx0  :: IT;  b_wxp1 :: IT   # X_ALIGN quads at boid-x offset 0 and +1
    b_wy0  :: IT;  b_wyp1 :: IT   # Y_ALIGN quads at boid-y offset 0 and +1
    b_wz0  :: IT;  b_wzp1 :: IT   # Z_ALIGN quads at boid-z offset 0 and +1

    b_weno_interior :: MT
end

Adapt.@adapt_structure WENO5Cache3D

function WENO5Cache3D(s::UniformCubicalComplex3D)
    ne_  = ne(s)
    nq_  = nquads(s)
    nb_  = nboids(s)
    nx_  = nx(s);  ny_ = ny(s);  nz_ = nz(s)
    nxb_ = nxb(s); nyb_ = nyb(s); nzb_ = nzb(s)

    # ── allocate ─────────────────────────────────────────────────────────
    e_src  = Vector{Int32}(undef, ne_);  e_tgt  = Vector{Int32}(undef, ne_)
    e_wm2  = Vector{Int32}(undef, ne_);  e_wm1  = Vector{Int32}(undef, ne_)
    e_wp2  = Vector{Int32}(undef, ne_);  e_wp3  = Vector{Int32}(undef, ne_)
    e_weno_interior = Vector{Int8}(undef, ne_)

    q_e1 = Vector{Int32}(undef, nq_);  q_e2 = Vector{Int32}(undef, nq_)
    q_e3 = Vector{Int32}(undef, nq_);  q_e4 = Vector{Int32}(undef, nq_)
    q_wam2 = Vector{Int32}(undef, nq_);  q_wam1 = Vector{Int32}(undef, nq_)
    q_wap2 = Vector{Int32}(undef, nq_);  q_wap3 = Vector{Int32}(undef, nq_)
    q_wbm2 = Vector{Int32}(undef, nq_);  q_wbm1 = Vector{Int32}(undef, nq_)
    q_wbp2 = Vector{Int32}(undef, nq_);  q_wbp3 = Vector{Int32}(undef, nq_)
    q_weno_interior = Vector{Int8}(undef, nq_)

    b_ex1 = Vector{Int32}(undef, nb_);  b_ex2 = Vector{Int32}(undef, nb_)
    b_ex3 = Vector{Int32}(undef, nb_);  b_ex4 = Vector{Int32}(undef, nb_)
    b_ey1 = Vector{Int32}(undef, nb_);  b_ey2 = Vector{Int32}(undef, nb_)
    b_ey3 = Vector{Int32}(undef, nb_);  b_ey4 = Vector{Int32}(undef, nb_)
    b_ez1 = Vector{Int32}(undef, nb_);  b_ez2 = Vector{Int32}(undef, nb_)
    b_ez3 = Vector{Int32}(undef, nb_);  b_ez4 = Vector{Int32}(undef, nb_)

    b_wxm2 = Vector{Int32}(undef, nb_);  b_wxm1 = Vector{Int32}(undef, nb_)
    b_wxp2 = Vector{Int32}(undef, nb_);  b_wxp3 = Vector{Int32}(undef, nb_)
    b_wym2 = Vector{Int32}(undef, nb_);  b_wym1 = Vector{Int32}(undef, nb_)
    b_wyp2 = Vector{Int32}(undef, nb_);  b_wyp3 = Vector{Int32}(undef, nb_)
    b_wzm2 = Vector{Int32}(undef, nb_);  b_wzm1 = Vector{Int32}(undef, nb_)
    b_wzp2 = Vector{Int32}(undef, nb_);  b_wzp3 = Vector{Int32}(undef, nb_)

    b_wx0 = Vector{Int32}(undef, nb_);
    b_wxp1 = Vector{Int32}(undef, nb_);
    b_wy0 = Vector{Int32}(undef, nb_);
    b_wyp1 = Vector{Int32}(undef, nb_);
    b_wz0 = Vector{Int32}(undef, nb_);
    b_wzp1 = Vector{Int32}(undef, nb_);

    b_weno_interior = Vector{Int8}(undef, nb_)

    # ── wedge_01: edge-indexed section ───────────────────────────────────
    for e in 1:ne_
        x, y, z, align = edge_to_coord(s, e)
        e_src[e] = Int32(src(s, x, y, z, align))
        e_tgt[e] = Int32(tgt(s, x, y, z, align))

        # Interior test: need offsets -2 and +3 along edge's own axis.
        # src is offset 0, tgt is offset +1.
        interior = if align == X_ALIGN
            (x > 3) & (x <= nx_ - 3)
        elseif align == Y_ALIGN
            (y > 3) & (y <= ny_ - 3)
        else # Z_ALIGN
            (z > 3) & (z <= nz_ - 3)
        end

        e_weno_interior[e] = Int8(interior)
        if interior
            # Vertices along edge's own axis at offsets -2, -1 (behind src)
            # and +2, +3 (beyond tgt).  src = offset 0, tgt = offset +1.
            if align == X_ALIGN
                e_wm2[e] = Int32(coord_to_vert(s, x - 2, y, z))
                e_wm1[e] = Int32(coord_to_vert(s, x - 1, y, z))
                e_wp2[e] = Int32(coord_to_vert(s, x + 2, y, z))
                e_wp3[e] = Int32(coord_to_vert(s, x + 3, y, z))
            elseif align == Y_ALIGN
                e_wm2[e] = Int32(coord_to_vert(s, x, y - 2, z))
                e_wm1[e] = Int32(coord_to_vert(s, x, y - 1, z))
                e_wp2[e] = Int32(coord_to_vert(s, x, y + 2, z))
                e_wp3[e] = Int32(coord_to_vert(s, x, y + 3, z))
            else # Z_ALIGN
                e_wm2[e] = Int32(coord_to_vert(s, x, y, z - 2))
                e_wm1[e] = Int32(coord_to_vert(s, x, y, z - 1))
                e_wp2[e] = Int32(coord_to_vert(s, x, y, z + 2))
                e_wp3[e] = Int32(coord_to_vert(s, x, y, z + 3))
            end
        else
            e_wm2[e] = e_wm1[e] = e_wp2[e] = e_wp3[e] = Int32(1)
        end
    end

    # ── wedge_11: quad-indexed section ───────────────────────────────────
    for q in 1:nq_
        x, y, z, align = quad_to_coord(s, q)
        e1, e2, e3, e4 = quad_edges(s, x, y, z, align)
        q_e1[q] = Int32(e1);  q_e2[q] = Int32(e2)
        q_e3[q] = Int32(e3);  q_e4[q] = Int32(e4)

        # Interior test: need offsets ±2,±3 along both active axes.
        # Active axes and their coordinate ranges per quad type:
        #   Z_ALIGN: axis A = x (1..nxq), axis B = y (1..nyq); z unconstrained
        #   Y_ALIGN: axis A = x (1..nxq), axis B = z (1..nzq); y unconstrained
        #   X_ALIGN: axis A = y (1..nyq), axis B = z (1..nzq); x unconstrained
        interior = if align == Z_ALIGN
            (x > 3) & (x < nx_ - 3) & (y > 3) & (y < ny_ - 3)
        elseif align == Y_ALIGN
            (x > 3) & (x < nx_ - 3) & (z > 3) & (z < nz_ - 3)
        else # X_ALIGN
            (y > 3) & (y < ny_ - 3) & (z > 3) & (z < nz_ - 3)
        end

        q_weno_interior[q] = Int8(interior)
        if interior
            if align == Z_ALIGN
                # Axis A: X_ALIGN edges, offset along x
                q_wam2[q] = Int32(quad_edge_offset_3D(s, x, y, z, Z_ALIGN, X_ALIGN, -2))
                q_wam1[q] = Int32(quad_edge_offset_3D(s, x, y, z, Z_ALIGN, X_ALIGN, -1))
                q_wap2[q] = Int32(quad_edge_offset_3D(s, x, y, z, Z_ALIGN, X_ALIGN,  2))
                q_wap3[q] = Int32(quad_edge_offset_3D(s, x, y, z, Z_ALIGN, X_ALIGN,  3))
                # Axis B: Y_ALIGN edges, offset along y
                q_wbm2[q] = Int32(quad_edge_offset_3D(s, x, y, z, Z_ALIGN, Y_ALIGN, -2))
                q_wbm1[q] = Int32(quad_edge_offset_3D(s, x, y, z, Z_ALIGN, Y_ALIGN, -1))
                q_wbp2[q] = Int32(quad_edge_offset_3D(s, x, y, z, Z_ALIGN, Y_ALIGN,  2))
                q_wbp3[q] = Int32(quad_edge_offset_3D(s, x, y, z, Z_ALIGN, Y_ALIGN,  3))
            elseif align == Y_ALIGN
                # Axis B: X_ALIGN edges, offset along x
                q_wbm2[q] = Int32(quad_edge_offset_3D(s, x, y, z, Y_ALIGN, X_ALIGN, -2))
                q_wbm1[q] = Int32(quad_edge_offset_3D(s, x, y, z, Y_ALIGN, X_ALIGN, -1))
                q_wbp2[q] = Int32(quad_edge_offset_3D(s, x, y, z, Y_ALIGN, X_ALIGN,  2))
                q_wbp3[q] = Int32(quad_edge_offset_3D(s, x, y, z, Y_ALIGN, X_ALIGN,  3))
                # Axis A: Z_ALIGN edges, offset along z
                q_wam2[q] = Int32(quad_edge_offset_3D(s, x, y, z, Y_ALIGN, Z_ALIGN, -2))
                q_wam1[q] = Int32(quad_edge_offset_3D(s, x, y, z, Y_ALIGN, Z_ALIGN, -1))
                q_wap2[q] = Int32(quad_edge_offset_3D(s, x, y, z, Y_ALIGN, Z_ALIGN,  2))
                q_wap3[q] = Int32(quad_edge_offset_3D(s, x, y, z, Y_ALIGN, Z_ALIGN,  3))
            else # X_ALIGN
                # Axis A: Y_ALIGN edges, offset along y
                q_wam2[q] = Int32(quad_edge_offset_3D(s, x, y, z, X_ALIGN, Y_ALIGN, -2))
                q_wam1[q] = Int32(quad_edge_offset_3D(s, x, y, z, X_ALIGN, Y_ALIGN, -1))
                q_wap2[q] = Int32(quad_edge_offset_3D(s, x, y, z, X_ALIGN, Y_ALIGN,  2))
                q_wap3[q] = Int32(quad_edge_offset_3D(s, x, y, z, X_ALIGN, Y_ALIGN,  3))
                # Axis B: Z_ALIGN edges, offset along z
                q_wbm2[q] = Int32(quad_edge_offset_3D(s, x, y, z, X_ALIGN, Z_ALIGN, -2))
                q_wbm1[q] = Int32(quad_edge_offset_3D(s, x, y, z, X_ALIGN, Z_ALIGN, -1))
                q_wbp2[q] = Int32(quad_edge_offset_3D(s, x, y, z, X_ALIGN, Z_ALIGN,  2))
                q_wbp3[q] = Int32(quad_edge_offset_3D(s, x, y, z, X_ALIGN, Z_ALIGN,  3))
            end
        else
            q_wam2[q] = q_wam1[q] = q_wap2[q] = q_wap3[q] = Int32(1)
            q_wbm2[q] = q_wbm1[q] = q_wbp2[q] = q_wbp3[q] = Int32(1)
        end
    end

    # ── wedge_12: boid-indexed section ───────────────────────────────────
    for b in 1:nb_
        x, y, z = boid_to_coord(s, b)

        # 4 x-aligned primal edges (y ∈ {y, y+1}, z ∈ {z, z+1})
        b_ex1[b] = Int32(coord_to_edge(s, x, y,     z,     X_ALIGN))
        b_ex2[b] = Int32(coord_to_edge(s, x, y + 1, z,     X_ALIGN))
        b_ex3[b] = Int32(coord_to_edge(s, x, y,     z + 1, X_ALIGN))
        b_ex4[b] = Int32(coord_to_edge(s, x, y + 1, z + 1, X_ALIGN))

        # 4 y-aligned primal edges (x ∈ {x, x+1}, z ∈ {z, z+1})
        b_ey1[b] = Int32(coord_to_edge(s, x,     y, z,     Y_ALIGN))
        b_ey2[b] = Int32(coord_to_edge(s, x + 1, y, z,     Y_ALIGN))
        b_ey3[b] = Int32(coord_to_edge(s, x,     y, z + 1, Y_ALIGN))
        b_ey4[b] = Int32(coord_to_edge(s, x + 1, y, z + 1, Y_ALIGN))

        # 4 z-aligned primal edges (x ∈ {x, x+1}, y ∈ {y, y+1})
        b_ez1[b] = Int32(coord_to_edge(s, x,     y,     z, Z_ALIGN))
        b_ez2[b] = Int32(coord_to_edge(s, x + 1, y,     z, Z_ALIGN))
        b_ez3[b] = Int32(coord_to_edge(s, x,     y + 1, z, Z_ALIGN))
        b_ez4[b] = Int32(coord_to_edge(s, x + 1, y + 1, z, Z_ALIGN))

        b_wx0[b]  = Int32(boid_quad_offset(s, x, y, z, X_ALIGN, 0))
        b_wxp1[b] = Int32(boid_quad_offset(s, x, y, z, X_ALIGN, 1))
        b_wy0[b]  = Int32(boid_quad_offset(s, x, y, z, Y_ALIGN, 0))
        b_wyp1[b] = Int32(boid_quad_offset(s, x, y, z, Y_ALIGN, 1))
        b_wz0[b]  = Int32(boid_quad_offset(s, x, y, z, Z_ALIGN, 0))
        b_wzp1[b] = Int32(boid_quad_offset(s, x, y, z, Z_ALIGN, 1))

        # Interior test: need offsets -2 and +3 along all three boid axes.
        interior = (x > 3) & (x < nxb_ - 3) &
                   (y > 3) & (y < nyb_ - 3) &
                   (z > 3) & (z < nzb_ - 3)

        b_weno_interior[b] = Int8(interior)
        if interior
            # x-axis: X_ALIGN (YZ) quads; boid (x,y,z) lies between
            # X_ALIGN quad at x and x+1, so offset 0 → quad at x+1,
            # offset -1 → quad at x, offset +1 → quad at x+2, etc.
            # We use the convention: the "current" quad pair is x and x+1,
            # so the stencil is centred on x+1 (the east face of the boid).
            # Offsets -2,-1 are behind (west), +2,+3 are ahead (east).
            b_wxm2[b] = Int32(boid_quad_offset(s, x, y, z, X_ALIGN, -2))
            b_wxm1[b] = Int32(boid_quad_offset(s, x, y, z, X_ALIGN, -1))
            b_wxp2[b] = Int32(boid_quad_offset(s, x, y, z, X_ALIGN,  2))
            b_wxp3[b] = Int32(boid_quad_offset(s, x, y, z, X_ALIGN,  3))
            # y-axis: Y_ALIGN (XZ) quads
            b_wym2[b] = Int32(boid_quad_offset(s, x, y, z, Y_ALIGN, -2))
            b_wym1[b] = Int32(boid_quad_offset(s, x, y, z, Y_ALIGN, -1))
            b_wyp2[b] = Int32(boid_quad_offset(s, x, y, z, Y_ALIGN,  2))
            b_wyp3[b] = Int32(boid_quad_offset(s, x, y, z, Y_ALIGN,  3))
            # z-axis: Z_ALIGN (XY) quads
            b_wzm2[b] = Int32(boid_quad_offset(s, x, y, z, Z_ALIGN, -2))
            b_wzm1[b] = Int32(boid_quad_offset(s, x, y, z, Z_ALIGN, -1))
            b_wzp2[b] = Int32(boid_quad_offset(s, x, y, z, Z_ALIGN,  2))
            b_wzp3[b] = Int32(boid_quad_offset(s, x, y, z, Z_ALIGN,  3))
        else
            b_wxm2[b] = b_wxm1[b] = b_wxp2[b] = b_wxp3[b] = Int32(1)
            b_wym2[b] = b_wym1[b] = b_wyp2[b] = b_wyp3[b] = Int32(1)
            b_wzm2[b] = b_wzm1[b] = b_wzp2[b] = b_wzp3[b] = Int32(1)
        end
    end

    return WENO5Cache3D(
        ne_,  e_src, e_tgt, e_wm2, e_wm1, e_wp2, e_wp3, e_weno_interior,
        nq_,  q_e1, q_e2, q_e3, q_e4,
              q_wam2, q_wam1, q_wap2, q_wap3,
              q_wbm2, q_wbm1, q_wbp2, q_wbp3,
              q_weno_interior,
        nb_,  b_ex1, b_ex2, b_ex3, b_ex4,
              b_ey1, b_ey2, b_ey3, b_ey4,
              b_ez1, b_ez2, b_ez3, b_ez4,
              b_wxm2, b_wxm1, b_wxp2, b_wxp3,
              b_wym2, b_wym1, b_wyp2, b_wyp3,
              b_wzm2, b_wzm1, b_wzp2, b_wzp3,
              b_wx0, b_wxp1, b_wy0, b_wyp1, b_wz0, b_wzp1,
              b_weno_interior)
end

# ── wedge_01: single pass, edge-indexed ───────────────────────────────────────
#
# Identical structure to 2D: WENO5 reconstructs the 0-form at the edge
# midpoint from 6 vertices along the edge's own axis, then multiplies by
# the 1-form value on that edge.  All three edge alignments (X/Y/Z) are
# handled uniformly through the cache arrays.
#
@kernel function kernel_weno5_wedge_01_3d_cached!(res,
        @Const(e_src), @Const(e_tgt),
        @Const(e_wm2), @Const(e_wm1),
        @Const(e_wp2), @Const(e_wp3),
        @Const(e_weno_interior),
        eps, @Const(f0), @Const(f1))
    e = @index(Global)
    @inbounds begin
        f1_val = f1[e]
        f0_recon = if Bool(e_weno_interior[e])
            vm2 = f0[e_wm2[e]];  vm1 = f0[e_wm1[e]]
            v0  = f0[e_src[e]];  vp1 = f0[e_tgt[e]]
            vp2 = f0[e_wp2[e]];  vp3 = f0[e_wp3[e]]
            if f1_val >= 0
                weno5_point(vm2, vm1, v0,  vp1, vp2, eps)
            else
                weno5_point(vp3, vp2, vp1, v0,  vm1, eps)
            end
        else
            (f0[e_src[e]] + f0[e_tgt[e]]) * eltype(f1)(0.5)
        end
        res[e] = f0_recon * f1_val
    end
end

function wedge_product_01!(res::AbstractVector{FT}, ::WENO5, cache::WENO5Cache3D,
        f0::AbstractVector{FT}, f1::AbstractVector{FT};
        eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
    backend = get_backend(f1)
    kernel_weno5_wedge_01_3d_cached!(backend)(
        res,
        cache.e_src, cache.e_tgt,
        cache.e_wm2, cache.e_wm1, cache.e_wp2, cache.e_wp3,
        cache.e_weno_interior,
        eps, f0, f1; ndrange = cache.nedges_)
    return res
end

function wedge_product_01(::WENO5, cache::WENO5Cache3D,
        f0::AbstractVector{FT}, f1::AbstractVector{FT};
        eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
    res = KernelAbstractions.zeros(get_backend(f1), FT, cache.nedges_)
    return wedge_product_01!(res, WENO5(), cache, f0, f1; eps)
end

# ── wedge_11: direction-split WENO5, quad-indexed ─────────────────────────────
#
# Three-pass design matching the 2D split (halves register pressure per pass).
#
# For each quad type the two reconstruction axes and their edge roles are:
#
#   Z_ALIGN (XY quad):
#     axis A = x: f1b_a0 = e1 (X-edge at y),   f1b_ap1 = e3 (X-edge at y+1)
#                 avg_a_flow from e2 (Y-edge at x+1) and e4 (Y-edge at x)
#     axis B = y: f1b_b0 = e4 (Y-edge at x),   f1b_bp1 = e2 (Y-edge at x+1)
#                 avg_b_flow from e1 (X-edge at y)   and e3 (X-edge at y+1)
#
#   Y_ALIGN (XZ quad):
#     axis A = x: f1b_a0 = e4 (X-edge at z),   f1b_ap1 = e2 (X-edge at z+1)
#                 avg_a_flow from e1 (Z-edge at x) and e3 (Z-edge at x+1)
#     axis B = z: f1b_b0 = e1 (Z-edge at x),   f1b_bp1 = e3 (Z-edge at x+1)
#                 avg_b_flow from e4 (X-edge at z)   and e2 (X-edge at z+1)
#
#   X_ALIGN (YZ quad):
#     axis A = y: f1b_a0 = e1 (Y-edge at z),   f1b_ap1 = e3 (Y-edge at z+1)
#                 avg_a_flow from e2 (Z-edge at y+1) and e4 (Z-edge at y)
#     axis B = z: f1b_b0 = e4 (Z-edge at y),   f1b_bp1 = e2 (Z-edge at y+1)
#                 avg_b_flow from e1 (Y-edge at z)   and e3 (Y-edge at z+1)
#
# All dual edges go low→high, so the combine step is:
#   res[q] = a_upwind * avg_a_flow + b_upwind * avg_b_flow
#
# Pass 1 — axis-A reconstruction → a_upwind[q]
@kernel function kernel_weno5_11_3d_pass_a!(a_upwind,
        @Const(q_e1), @Const(q_e2), @Const(q_e3), @Const(q_e4),
        @Const(q_wam2), @Const(q_wam1), @Const(q_wap2), @Const(q_wap3),
        @Const(q_weno_interior),
        eps, @Const(f1a), @Const(f1b))
    q = @index(Global)
    @inbounds begin
        # avg_a_flow: for Z/Y quads, the Y/Z-aligned pair; for X quads, the Z pair.
        # In all cases those are e2 and e4 in our quad_edges ordering.
        avg_a_flow = (f1a[q_e2[q]] + f1a[q_e4[q]]) * eltype(f1a)(0.5)
        if Bool(q_weno_interior[q])
            # f1b_a0 = e1, f1b_ap1 = e3  (the axis-A lower and upper edges)
            f1b_am2 = f1b[q_wam2[q]];  f1b_am1 = f1b[q_wam1[q]]
            f1b_a0  = f1b[q_e1[q]];    f1b_ap1 = f1b[q_e3[q]]
            f1b_ap2 = f1b[q_wap2[q]];  f1b_ap3 = f1b[q_wap3[q]]
            a_upwind[q] = if avg_a_flow >= 0
                weno5_point(f1b_am2, f1b_am1, f1b_a0,  f1b_ap1, f1b_ap2, eps)
            else
                weno5_point(f1b_ap3, f1b_ap2, f1b_ap1, f1b_a0,  f1b_am1, eps)
            end
        else
            a_upwind[q] = ifelse(avg_a_flow >= 0, f1b[q_e1[q]], f1b[q_e3[q]])
        end
    end
end

# Pass 2 — axis-B reconstruction → b_upwind[q]
@kernel function kernel_weno5_11_3d_pass_b!(b_upwind,
        @Const(q_e1), @Const(q_e2), @Const(q_e3), @Const(q_e4),
        @Const(q_wbm2), @Const(q_wbm1), @Const(q_wbp2), @Const(q_wbp3),
        @Const(q_weno_interior),
        eps, @Const(f1a), @Const(f1b))
    q = @index(Global)
    @inbounds begin
        # avg_b_flow: the axis-B pair, which is e1 and e3 in our ordering.
        avg_b_flow = (f1a[q_e1[q]] + f1a[q_e3[q]]) * eltype(f1a)(0.5)
        if Bool(q_weno_interior[q])
            # f1b_b0 = e4, f1b_bp1 = e2  (the axis-B lower and upper edges)
            f1b_bm2 = f1b[q_wbm2[q]];  f1b_bm1 = f1b[q_wbm1[q]]
            f1b_b0  = f1b[q_e4[q]];    f1b_bp1 = f1b[q_e2[q]]
            f1b_bp2 = f1b[q_wbp2[q]];  f1b_bp3 = f1b[q_wbp3[q]]
            b_upwind[q] = if avg_b_flow >= 0
                weno5_point(f1b_bm2, f1b_bm1, f1b_b0,  f1b_bp1, f1b_bp2, eps)
            else
                weno5_point(f1b_bp3, f1b_bp2, f1b_bp1, f1b_b0,  f1b_bm1, eps)
            end
        else
            b_upwind[q] = ifelse(avg_b_flow >= 0, f1b[q_e4[q]], f1b[q_e2[q]])
        end
    end
end

# Pass 3 — combine
@kernel function kernel_weno5_11_3d_combine!(res,
        @Const(q_e1), @Const(q_e2), @Const(q_e3), @Const(q_e4),
        @Const(a_upwind), @Const(b_upwind),
        @Const(f1a))
    q = @index(Global)
    @inbounds begin
        avg_a_flow = (f1a[q_e2[q]] + f1a[q_e4[q]]) * eltype(f1a)(0.5)
        avg_b_flow = (f1a[q_e1[q]] + f1a[q_e3[q]]) * eltype(f1a)(0.5)
        res[q] = b_upwind[q] * avg_b_flow - a_upwind[q] * avg_a_flow
    end
end

function wedge_product_11!(res::AbstractVector{FT},
        tmp_a::AbstractVector{FT}, tmp_b::AbstractVector{FT},
        ::WENO5, cache::WENO5Cache3D,
        f1a::AbstractVector{FT}, f1b::AbstractVector{FT};
        eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
    backend = get_backend(f1a)
    kernel_weno5_11_3d_pass_a!(backend)(
        tmp_a,
        cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
        cache.q_wam2, cache.q_wam1, cache.q_wap2, cache.q_wap3,
        cache.q_weno_interior, eps, f1a, f1b; ndrange = cache.nquads_)
    kernel_weno5_11_3d_pass_b!(backend)(
        tmp_b,
        cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
        cache.q_wbm2, cache.q_wbm1, cache.q_wbp2, cache.q_wbp3,
        cache.q_weno_interior, eps, f1a, f1b; ndrange = cache.nquads_)
    kernel_weno5_11_3d_combine!(backend)(
        res,
        cache.q_e1, cache.q_e2, cache.q_e3, cache.q_e4,
        tmp_a, tmp_b, f1a; ndrange = cache.nquads_)
    return res
end

function wedge_product_11!(res::AbstractVector{FT}, ::WENO5, cache::WENO5Cache3D,
        f1a::AbstractVector{FT}, f1b::AbstractVector{FT};
        eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
    backend = get_backend(f1a)
    tmp_a = KernelAbstractions.zeros(backend, FT, cache.nquads_)
    tmp_b = KernelAbstractions.zeros(backend, FT, cache.nquads_)
    return wedge_product_11!(res, tmp_a, tmp_b, WENO5(), cache, f1a, f1b; eps)
end

function wedge_product_11(::WENO5, cache::WENO5Cache3D,
        f1a::AbstractVector{FT}, f1b::AbstractVector{FT};
        eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
    res = KernelAbstractions.zeros(get_backend(f1a), FT, cache.nquads_)
    return wedge_product_11!(res, WENO5(), cache, f1a, f1b; eps)
end

# ── wedge_12: direction-split WENO5, boid-indexed ─────────────────────────────
#
# Four-pass design: x-recon, y-recon, z-recon, combine.
#
# Velocity at each boid centre is the average of the 4 primal 1-form values
# on the 4 parallel edges of that alignment that bound the boid.
#
# The WENO stencil for each axis uses the 2-form (f2) values on a line of
# quads parallel to that axis passing through the boid:
#
#   x-axis: X_ALIGN (YZ) quads at x-2, x-1, x, x+1, x+2, x+3
#              where quad at x   = boid_quad_offset(s,x,y,z,X_ALIGN, 0)
#                    quad at x+1 = boid_quad_offset(s,x,y,z,X_ALIGN, 1)
#   y-axis: Y_ALIGN (XZ) quads at y-2, y-1, y, y+1, y+2, y+3
#   z-axis: Z_ALIGN (XY) quads at z-2, z-1, z, z+1, z+2, z+3
#
# The "current" quad pair for boid (x,y,z) along x is X_ALIGN quads at x
# and x+1; we reconstruct to the interface between them (offset 0 and +1).
# The stencil stored in the cache covers offsets -2,-1 (behind) and +2,+3
# (ahead), with offset 0 → boid_quad_offset(0) and +1 → boid_quad_offset(1)
# recovered live via the boid index and coord_to_quad — but since we have
# already cached those as specific named quad indices, we store offset 0 and
# +1 implicitly as the live reads from boid_quad_offset(0) and (1):
#
#   f2_x0  = f2[coord_to_quad(s, x,   y, z, X_ALIGN)]  (west face)
#   f2_xp1 = f2[coord_to_quad(s, x+1, y, z, X_ALIGN)]  (east face)
#
# These are not in the cache but are recovered from boid_quads, so we store
# the west and east X_ALIGN quad indices directly in the boid section using
# the existing b_ex* velocity edge slots — no: those are edge indices.
# We need the two "centre" quads as well.  We add them inline in the kernel
# by recalculating from the precomputed stencil arrays:
#
#   f2_x0  is at the same position as b_wxm2 shifted by +2, i.e. offset 0.
#   Equivalently: boid_quad_offset at offset 0 = coord_to_quad(x, y, z, X_ALIGN).
#
# To avoid live index arithmetic in the kernel, we cache offset 0 and +1
# as b_wx0 and b_wxp1 (and similarly for y/z).  These are inexpensive to add.
#
# ── Additional cache fields for "centre" quads ───────────────────────────────
#
# We extend the cache with six more arrays rather than pollute the kernel with
# live arithmetic.  These are appended to WENO5Cache3D as b_wx0, b_wxp1,
# b_wy0, b_wyp1, b_wz0, b_wzp1.
#
# Rather than change the already-written struct (and risk breaking the
# constructor), we note that these six values can be derived exactly from the
# stencil arrays already present:
#
#   b_wx0[b]  = boid_quad_offset at +0 = boid_quad_offset at -2 + stride 2
#             but stride arithmetic is unsafe in kernels without coordinates.
#
# The cleanest approach is to extend the struct.  The struct and constructor
# above should have these fields; they are added here and the constructor
# fills them.  The kernels below use them.
#
# ── REVISED struct fields (add after b_wzp3 / before b_weno_interior) ────────
#
#   b_wx0  :: IT;  b_wxp1 :: IT   -- X_ALIGN quads at x offset 0 and +1
#   b_wy0  :: IT;  b_wyp1 :: IT   -- Y_ALIGN quads at y offset 0 and +1
#   b_wz0  :: IT;  b_wzp1 :: IT   -- Z_ALIGN quads at z offset 0 and +1
#
# These are always valid (no interior guard needed) since every boid has
# valid quads on all six faces by definition.
#
# NOTE: The constructor loop already computes boid_quad_offset for -2,-1,+2,+3.
# Add the following inside the boid loop (unconditionally, before the interior
# check):
#
#   b_wx0[b]  = Int32(boid_quad_offset(s, x, y, z, X_ALIGN, 0))
#   b_wxp1[b] = Int32(boid_quad_offset(s, x, y, z, X_ALIGN, 1))
#   b_wy0[b]  = Int32(boid_quad_offset(s, x, y, z, Y_ALIGN, 0))
#   b_wyp1[b] = Int32(boid_quad_offset(s, x, y, z, Y_ALIGN, 1))
#   b_wz0[b]  = Int32(boid_quad_offset(s, x, y, z, Z_ALIGN, 0))
#   b_wzp1[b] = Int32(boid_quad_offset(s, x, y, z, Z_ALIGN, 1))
#
# Pass 1 — x-direction reconstruction → x_upwind[b]
@kernel function kernel_weno5_12_3d_pass_x!(x_upwind,
        @Const(b_ex1), @Const(b_ex2), @Const(b_ex3), @Const(b_ex4),
        @Const(b_wxm2), @Const(b_wxm1),
        @Const(b_wx0),  @Const(b_wxp1),
        @Const(b_wxp2), @Const(b_wxp3),
        @Const(b_weno_interior),
        eps, @Const(f1), @Const(f2))
    b = @index(Global)
    @inbounds begin
        avg_x_flow = (f1[b_ex1[b]] + f1[b_ex2[b]] +
                      f1[b_ex3[b]] + f1[b_ex4[b]]) * eltype(f1)(0.25)
        if Bool(b_weno_interior[b])
            f2_xm2 = f2[b_wxm2[b]];  f2_xm1 = f2[b_wxm1[b]]
            f2_x0  = f2[b_wx0[b]];   f2_xp1 = f2[b_wxp1[b]]
            f2_xp2 = f2[b_wxp2[b]];  f2_xp3 = f2[b_wxp3[b]]
            x_upwind[b] = if avg_x_flow >= 0
                weno5_point(f2_xm2, f2_xm1, f2_x0,  f2_xp1, f2_xp2, eps)
            else
                weno5_point(f2_xp3, f2_xp2, f2_xp1, f2_x0,  f2_xm1, eps)
            end
        else
            x_upwind[b] = ifelse(avg_x_flow >= 0, f2[b_wx0[b]], f2[b_wxp1[b]])
        end
    end
end

# Pass 2 — y-direction reconstruction → y_upwind[b]
@kernel function kernel_weno5_12_3d_pass_y!(y_upwind,
        @Const(b_ey1), @Const(b_ey2), @Const(b_ey3), @Const(b_ey4),
        @Const(b_wym2), @Const(b_wym1),
        @Const(b_wy0),  @Const(b_wyp1),
        @Const(b_wyp2), @Const(b_wyp3),
        @Const(b_weno_interior),
        eps, @Const(f1), @Const(f2))
    b = @index(Global)
    @inbounds begin
        avg_y_flow = (f1[b_ey1[b]] + f1[b_ey2[b]] +
                      f1[b_ey3[b]] + f1[b_ey4[b]]) * eltype(f1)(0.25)
        if Bool(b_weno_interior[b])
            f2_ym2 = f2[b_wym2[b]];  f2_ym1 = f2[b_wym1[b]]
            f2_y0  = f2[b_wy0[b]];   f2_yp1 = f2[b_wyp1[b]]
            f2_yp2 = f2[b_wyp2[b]];  f2_yp3 = f2[b_wyp3[b]]
            y_upwind[b] = if avg_y_flow >= 0
                weno5_point(f2_ym2, f2_ym1, f2_y0,  f2_yp1, f2_yp2, eps)
            else
                weno5_point(f2_yp3, f2_yp2, f2_yp1, f2_y0,  f2_ym1, eps)
            end
        else
            y_upwind[b] = ifelse(avg_y_flow >= 0, f2[b_wy0[b]], f2[b_wyp1[b]])
        end
    end
end

# Pass 3 — z-direction reconstruction → z_upwind[b]
@kernel function kernel_weno5_12_3d_pass_z!(z_upwind,
        @Const(b_ez1), @Const(b_ez2), @Const(b_ez3), @Const(b_ez4),
        @Const(b_wzm2), @Const(b_wzm1),
        @Const(b_wz0),  @Const(b_wzp1),
        @Const(b_wzp2), @Const(b_wzp3),
        @Const(b_weno_interior),
        eps, @Const(f1), @Const(f2))
    b = @index(Global)
    @inbounds begin
        avg_z_flow = (f1[b_ez1[b]] + f1[b_ez2[b]] +
                      f1[b_ez3[b]] + f1[b_ez4[b]]) * eltype(f1)(0.25)
        if Bool(b_weno_interior[b])
            f2_zm2 = f2[b_wzm2[b]];  f2_zm1 = f2[b_wzm1[b]]
            f2_z0  = f2[b_wz0[b]];   f2_zp1 = f2[b_wzp1[b]]
            f2_zp2 = f2[b_wzp2[b]];  f2_zp3 = f2[b_wzp3[b]]
            z_upwind[b] = if avg_z_flow >= 0
                weno5_point(f2_zm2, f2_zm1, f2_z0,  f2_zp1, f2_zp2, eps)
            else
                weno5_point(f2_zp3, f2_zp2, f2_zp1, f2_z0,  f2_zm1, eps)
            end
        else
            z_upwind[b] = ifelse(avg_z_flow >= 0, f2[b_wz0[b]], f2[b_wzp1[b]])
        end
    end
end

# Pass 4 — combine: all dual edges low→high, so all terms are positive
@kernel function kernel_weno5_12_3d_combine!(res,
        @Const(b_ex1), @Const(b_ex2), @Const(b_ex3), @Const(b_ex4),
        @Const(b_ey1), @Const(b_ey2), @Const(b_ey3), @Const(b_ey4),
        @Const(b_ez1), @Const(b_ez2), @Const(b_ez3), @Const(b_ez4),
        @Const(x_upwind), @Const(y_upwind), @Const(z_upwind),
        @Const(f1))
    b = @index(Global)
    @inbounds begin
        avg_x_flow = (f1[b_ex1[b]] + f1[b_ex2[b]] +
                      f1[b_ex3[b]] + f1[b_ex4[b]]) * eltype(f1)(0.25)
        avg_y_flow = (f1[b_ey1[b]] + f1[b_ey2[b]] +
                      f1[b_ey3[b]] + f1[b_ey4[b]]) * eltype(f1)(0.25)
        avg_z_flow = (f1[b_ez1[b]] + f1[b_ez2[b]] +
                      f1[b_ez3[b]] + f1[b_ez4[b]]) * eltype(f1)(0.25)
        res[b] = x_upwind[b] * avg_x_flow +
                 y_upwind[b] * avg_y_flow +
                 z_upwind[b] * avg_z_flow
    end
end

function wedge_product_12!(res::AbstractVector{FT},
        tmp_x::AbstractVector{FT}, tmp_y::AbstractVector{FT}, tmp_z::AbstractVector{FT},
        ::WENO5, cache::WENO5Cache3D,
        f1::AbstractVector{FT}, f2::AbstractVector{FT};
        eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
    backend = get_backend(f1)
    kernel_weno5_12_3d_pass_x!(backend)(
        tmp_x,
        cache.b_ex1, cache.b_ex2, cache.b_ex3, cache.b_ex4,
        cache.b_wxm2, cache.b_wxm1,
        cache.b_wx0,  cache.b_wxp1,
        cache.b_wxp2, cache.b_wxp3,
        cache.b_weno_interior, eps, f1, f2; ndrange = cache.nboids_)
    kernel_weno5_12_3d_pass_y!(backend)(
        tmp_y,
        cache.b_ey1, cache.b_ey2, cache.b_ey3, cache.b_ey4,
        cache.b_wym2, cache.b_wym1,
        cache.b_wy0,  cache.b_wyp1,
        cache.b_wyp2, cache.b_wyp3,
        cache.b_weno_interior, eps, f1, f2; ndrange = cache.nboids_)
    kernel_weno5_12_3d_pass_z!(backend)(
        tmp_z,
        cache.b_ez1, cache.b_ez2, cache.b_ez3, cache.b_ez4,
        cache.b_wzm2, cache.b_wzm1,
        cache.b_wz0,  cache.b_wzp1,
        cache.b_wzp2, cache.b_wzp3,
        cache.b_weno_interior, eps, f1, f2; ndrange = cache.nboids_)
    kernel_weno5_12_3d_combine!(backend)(
        res,
        cache.b_ex1, cache.b_ex2, cache.b_ex3, cache.b_ex4,
        cache.b_ey1, cache.b_ey2, cache.b_ey3, cache.b_ey4,
        cache.b_ez1, cache.b_ez2, cache.b_ez3, cache.b_ez4,
        tmp_x, tmp_y, tmp_z, f1; ndrange = cache.nboids_)
    return res
end

function wedge_product_12!(res::AbstractVector{FT}, ::WENO5, cache::WENO5Cache3D,
        f1::AbstractVector{FT}, f2::AbstractVector{FT};
        eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
    backend = get_backend(f1)
    tmp_x = KernelAbstractions.zeros(backend, FT, cache.nboids_)
    tmp_y = KernelAbstractions.zeros(backend, FT, cache.nboids_)
    tmp_z = KernelAbstractions.zeros(backend, FT, cache.nboids_)
    return wedge_product_12!(res, tmp_x, tmp_y, tmp_z, WENO5(), cache, f1, f2; eps)
end

function wedge_product_12(::WENO5, cache::WENO5Cache3D,
        f1::AbstractVector{FT}, f2::AbstractVector{FT};
        eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
    res = KernelAbstractions.zeros(get_backend(f1), FT, cache.nboids_)
    return wedge_product_12!(res, WENO5(), cache, f1, f2; eps)
end

# ── Factory and Val dispatch ──────────────────────────────────────────────────

AdvectionCache(::WENO5, s::UniformCubicalComplex3D) = WENO5Cache3D(s)

wedge_product(::Val{0}, ::Val{1}, ::WENO5, cache::WENO5Cache3D, f0, f1) =
    wedge_product_01(WENO5(), cache, f0, f1)

wedge_product(::Val{1}, ::Val{0}, ::WENO5, cache::WENO5Cache3D, f1, f0) =
    wedge_product_01(WENO5(), cache, f0, f1)

wedge_product(::Val{1}, ::Val{1}, ::WENO5, cache::WENO5Cache3D, f1a, f1b) =
    wedge_product_11(WENO5(), cache, f1a, f1b)

wedge_product(::Val{1}, ::Val{2}, ::WENO5, cache::WENO5Cache3D, f1, f2) =
    wedge_product_12(WENO5(), cache, f1, f2)