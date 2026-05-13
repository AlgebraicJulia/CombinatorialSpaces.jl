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

# ── WENO5Cache ────────────────────────────────────────────────────────────────────
struct WENO5Cache{IT <: AbstractVector{Int32}, MT <: AbstractVector{Int8}} <: AbstractAdvectionCache
  nquads_ :: Int
  q_e1 :: IT;  q_e2 :: IT;  q_e3 :: IT;  q_e4 :: IT
  # Extended stencil — x-aligned edges at y-offsets {-2,-1,+2,+3} from each quad
  q_wxm2 :: IT;  q_wxm1 :: IT;  q_wxp2 :: IT;  q_wxp3 :: IT
  # Extended stencil — y-aligned edges at x-offsets {-2,-1,+2,+3} from each quad
  q_wym2 :: IT;  q_wym1 :: IT;  q_wyp2 :: IT;  q_wyp3 :: IT
  # 1 = full WENO5 stencil in bounds, 0 = boundary quad (falls back to upwinding)
  q_weno_interior :: MT
end
Adapt.@adapt_structure WENO5Cache

function WENO5Cache(s::UniformCubicalComplex2D)
  nq_  = nquads(s);  nx_ = nx(s);  ny_ = ny(s)
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
    interior = (x > 2) & (x < nx_ - 2) & (y > 2) & (y < ny_ - 2)
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
  return WENO5Cache(nq_, q_e1, q_e2, q_e3, q_e4,
                    q_wxm2, q_wxm1, q_wxp2, q_wxp3,
                    q_wym2, q_wym1, q_wyp2, q_wyp3,
                    q_weno_interior)
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
