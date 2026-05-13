module TestUnitful
using Test

using CombinatorialSpaces

using Unitful: unit, ustrip, @u_str
using LinearAlgebra: Diagonal, diag
using StaticArrays: SVector

@testset "Unitful density 0-form hodge star in 1D" begin
  # Unitful Point3 coordinates in meters produce meter-valued primal edge lengths.
  len_t = typeof(1.0u"m")
  point3m_primal = EmbeddedDeltaSet1D{Bool,Point3{len_t}}()
  add_vertices!(point3m_primal, 3, point=[
    Point3(0.0u"m", 0.0u"m", 0.0u"m"),
    Point3(1.0u"m", 0.0u"m", 0.0u"m"),
    Point3(3.0u"m", 0.0u"m", 0.0u"m"),
  ])
  add_edges!(point3m_primal, [1,2], [2,3], edge_orientation=true)
  point3m_dual = EmbeddedDeltaDualComplex1D{Bool,len_t,Point3{len_t}}(point3m_primal)
  subdivide_duals!(point3m_dual, Barycenter())
  @test volume(point3m_dual, E(1:2)) == [1.0, 2.0] .* u"m"

  # ⋆(0): VForm[U] → DualForm{1}[U·m]
  # Operator diagonal entries = dual edge lengths [m].
  # inv_hodge_star(0): DualForm{1}[U·m] → VForm[U]
  # Operator diagonal entries = 1/(dual edge length) [m⁻¹].
  str0 = ⋆(0, point3m_dual)
  invstr0 = inv_hodge_star(0, point3m_dual)
  @test str0    == Diagonal([0.5, 1.5, 1.0] .* u"m")
  @test invstr0 == Diagonal([2.0, 2/3, 1.0] .* u"m^-1")

  # Input: density c [kg/m] (mass per length on a 1D mesh).
  # ⋆(0) [m] · c [kg/m] = ⋆(c) [kg]  (integrates density over the dual cell).
  # inv_hodge_star(0) [m⁻¹] · ⋆(c) [kg] = c [kg/m]  (recovers the original density).
  density_vform = VForm([2.0, 4.0, 6.0] .* u"kg/m")
  @test str0 * density_vform == DualForm{1}([1.0, 6.0, 6.0] .* u"kg")
  @test invstr0 * str0 * density_vform == density_vform

  # ⋆(1): EForm[U] → DualForm{0}[U·m⁻¹]
  # Operator diagonal entries = 1/edge_length [m⁻¹] (hodge_diag(1, s, e) = 1/vol(e)).
  # inv_hodge_star(1): DualForm{0}[U·m⁻¹] → EForm[U]
  # Operator diagonal entries = edge_length [m].
  str1 = ⋆(1, point3m_dual)
  invstr1 = inv_hodge_star(1, point3m_dual)
  @test str1    == Diagonal([1.0, 0.5] .* u"m^-1")
  @test invstr1 == Diagonal([1.0, 2.0] .* u"m")

  # Input: volumetric flux u [m²/s] (area per time on a 1D mesh).
  # ⋆(1) [m⁻¹] · u [m²/s] = ũ [m/s]  (converts flux to velocity at dual vertex).
  # inv_hodge_star(1) [m] · ũ [m/s] = u [m²/s]  (recovers the original flux).
  flux_eform = EForm([2.0, 4.0] .* u"m^2/s")
  @test str1 * flux_eform == DualForm{0}([2.0, 2.0] .* u"m/s")
  @test invstr1 * str1 * flux_eform == flux_eform

  # Dual exterior derivative d̃₀: DualForm{0} → DualForm{1}.
  # The matrix contains only ±1 entries (no geometric factors), so it is
  # dimensionless and preserves the units of its input.
  # Connectivity: v1 ← +q_e1, v2 ← −q_e1 + q_e2, v3 ← −q_e2.
  # Input: velocity q̃ [m/s] at dual 0-cells (edge midpoints).
  # Output: DualForm{1}[m/s] at dual 1-cells (primal vertices).
  vel_dual0 = DualForm{0}([3.0, 1.0] .* u"m/s")
  @test d(point3m_dual, vel_dual0) == DualForm{1}([3.0, -2.0, -1.0] .* u"m/s")

  edge_lengths = volume(point3m_dual, E(1:2))

  # Primal exterior derivative d₀: VForm[U] → EForm[U].
  # The matrix contains only ±1 entries, so it is dimensionless and preserves units.
  # Concentration gradient example:
  #   Input:  c [kg/m] (linear density at primal vertices).
  #   d₀ [dimensionless] · c = Δc [kg/m]  (difference of concentrations along each edge).
  #   Dividing by edge length [m] gives the physical gradient ∇c [kg/m²].
  conc_vform = VForm([2.0, 4.0, 6.0] .* u"kg/m")
  conc_diff = d(point3m_primal, conc_vform)
  @test conc_diff == EForm([2.0, 2.0] .* u"kg/m")
  conc_grad = conc_diff.data ./ edge_lengths
  @test conc_grad ≈ [2.0, 1.0] .* u"kg/m^2"
  @test all(unit.(conc_grad) .== unit(1.0u"kg/m^2"))
  @test ustrip.(u"kg/m^2", conc_grad) ≈ [2.0, 1.0]

  # 1D potential flow: ϕ [m²/s] → d₀(ϕ) [m²/s] → u = d₀(ϕ)/L [m/s].
  # The two primal edges have lengths 1 m and 2 m respectively.
  # d₀ [dimensionless] · ϕ [m²/s] = dΦ [m²/s]  (flux differences along edges).
  # Dividing by edge length [m] gives velocity [m/s].
  Φ = VForm([1.0, 4.0, 10.0] .* u"m^2/s")
  @test all(unit.(Φ.data) .== unit(1.0u"m^2/s"))
  dΦ = d(point3m_primal, Φ)
  @test dΦ == EForm([3.0, 6.0] .* u"m^2/s")
  @test all(unit.(dΦ.data) .== unit(1.0u"m^2/s"))
  u = dΦ.data ./ edge_lengths
  @test u ≈ [3.0, 3.0] .* u"m/s"
  @test all(unit.(u) .== unit(1.0u"m/s"))
  @test ustrip.(u"m/s", u) ≈ [3.0, 3.0]

  # ♯ (sharp) operator: EForm[U] → vector field at vertices/edge-midpoints.
  # On a meter-scale mesh, dΦ [m²/s] is divided by edge length [m], giving [m/s].
  # The unitless version of dΦ is divided by [m⁻¹] (inverse length), also giving [m/s].
  u_d = ♯(point3m_dual, dΦ.data, PDSharp())
  u_d0 = ♯(point3m_dual, ustrip.(u"m^2/s", dΦ.data), PDSharp())
  @test all(all(unit.(Tuple(v)) .== unit(1.0u"m/s")) for v in u_d)
  @test all(all(isapprox.(ustrip.(u"m/s", Tuple(v)),
                          ustrip.(u"m^-1", Tuple(v0))))
            for (v, v0) in zip(u_d, u_d0))

  u_p = ♯(point3m_dual, dΦ.data, PPSharp())
  u_p0 = ♯(point3m_dual, ustrip.(u"m^2/s", dΦ.data), PPSharp())
  @test all(all(unit.(Tuple(v)) .== unit(1.0u"m/s")) for v in u_p)
  @test all(all(isapprox.(ustrip.(u"m/s", Tuple(v)),
                          ustrip.(u"m^-1", Tuple(v0))))
            for (v, v0) in zip(u_p, u_p0))
end

function test_unitful_dec_operators_2d(subdivision)
  tri_height = √3/2  # height of an equilateral triangle with unit side length
  len_t = typeof(1.0u"m")
  area_t = typeof(1.0u"m^2")
  geom_t = Union{len_t, area_t}
  primal_s = EmbeddedDeltaSet2D{Bool,Point3{len_t}}()
  add_vertices!(primal_s, 3, point=[
    Point3(0.0u"m", 0.0u"m", 0.0u"m"),
    Point3(1.0u"m", 0.0u"m", 0.0u"m"),
    Point3(0.5u"m", tri_height * u"m", 0.0u"m"),
  ])
  glue_triangle!(primal_s, 1, 2, 3, tri_orientation=true)
  primal_s[:edge_orientation] = true
  s = EmbeddedDeltaDualComplex2D{Bool,geom_t,Point3{len_t}}(primal_s)
  subdivide_duals!(s, subdivision)

  # Plain (non-unitful) mesh with identical geometry (1 unit = 1 m).
  # Used to provide the non-unitful reference pathway for equality checks.
  plain_primal_s = EmbeddedDeltaSet2D{Bool,Point3d}()
  add_vertices!(plain_primal_s, 3, point=[
    Point3d(0.0, 0.0, 0.0),
    Point3d(1.0, 0.0, 0.0),
    Point3d(0.5, tri_height, 0.0),
  ])
  glue_triangle!(plain_primal_s, 1, 2, 3, tri_orientation=true)
  plain_primal_s[:edge_orientation] = true
  plain_s = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}(plain_primal_s)
  subdivide_duals!(plain_s, subdivision)

  @test volume(s, E(1:3)) ≈ [1.0, 1.0, 1.0] .* u"m"
  @test volume(s, Tri(1)) ≈ (tri_height/2)u"m^2"

  # ⋆(0): VForm[U] → DualForm{2}[U·m²]
  # Operator diagonal entries = dual 2-cell area at each primal vertex [m²].
  # inv_hodge_star(0): DualForm{2}[U·m²] → VForm[U]
  # Operator diagonal entries = 1/(dual area) [m⁻²].
  str0 = ⋆(0, s)
  invstr0 = inv_hodge_star(0, s)
  @test all(unit.(diag(str0)) .== unit(1.0u"m^2"))
  @test all(unit.(diag(invstr0)) .== unit(1.0u"m^-2"))

  # Input: surface density ρ [kg/m²].
  # ⋆(0) [m²] · ρ [kg/m²] = ⋆(ρ) [kg]  (integrates density over the dual cell area).
  # inv_hodge_star(0) [m⁻²] · ⋆(ρ) [kg] = ρ [kg/m²]  (recovers the original density).
  density_vform = VForm([2.0, 4.0, 6.0] .* u"kg/m^2")
  @test all(unit.(str0 * density_vform) .== unit(1.0u"kg"))
  @test (invstr0 * (str0 * density_vform)) ≈ density_vform.data

  # ⋆(2): TriForm[U] → DualForm{0}[U·m⁻²]
  # Operator diagonal entries = 1/(primal triangle area) [m⁻²].
  # inv_hodge_star(2): DualForm{0}[U·m⁻²] → TriForm[U]
  # Operator diagonal entries = primal triangle area [m²].
  str2 = ⋆(2, s)
  invstr2 = inv_hodge_star(2, s)
  @test all(unit.(diag(str2)) .== unit(1.0u"m^-2"))
  @test all(unit.(diag(invstr2)) .== unit(1.0u"m^2"))

  # Input: mass m [kg] (integrated over each triangle).
  # ⋆(2) [m⁻²] · m [kg] = ⋆(m) [kg/m²]  (converts mass to surface density).
  # inv_hodge_star(2) [m²] · ⋆(m) [kg/m²] = m [kg]  (recovers the original mass).
  mass_tform = TriForm([3.0] .* u"kg")
  @test all(unit.(str2 * mass_tform) .== unit(1.0u"kg/m^2"))
  @test (invstr2 * (str2 * mass_tform)) ≈ mass_tform.data

  # ⋆(1) (GeometricHodge): EForm[U] → DualForm{1}[U]
  # Operator entries are ratios of dual edge length to primal edge length [dimensionless].
  # In 2D, inv_hodge_star(1) = -⋆(1)⁻¹ (the inverse carries an additional sign),
  # so that inv⋆₁ ∘ ⋆₁ = -I on primal 1-forms.
  str1 = ⋆(1, s)
  invstr1 = inv_hodge_star(1, s)

  # Input: mass flux q [kg/s] (a primal 1-form).
  # ⋆(1) [dimensionless] · q [kg/s] = q̃ [kg/s]  (rotates flux to the dual edge).
  # inv_hodge_star(1) · q̃ [kg/s] = −q [kg/s]  (the 2D sign convention).
  q = EForm([2.0, 4.0, 6.0] .* u"kg/s")
  q̃ = str1 * q
  @test all(unit.(q̃) .== unit(1.0u"kg/s"))
  @test ustrip.(u"kg/s", q̃) ≈ (⋆(1, plain_s) * EForm([2.0, 4.0, 6.0]))
  q_back = invstr1 * q̃
  @test all(unit.(q_back) .== unit(1.0u"kg/s"))
  @test q_back ≈ -q.data

  q̃_back = str1 * (invstr1 * q̃)
  @test all(unit.(q̃_back) .== unit(1.0u"kg/s"))
  @test q̃_back ≈ -q̃

  # Laplace-Beltrami operator Δ = d ∘ δ on a 2D meter-scale mesh.
  # δ = inv⋆ ∘ d ∘ ⋆ picks up a factor of [m⁻²] from the hodge operators:
  #   Δ(ρ [kg/m²]) → [kg/m⁴]       (0-form: two applications of ⋆(0)/⋆(2))
  #   Δ(q [kg/s])  → [kg/(m²·s)]   (1-form: applications of ⋆(1) and ⋆(0)/⋆(2))
  #   Δ(m [kg])    → [kg/m²]        (2-form: two applications of ⋆(2)/⋆(0))
  ρ = VForm([1.0, 2.0, 3.0] .* u"kg/m^2")
  q_form = EForm([1.0, 2.0, 1.0] .* u"kg/s")
  m = TriForm([1.0] .* u"kg")
  lap0 = Δ(s, ρ)
  lap1 = Δ(s, q_form)
  lap1_diag = Δ(s, q_form; hodge=DiagonalHodge())
  lap2 = Δ(s, m)
  lap2_diag = Δ(s, m; hodge=DiagonalHodge())
  @test lap0 isa VForm
  @test lap1 isa EForm
  @test lap1_diag isa EForm
  @test lap2 isa TriForm
  @test lap2_diag isa TriForm
  @test all(unit.(lap0.data) .== unit(1.0u"kg/m^4"))
  @test all(unit.(lap1.data) .== unit(1.0u"kg/m^2/s"))
  @test all(unit.(lap1_diag.data) .== unit(1.0u"kg/m^2/s"))
  @test all(unit.(lap2.data) .== unit(1.0u"kg/m^2"))
  @test all(unit.(lap2_diag.data) .== unit(1.0u"kg/m^2"))

  # ♭ (flat) operator: vector field [U] → EForm.
  # PPFlat: projects vertex-based vectors onto primal edges via dot product with
  #   edge tangent vectors [m], so input [m] gives output [m²].
  # DPPFlat: projects triangle-based vectors similarly.
  # A dimensionless vector field on a meter-scale mesh acquires one factor of [m]
  # (one factor from the edge tangent), so m-unit input yields m²-unit output.
  test_vec = SVector(2.0, 3.0, 0.0)
  u_p_vals = fill(test_vec, nv(s))
  u_p = fill(test_vec .* 1.0u"m", nv(s))
  u_t_vals = fill(test_vec, ntriangles(s))
  u_t = fill(test_vec .* 1.0u"m", ntriangles(s))

  α_pp = ♭(s, u_p, PPFlat())
  α_dpp = ♭(s, u_t, DPPFlat())
  @test all(unit.(α_pp) .== unit(1.0u"m^2"))
  @test all(unit.(α_dpp) .== unit(1.0u"m^2"))
  @test ustrip.(u"m^2", α_pp) ≈ ♭(plain_s, u_p_vals, PPFlat())
  @test ustrip.(u"m^2", α_dpp) ≈ ♭(plain_s, u_t_vals, DPPFlat())

  # ♯ (sharp) operator: EForm[U·m] → vector field[U] at primal vertices.
  # PPSharp inverts ♭ PPFlat: input α_pp [m²] divided by edge tangent [m] gives [m].
  u_sharp = ♯(s, α_pp, PPSharp())
  @test all(all(unit.(Tuple(v)) .== unit(1.0u"m")) for v in u_sharp)

  # Dual exterior derivative d̃₁: DualForm{0} → DualForm{1}.
  # The matrix contains only ±1 entries (no geometric factors), so it is
  # dimensionless and preserves the units of its input.
  # DualForm{0} in 2D is indexed by primal triangles (outer dual 0-cells),
  # not by all dual vertices in the subdivided complex.
  # Input: velocity at dual 0-cells [m/s].
  # Output: DualForm{1}[m/s] at dual 1-cells (primal edges).
  vel_dual0 = DualForm{0}(collect(1.0:ntriangles(s)) .* u"m/s")
  dual_deriv = d(s, vel_dual0)
  dual_deriv_plain = d(plain_s, DualForm{0}(ustrip.(u"m/s", vel_dual0.data)))
  @test ustrip.(u"m/s", dual_deriv.data) == dual_deriv_plain.data
  @test all(unit.(dual_deriv.data) .== unit(1.0u"m/s"))

  # Wedge products: unit propagation.
  # All primal wedge-product weights are dimensionless ratios of dual volumes to
  # primal volumes (both measured in the same unit, m^k for a k-simplex), so the
  # output unit is simply the product of the two input form units.
  # Operator: dimensionless (no geometric unit factor).
  # Output unit = left_input_unit × right_input_unit.

  # Input 0-form: surface density [kg/m²] at each primal vertex.
  f_0 = VForm([1.0, 2.0, 3.0] .* u"kg/m^2")

  # 00 wedge: VForm[kg/m²] ∧ VForm[K] → VForm[kg⋅K/m²]
  # No geometric weights; output is the pointwise product f[x]⋅g[x].
  g_0 = VForm([3.0, 2.0, 1.0] .* u"K")
  w_00 = ∧(s, f_0, g_0)
  @test all(unit.(w_00.data) .== unit(1.0u"kg*K/m^2"))
  @test ustrip.(u"kg*K/m^2", w_00.data) ≈
        ∧(plain_s, VForm(ustrip.(u"kg/m^2", f_0.data)), VForm(ustrip.(u"K", g_0.data))).data

  # 01 wedge: VForm[kg/m²] ∧ EForm[m/s] → EForm[kg/(m⋅s)]
  # Weights: dual_volume(1, s, x′) / volume(1, s, e) = dual_edge_length [m] /
  #   primal_edge_length [m] = dimensionless.
  β_1 = EForm([1.0, 2.0, 3.0] .* u"m/s")
  w_01 = ∧(s, f_0, β_1)
  @test all(unit.(w_01.data) .== unit(1.0u"kg/(m*s)"))
  @test ustrip.(u"kg/(m*s)", w_01.data) ≈
        ∧(plain_s, VForm(ustrip.(u"kg/m^2", f_0.data)), EForm(ustrip.(u"m/s", β_1.data))).data

  # 10 wedge: EForm[m/s] ∧ VForm[kg/m²] → EForm[kg/(m⋅s)]
  # Same dimensionless weights as the 01 case, but with operands swapped.
  w_10 = ∧(s, β_1, f_0)
  @test all(unit.(w_10.data) .== unit(1.0u"kg/(m*s)"))
  @test ustrip.(u"kg/(m*s)", w_10.data) ≈
        ∧(plain_s, EForm(ustrip.(u"m/s", β_1.data)), VForm(ustrip.(u"kg/m^2", f_0.data))).data

  # 11 wedge: EForm[kg/s] ∧ EForm[m] → TriForm[kg⋅m/s]
  # Weights: sum_of_dual_areas(2) / primal_area(2) = [m²] / [m²] = dimensionless.
  α_1 = EForm([1.0, 2.0, 3.0] .* u"kg/s")
  β_1b = EForm([3.0, 2.0, 1.0] .* u"m")
  w_11 = ∧(s, α_1, β_1b)
  @test all(unit.(w_11.data) .== unit(1.0u"kg*m/s"))
  @test ustrip.(u"kg*m/s", w_11.data) ≈
        ∧(plain_s, EForm(ustrip.(u"kg/s", α_1.data)), EForm(ustrip.(u"m", β_1b.data))).data

  # 02 wedge: VForm[kg/m²] ∧ TriForm[m²] → TriForm[kg]
  # Weights: dual_volume(2, s, x′) / volume(2, s, t) = dual_face_area [m²] /
  #   primal_triangle_area [m²] = dimensionless.
  β_2 = TriForm([2.0] .* u"m^2")
  w_02 = ∧(s, f_0, β_2)
  @test all(unit.(w_02.data) .== unit(1.0u"kg"))
  @test ustrip.(u"kg", w_02.data) ≈
        ∧(plain_s, VForm(ustrip.(u"kg/m^2", f_0.data)), TriForm(ustrip.(u"m^2", β_2.data))).data

  # 20 wedge: TriForm[m²] ∧ VForm[kg/m²] → TriForm[kg]
  # Same dimensionless weights as the 02 case, but with operands swapped.
  w_20 = ∧(s, β_2, f_0)
  @test w_20 isa TriForm
  @test all(unit.(w_20.data) .== unit(1.0u"kg"))
  @test ustrip.(u"kg", w_20.data) ≈
        ∧(plain_s, TriForm(ustrip.(u"m^2", β_2.data)), VForm(ustrip.(u"kg/m^2", f_0.data))).data

  # Lie derivative: ℒ(X♭, α) maps EForm[U_X] × DualForm{0}[U_α] → DualForm{0}[U_X·U_α/m²].
  # Via Cartan's formula: ℒ_{X♭} α = i_{X♭}(d̃₀ α).
  # Unit chain: d̃₀ is dimensionless (±1 entries), ⋆₁⁻¹ is dimensionless (dual_edge_length /
  #   primal_edge_length = m/m), ∧₁₁ is dimensionless (m²/m²), ⋆₂ divides by area [m²].
  # For X♭[m/s] and α[kg]: output ∈ DualForm{0}[kg/(m·s)].
  X♭_lie = EForm(collect(1.0:ne(s)) .* u"m/s")
  α_lie  = DualForm{0}(collect(1.0:ntriangles(s)) .* u"kg")
  result_lie = ℒ(s, X♭_lie, α_lie; hodge=GeometricHodge())
  @test all(unit.(result_lie.data) .== unit(1.0u"kg/(m*s)"))
  result_lie_plain = ℒ(plain_s, EForm(ustrip.(u"m/s", X♭_lie.data)),
                       DualForm{0}(ustrip.(u"kg", α_lie.data)); hodge=GeometricHodge())
  @test ustrip.(u"kg/(m*s)", result_lie.data) ≈ result_lie_plain.data

  # Vorticity equation: dω/dt + ℒ(u♭, ω) = 0.
  # Physical interpretation:
  #   u♭ ∈ EForm[m²/s]: velocity 1-form (velocity [m/s] × length [m]).
  #   ω ∈ DualForm{0}[1/s]: vorticity (curl of velocity in 2D, units s⁻¹).
  #   ℒ(u♭, ω) ∈ DualForm{0}[1/s²]: vorticity advection [m²/s × s⁻¹ / m² = s⁻²].
  #   dω/dt ∈ DualForm{0}[1/s²]: time derivative of vorticity.
  # Dimensional consistency: ℒ(u♭, ω) must carry [1/s²].
  u_flow = EForm(collect(1.0:ne(s)) .* u"m^2/s")
  ω_vort = DualForm{0}(collect(1.0:ntriangles(s)) .* u"s^-1")
  lie_vort = ℒ(s, u_flow, ω_vort; hodge=GeometricHodge())
  @test all(unit.(lie_vort.data) .== unit(1.0u"s^-2"))
end

@testset "Unitful DEC operators in 2D" begin
  @testset "subdivision = $(nameof(typeof(subdivision)))" for subdivision in (Barycenter(), Circumcenter())
    test_unitful_dec_operators_2d(subdivision)
  end
end

function test_unitful_dec_operators_3d(subdivision)
  len_t = typeof(1.0u"m")
  area_t = typeof(1.0u"m^2")
  vol_t = typeof(1.0u"m^3")
  geom_t = Union{len_t, area_t, vol_t}
  primal_s = EmbeddedDeltaSet3D{Bool, Point3{len_t}}()
  add_vertices!(primal_s, 4, point=[
    Point3(0.0u"m", 0.0u"m", 0.0u"m"),
    Point3(1.0u"m", 0.0u"m", 0.0u"m"),
    Point3(0.0u"m", 1.0u"m", 0.0u"m"),
    Point3(0.0u"m", 0.0u"m", 1.0u"m"),
  ])
  glue_sorted_tetrahedron!(primal_s, 1, 2, 3, 4)
  s = EmbeddedDeltaDualComplex3D{Bool, geom_t, Point3{len_t}}(primal_s)
  subdivide_duals!(s, subdivision)

  # Plain (non-unitful) mesh with identical geometry (1 unit = 1 m).
  # Used to provide the non-unitful reference pathway for equality checks.
  plain_primal_s = EmbeddedDeltaSet3D{Bool, Point3d}()
  add_vertices!(plain_primal_s, 4, point=[
    Point3d(0.0, 0.0, 0.0),
    Point3d(1.0, 0.0, 0.0),
    Point3d(0.0, 1.0, 0.0),
    Point3d(0.0, 0.0, 1.0),
  ])
  glue_sorted_tetrahedron!(plain_primal_s, 1, 2, 3, 4)
  plain_s = EmbeddedDeltaDualComplex3D{Bool, Float64, Point3d}(plain_primal_s)
  subdivide_duals!(plain_s, subdivision)

  @test volume(s, Tet(1)) ≈ (1/6)u"m^3"
  # Edges 1-3 connect axis-aligned vertices (length √2); edges 4-6 are axis-aligned (length 1).
  @test volume(s, E(1:6)) ≈ [√2, √2, √2, 1.0, 1.0, 1.0] .* u"m"

  # ⋆(0): VForm[U] → DualForm{3}[U·m³]
  # Operator diagonal entries = dual 3-cell volume at each primal vertex [m³].
  # inv_hodge_star(0): DualForm{3}[U·m³] → VForm[U]
  # Operator diagonal entries = 1/(dual volume) [m⁻³].
  str0 = ⋆(0, s, DiagonalHodge())
  invstr0 = inv_hodge_star(0, s, DiagonalHodge())
  @test all(unit.(diag(str0)) .== unit(1.0u"m^3"))
  @test all(unit.(diag(invstr0)) .== unit(1.0u"m^-3"))
  @test ustrip.(u"m^3", diag(str0)) ≈ diag(⋆(0, plain_s, DiagonalHodge()))
  @test ustrip.(u"m^-3", diag(invstr0)) ≈ diag(inv_hodge_star(0, plain_s, DiagonalHodge()))

  # Input: volumetric density ρ [kg/m³].
  # ⋆(0) [m³] · ρ [kg/m³] = ⋆(ρ) [kg]  (integrates density over the dual cell volume).
  # inv_hodge_star(0) [m⁻³] · ⋆(ρ) [kg] = ρ [kg/m³]  (recovers the original density).
  density_vform = VForm([1.0, 2.0, 3.0, 4.0] .* u"kg/m^3")
  @test all(unit.(str0 * density_vform) .== unit(1.0u"kg"))
  @test (invstr0 * (str0 * density_vform)) ≈ density_vform.data

  # ⋆(3): TetForm[U] → DualForm{0}[U·m⁻³]
  # Operator diagonal entries = 1/(primal tet volume) [m⁻³].
  # inv_hodge_star(3): DualForm{0}[U·m⁻³] → TetForm[U]
  # Operator diagonal entries = primal tet volume [m³].
  str3 = ⋆(3, s, DiagonalHodge())
  invstr3 = inv_hodge_star(3, s, DiagonalHodge())
  @test all(unit.(diag(str3)) .== unit(1.0u"m^-3"))
  @test all(unit.(diag(invstr3)) .== unit(1.0u"m^3"))
  @test ustrip.(u"m^-3", diag(str3)) ≈ diag(⋆(3, plain_s, DiagonalHodge()))
  @test ustrip.(u"m^3", diag(invstr3)) ≈ diag(inv_hodge_star(3, plain_s, DiagonalHodge()))

  # Input: mass m [kg] (integrated over each tetrahedron).
  # ⋆(3) [m⁻³] · m [kg] = ⋆(m) [kg/m³]  (converts mass to volumetric density).
  # inv_hodge_star(3) [m³] · ⋆(m) [kg/m³] = m [kg]  (recovers the original mass).
  mass_tform = TetForm([3.0] .* u"kg")
  @test all(unit.(str3 * mass_tform) .== unit(1.0u"kg/m^3"))
  @test (invstr3 * (str3 * mass_tform)) ≈ mass_tform.data

  # ⋆(1) (DiagonalHodge): EForm[U] → DualForm{2}[U·m]
  # Operator diagonal entries = dual 2-area / primal edge length [m²/m = m].
  # inv_hodge_star(1): DualForm{2}[U·m] → EForm[U]
  # Operator diagonal entries = primal edge length / dual 2-area [m/m² = m⁻¹].
  str1 = ⋆(1, s, DiagonalHodge())
  invstr1 = inv_hodge_star(1, s, DiagonalHodge())
  @test all(unit.(diag(str1)) .== unit(1.0u"m"))
  @test all(unit.(diag(invstr1)) .== unit(1.0u"m^-1"))
  @test ustrip.(u"m", diag(str1)) ≈ diag(⋆(1, plain_s, DiagonalHodge()))
  @test ustrip.(u"m^-1", diag(invstr1)) ≈ diag(inv_hodge_star(1, plain_s, DiagonalHodge()))

  # ⋆(2) (DiagonalHodge): TriForm[U] → DualForm{1}[U·m⁻¹]
  # Operator diagonal entries = dual 1-length / primal triangle area [m/m² = m⁻¹].
  # inv_hodge_star(2): DualForm{1}[U·m⁻¹] → TriForm[U]
  # Operator diagonal entries = primal triangle area / dual 1-length [m²/m = m].
  str2 = ⋆(2, s, DiagonalHodge())
  invstr2 = inv_hodge_star(2, s, DiagonalHodge())
  @test all(unit.(diag(str2)) .== unit(1.0u"m^-1"))
  @test all(unit.(diag(invstr2)) .== unit(1.0u"m"))
  @test ustrip.(u"m^-1", diag(str2)) ≈ diag(⋆(2, plain_s, DiagonalHodge()))
  @test ustrip.(u"m", diag(invstr2)) ≈ diag(inv_hodge_star(2, plain_s, DiagonalHodge()))

  # Laplace-de Rham operator Δ = dδ + δd on a 3D meter-scale mesh.
  # In 3D: Δ picks up a factor of [m⁻²] from the hodge operators.
  #   Δ(ρ [kg/m³])  → [kg/m⁵]       (0-form: ⋆(1)[m] and inv_⋆(0)[m⁻³])
  #   Δ(q [kg/m²s]) → [kg/(m⁴·s)]   (1-form)
  #   Δ(f [kg/ms])  → [kg/(m³·s)]   (2-form)
  #   Δ(m [kg/m])   → [kg/m³]        (3-form: dδ only, d of 3-form vanishes)
  ρ = VForm([1.0, 2.0, 3.0, 4.0] .* u"kg/m^3")
  q = EForm(collect(1.0:ne(s)) .* u"kg/m^2/s")
  f = TriForm(collect(1.0:ntriangles(s)) .* u"kg/m/s")
  m = TetForm([1.0] .* u"kg/m")
  lap0 = Δ(s, ρ; hodge=DiagonalHodge())
  lap1 = Δ(s, q; hodge=DiagonalHodge())
  lap2 = Δ(s, f; hodge=DiagonalHodge())
  lap3 = Δ(s, m; hodge=DiagonalHodge())
  @test lap0 isa VForm
  @test lap1 isa EForm
  @test lap2 isa TriForm
  @test lap3 isa TetForm
  @test all(unit.(lap0.data) .== unit(1.0u"kg/m^5"))
  @test all(unit.(lap1.data) .== unit(1.0u"kg/m^4/s"))
  @test all(unit.(lap2.data) .== unit(1.0u"kg/m^3/s"))
  @test all(unit.(lap3.data) .== unit(1.0u"kg/m^3"))

  # Wedge products: all primal-primal weights are dimensionless ratios
  # (dual k-volume / primal k-volume, same units cancel), so the output unit
  # is simply the product of the two input form units.

  f_0 = VForm([1.0, 2.0, 3.0, 4.0] .* u"kg/m^3")
  g_0 = VForm([4.0, 3.0, 2.0, 1.0] .* u"K")

  # 0∧0: VForm[kg/m³] ∧ VForm[K] → VForm[kg⋅K/m³]
  w_00 = ∧(s, f_0, g_0)
  @test all(unit.(w_00.data) .== unit(1.0u"kg*K/m^3"))
  @test ustrip.(u"kg*K/m^3", w_00.data) ≈
        ∧(plain_s, VForm(ustrip.(u"kg/m^3", f_0.data)), VForm(ustrip.(u"K", g_0.data))).data

  e_1 = EForm(collect(1.0:ne(s)) .* u"m/s")

  # 0∧1: VForm[kg/m³] ∧ EForm[m/s] → EForm[kg/(m²⋅s)]
  w_01 = ∧(s, f_0, e_1)
  @test all(unit.(w_01.data) .== unit(1.0u"kg/m^2/s"))
  @test ustrip.(u"kg/m^2/s", w_01.data) ≈
        ∧(plain_s, VForm(ustrip.(u"kg/m^3", f_0.data)), EForm(ustrip.(u"m/s", e_1.data))).data

  # 1∧0: EForm[m/s] ∧ VForm[kg/m³] → EForm[kg/(m²⋅s)]
  w_10 = ∧(s, e_1, f_0)
  @test all(unit.(w_10.data) .== unit(1.0u"kg/m^2/s"))
  @test ustrip.(u"kg/m^2/s", w_10.data) ≈
        ∧(plain_s, EForm(ustrip.(u"m/s", e_1.data)), VForm(ustrip.(u"kg/m^3", f_0.data))).data

  t_2 = TriForm(collect(1.0:ntriangles(s)) .* u"kg/m^2")

  # 0∧2: VForm[kg/m³] ∧ TriForm[kg/m²] → TriForm[kg²/m⁵]
  w_02 = ∧(s, f_0, t_2)
  @test all(unit.(w_02.data) .== unit(1.0u"kg^2/m^5"))
  @test ustrip.(u"kg^2/m^5", w_02.data) ≈
        ∧(plain_s, VForm(ustrip.(u"kg/m^3", f_0.data)), TriForm(ustrip.(u"kg/m^2", t_2.data))).data

  # 2∧0: TriForm[kg/m²] ∧ VForm[kg/m³] → TriForm[kg²/m⁵]
  w_20 = ∧(s, t_2, f_0)
  @test all(unit.(w_20.data) .== unit(1.0u"kg^2/m^5"))
  @test ustrip.(u"kg^2/m^5", w_20.data) ≈
        ∧(plain_s, TriForm(ustrip.(u"kg/m^2", t_2.data)), VForm(ustrip.(u"kg/m^3", f_0.data))).data

  tet_3 = TetForm([1.0] .* u"kg")

  # 0∧3: VForm[kg/m³] ∧ TetForm[kg] → TetForm[kg²/m³]
  w_03 = ∧(s, f_0, tet_3)
  @test all(unit.(w_03.data) .== unit(1.0u"kg^2/m^3"))
  @test ustrip.(u"kg^2/m^3", w_03.data) ≈
        ∧(plain_s, VForm(ustrip.(u"kg/m^3", f_0.data)), TetForm(ustrip.(u"kg", tet_3.data))).data

  e_1b = EForm(collect(1.0:ne(s)) .* u"kg/s")

  # 1∧1: EForm[kg/s] ∧ EForm[m/s] → TriForm[kg⋅m/s²]
  w_11 = ∧(s, e_1b, e_1)
  @test all(unit.(w_11.data) .== unit(1.0u"kg*m/s^2"))
  @test ustrip.(u"kg*m/s^2", w_11.data) ≈
        ∧(plain_s, EForm(ustrip.(u"kg/s", e_1b.data)), EForm(ustrip.(u"m/s", e_1.data))).data

  # 2∧1: TriForm[kg/m²] ∧ EForm[m/s] → TetForm[kg/(m⋅s)]
  w_21 = ∧(s, t_2, e_1)
  @test all(unit.(w_21.data) .== unit(1.0u"kg/m/s"))
  @test ustrip.(u"kg/m/s", w_21.data) ≈
        ∧(plain_s, TriForm(ustrip.(u"kg/m^2", t_2.data)), EForm(ustrip.(u"m/s", e_1.data))).data

  # 1∧2: EForm[m/s] ∧ TriForm[kg/m²] → TetForm[kg/(m⋅s)]
  w_12 = ∧(s, e_1, t_2)
  @test all(unit.(w_12.data) .== unit(1.0u"kg/m/s"))
  @test ustrip.(u"kg/m/s", w_12.data) ≈
        ∧(plain_s, EForm(ustrip.(u"m/s", e_1.data)), TriForm(ustrip.(u"kg/m^2", t_2.data))).data

  # Interior product i_{X♭}(α): a primal EForm (encoding a covector field) and a
  # dual n-form; yields a dual (n-1)-form.
  # All geometric weights come from DiagonalHodge operators applied internally, so
  # the output unit inherits the EForm unit × input DualForm unit × hodge factors.
  X♭ = EForm(collect(1.0:ne(s)) .* u"m/s")
  plain_X♭ = EForm(ustrip.(u"m/s", X♭.data))

  # ip(X♭[m/s], DualForm{1}[kg/s]; DiagonalHodge) → DualForm{0}[kg/(m⋅s²)]
  # Unit chain: inv_⋆(2)[m]: DualForm{1}[kg/s] → TriForm[kg⋅m/s];
  #   ∧(2,1)[1]: TriForm[kg⋅m/s] ∧ EForm[m/s] → TetForm[kg⋅m²/s²];
  #   ⋆(3)[m⁻³]: TetForm[kg⋅m²/s²] → DualForm{0}[kg/(m⋅s²)].
  df1 = DualForm{1}(collect(1.0:ntriangles(s)) .* u"kg/s")
  ip1 = interior_product(s, X♭, df1; hodge=DiagonalHodge())
  @test ip1 isa DualForm{0}
  @test all(unit.(ip1.data) .== unit(1.0u"kg/m/s^2"))
  ip1_plain = interior_product(plain_s, plain_X♭, DualForm{1}(ustrip.(u"kg/s", df1.data));
                               hodge=DiagonalHodge())
  @test ustrip.(u"kg/m/s^2", ip1.data) ≈ ip1_plain.data

  # ip(X♭[m/s], DualForm{2}[m²]; DiagonalHodge) → DualForm{1}[m/s]
  # Unit chain: inv_⋆(1)[m⁻¹]: DualForm{2}[m²] → EForm[m];
  #   ∧(1,1)[1]: EForm[m] ∧ EForm[m/s] → TriForm[m²/s];
  #   ⋆(2)[m⁻¹]: TriForm[m²/s] → DualForm{1}[m/s].
  df2 = DualForm{2}(collect(1.0:ne(s)) .* u"m^2")
  ip2 = interior_product(s, X♭, df2; hodge=DiagonalHodge())
  @test ip2 isa DualForm{1}
  @test all(unit.(ip2.data) .== unit(1.0u"m/s"))
  ip2_plain = interior_product(plain_s, plain_X♭, DualForm{2}(ustrip.(u"m^2", df2.data));
                               hodge=DiagonalHodge())
  @test ustrip.(u"m/s", ip2.data) ≈ ip2_plain.data

  # ip(X♭[m/s], DualForm{3}[kg/m³]; DiagonalHodge) → DualForm{2}[kg/(m⁴⋅s)]
  # Unit chain: inv_⋆(0)[m⁻³]: DualForm{3}[kg/m³] → VForm[kg/m⁶];
  #   ∧(0,1)[1]: VForm[kg/m⁶] ∧ EForm[m/s] → EForm[kg/(m⁵⋅s)];
  #   ⋆(1)[m]: EForm[kg/(m⁵⋅s)] → DualForm{2}[kg/(m⁴⋅s)].
  df3 = DualForm{3}(collect(1.0:nv(s)) .* u"kg/m^3")
  ip3 = interior_product(s, X♭, df3; hodge=DiagonalHodge())
  @test ip3 isa DualForm{2}
  @test all(unit.(ip3.data) .== unit(1.0u"kg/m^4/s"))
  ip3_plain = interior_product(plain_s, plain_X♭, DualForm{3}(ustrip.(u"kg/m^3", df3.data));
                               hodge=DiagonalHodge())
  @test ustrip.(u"kg/m^4/s", ip3.data) ≈ ip3_plain.data

  # Lie derivative ℒ_{X♭}(α) via Cartan's formula: ℒ = i∘d̃ + d̃∘i.
  # All dual-derivative operators (d̃) are dimensionless (±1 entries).

  # ℒ(X♭[m/s], DualForm{0}[kg]) → DualForm{0}[kg/(m⋅s)]
  # Via ℒ = i∘d̃₀: d̃₀[1] maps DualForm{0}[kg] → DualForm{1}[kg],
  #   then i(X♭[m/s], DualForm{1}[kg]) → DualForm{0}[kg/(m⋅s)].
  df0 = DualForm{0}(collect(1.0:ntetrahedra(s)) .* u"kg")
  lie0 = ℒ(s, X♭, df0; hodge=DiagonalHodge())
  @test lie0 isa DualForm{0}
  @test all(unit.(lie0.data) .== unit(1.0u"kg/m/s"))
  lie0_plain = ℒ(plain_s, plain_X♭, DualForm{0}(ustrip.(u"kg", df0.data));
                 hodge=DiagonalHodge())
  @test ustrip.(u"kg/m/s", lie0.data) ≈ lie0_plain.data

  # ℒ(X♭[m/s], DualForm{1}[kg/s]) → DualForm{1}[kg/(m⋅s²)]
  # Via ℒ = i∘d̃₁ + d̃₀∘i: both terms contribute DualForm{1}[kg/(m⋅s²)].
  lie1 = ℒ(s, X♭, df1; hodge=DiagonalHodge())
  @test lie1 isa DualForm{1}
  @test all(unit.(lie1.data) .== unit(1.0u"kg/m/s^2"))
  lie1_plain = ℒ(plain_s, plain_X♭, DualForm{1}(ustrip.(u"kg/s", df1.data));
                 hodge=DiagonalHodge())
  @test ustrip.(u"kg/m/s^2", lie1.data) ≈ lie1_plain.data

  # ℒ(X♭[m/s], DualForm{2}[m²]) → DualForm{2}[m/s]
  # Via ℒ = d̃₁∘i: i(X♭[m/s], DualForm{2}[m²]) → DualForm{1}[m/s],
  #   then d̃₁[1] maps DualForm{1}[m/s] → DualForm{2}[m/s].
  lie2 = ℒ(s, X♭, df2; hodge=DiagonalHodge())
  @test lie2 isa DualForm{2}
  @test all(unit.(lie2.data) .== unit(1.0u"m/s"))
  lie2_plain = ℒ(plain_s, plain_X♭, DualForm{2}(ustrip.(u"m^2", df2.data));
                 hodge=DiagonalHodge())
  @test ustrip.(u"m/s", lie2.data) ≈ lie2_plain.data
end

@testset "Unitful DEC operators in 3D" begin
  @testset "subdivision = $(nameof(typeof(subdivision)))" for subdivision in (Barycenter(), Circumcenter())
    test_unitful_dec_operators_3d(subdivision)
  end
end

end
