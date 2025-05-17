module MeshOptimization

using ..SimplicialSets

using GeometryBasics: Point3d
using LinearAlgebra
using Parameters
using Random
using SparseArrays
using StaticArrays: SVector

export optimize_mesh!, AbstractMeshOptimizer, SimulatedAnnealing, equilaterality

# Given a 2D simplicial set and its 2nd boundary matrix,
# return the edge lengths per triangle.
function edge_lengths(s, ∂₂)
  rows = rowvals(∂₂)
  vals = nonzeros(∂₂)
  m, n = size(∂₂)
  map(1:n) do j
    [volume(1,s,rows[i]) for i in nzrange(∂₂, j)]
  end
end

"""    function equilaterality(e0, e1, e2)

Given three edge lengths, compute the failure to be equilateral, using:

```math
(E_0 - \bar{E})^2 + (E_1 - \bar{E})^2 + (E_1 - \bar{E})^2,
```

where ``E_0, E_1, E_2 = |e_0, e_1, e_2|_1``, and ``\bar{E}`` is their average.

0 implies all edge lengths are equal and the triangle is equilateral.
"""
function equilaterality(e0, e1, e2)
  e0, e1, e2 = normalize([e0, e1, e2], 1)
  avg = (e0+e1+e2)/3
  (e0 - avg)^2 + (e1 - avg)^2 + (e2 - avg)^2
end

# Compute the equilaterality of each triple of edge lengths.
equilateralities(els) = map(x -> equilaterality(x...), els)

"""    function equilaterality(s::HasDeltaSet2D, ∂₂=∂(2,s))

Compute the sum of squared equilaterality of each triangle over a mesh.

0 implies all triangles are equilateral.
"""
function equilaterality(s::HasDeltaSet2D, ∂₂=∂(2,s))
  sum(abs2, equilateralities(edge_lengths(s, ∂₂)))
end

abstract type AbstractMeshOptimizer end

""" struct SimulatedAnnealing

The simulated annealing algorithm's parameters for mesh optimization.

**Keyword arguments:**

`ϵ`: Multiply the randomly-sampled j ~ N(0,1) by this number. (Defaults to `1e-3`)

`epochs`: The number of times to iterate over each point in the mesh. (Defaults to `100`)

`hold_boundaries`: Whether to hold the boundaries of the mesh fixed. (Defaults to `true`)

`anneal`: Whether to accept jitters than increase the cost function. (Defaults to `true`)

`jitter3D`: Whether to jitter in the z-dimension, too. (Defaults to `false`)

`spherical`:  Whether to constrain the new jittered point to lie on the unit sphere. (Defaults to `false`)

`cost`: The cost function to use when annealing. (Defaults to `equilaterality`.)

See also: [`optimize_mesh!`](@ref).
"""
@with_kw struct SimulatedAnnealing
  ϵ::AbstractFloat      = 1e-3
  epochs::Integer       = 100
  hold_boundaries::Bool = true
  anneal::Bool          = true
  jitter3D::Bool        = false
  spherical::Bool       = false
  cost::Function        = equilaterality
end

# TODO: Optim.jl exports optimize!. Does that matter?
# TODO: Explore the effect of exp(-(temp_eq-orig_eq) / temperature)
# TODO: The default cost function is computed over the entire mesh twice;
# if the cost function is known to be local, this could be ameliorated.
"""    function optimize_mesh!(s::HasDeltaSet2D, alg::SimulatedAnnealing)

Optimize the given mesh using a simulated annealing algorithm.

Note that the selection probability is directly calculated (without exp), and does not depend on the magnitude of error improvement.

See also: [`SimulatedAnnealing`](@ref).
"""
function optimize_mesh!(s::HasDeltaSet2D, alg::SimulatedAnnealing)
  @unpack_SimulatedAnnealing alg
  ∂₂ = ∂(2,s)
  int = interior(Val{0}, s)
  cooling_schedule = range(0.05, .001; length=epochs)
  map(1:epochs, cooling_schedule) do epoch, temperature
    # TODO: You could vectorize (with a MVN) instead of iterating over points.
    for v in (hold_boundaries ? int : vertices(s))
      jitter = jitter3D ?
        Point3d(randn(3) * ϵ...) :
        Point3d(randn(2) * ϵ..., 0)
      original = s[v, :point]
      jittered = spherical ? normalize(original + jitter) : original + jitter
      orig_eq = cost(s, ∂₂)
      # Observe that we edit the mesh here:
      s[v, :point] = jittered
      temp_eq = cost(s, ∂₂)
      # Accept this change, or undo it.
      jump_anyway = anneal && (rand() < temperature)
      if temp_eq < orig_eq || jump_anyway
        s[v, :point] = jittered
      else
        s[v, :point] = original
      end
    end
    epoch % 100 == 0 && @debug "Cost at epoch $(epoch): $(cost(s, ∂₂))"
    cost(s, ∂₂)
  end
end

end # module MeshOptimization

