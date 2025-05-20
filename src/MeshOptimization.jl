module MeshOptimization

using ..SimplicialSets

using GeometryBasics: Point3d
using LinearAlgebra
using Parameters
using Random
using SparseArrays
using StaticArrays: SVector

export optimize_mesh!, AbstractMeshOptimizer, SimulatedAnnealing, equilaterality

# Given a 2D simplicial set, return the edge lengths per triangle.
function edge_lengths(s)
  map(s[:∂e0], s[:∂e1], s[:∂e2]) do e0, e1, e2
    [volume(1,s,e0), volume(1,s,e1), volume(1,s,e2)]
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

"""    function equilaterality(s::HasDeltaSet2D)

Compute the sum of squared equilaterality of each triangle over a mesh.

0 implies all triangles are equilateral.
"""
function equilaterality(s::HasDeltaSet2D)
  sum(abs2, equilateralities(edge_lengths(s)))
end

abstract type AbstractMeshOptimizer end

""" struct SimulatedAnnealing

The simulated annealing algorithm's parameters for mesh optimization.

**Keyword arguments:**

`ϵ`: Multiply the randomly-sampled j ~ N(0,1) by this number. (Defaults to `1e-3`)

`epochs`: The number of times to iterate over each point in the mesh. (Defaults to `100`)

`debug_epochs`: When debugging is active, print `cost` every `debug_epochs` epochs. (Defaults to `10`)

`hold_boundaries`: Whether to hold the boundaries of the mesh fixed. (Defaults to `true`)

`anneal`: Whether to accept jitters than increase the cost function. (Defaults to `true`)

`jitter3D`: Whether to jitter in the z-dimension, too. (Defaults to `false`)

`spherical`:  Whether to constrain the new jittered point to lie on the sphere. (Defaults to `false`)

`radius`:  If `spherical` is `true`, the radius of the sphere. (Defaults to `1.0`)

`cost`: The cost function to use when annealing. (Defaults to `equilaterality`.)

`cooling_schedule`: The cooling schedule to be used when annealing. (Defaults to `linear_cooling_schedule`.)

See also: [`optimize_mesh!`](@ref).
"""
@with_kw struct SimulatedAnnealing
  ϵ::AbstractFloat           = 1e-3
  epochs::Integer            = 100
  debug_epochs::Integer      = 10
  hold_boundaries::Bool      = true
  anneal::Bool               = true
  jitter3D::Bool             = false
  spherical::Bool            = false
  radius::AbstractFloat      = 1.0
  cost::Function             = equilaterality
  cooling_schedule::Function = linear_cooling_schedule
end

optimize_mesh!(s::HasDeltaSet2D) = optimize_mesh!(s, SimulatedAnnealing())

linear_cooling_schedule(epochs, epoch) = range(0.05, .001; length=epochs)[epoch]

# Extract the point attribute of the ACSet.
function optimize_mesh!(s::EmbeddedDeltaSet2D{_o, point_type} where _o, alg::SimulatedAnnealing) where point_type
  jitter3D, ϵ = alg.jitter3D, alg.ϵ
  function noise_generator()
    if length(point_type) == 3
      jitter3D ?
        point_type(randn(3) * ϵ...) :
        point_type(randn(2) * ϵ..., 0)
    else
      point_type(randn(2) * ϵ...)
    end
  end
  optimize_mesh!(s, alg, noise_generator)
end

# TODO: Support the 3D analog directly on tetrahedra, or indirectly using the equilaterality of triangles.
# TODO: Explore the effect of exp(-(temp_eq-orig_eq) / temperature)
# TODO: The default cost function is computed over the entire mesh twice;
# if the cost function is known to be local, this could be ameliorated.
"""    function optimize_mesh!(s::HasDeltaSet2D, alg::SimulatedAnnealing, noise_generator::Function)

Optimize the given mesh using a simulated annealing algorithm.

Note that the selection probability is directly calculated (without exp), and does not depend on the magnitude of error improvement.

See also: [`SimulatedAnnealing`](@ref).
"""
function optimize_mesh!(s::HasDeltaSet2D, alg::SimulatedAnnealing, noise_generator::Function)
  @unpack_SimulatedAnnealing alg
  int = interior(Val{0}, s)
  map(1:epochs) do epoch
    # TODO: You could vectorize (with a MVN) instead of iterating over points.
    for v in (hold_boundaries ? int : vertices(s))
      original = s[v, :point]
      jitter = noise_generator()
      jittered = spherical ?
        normalize(original + jitter)*radius :
        original + jitter
      orig_eq = cost(s)
      # Observe that we edit the mesh here:
      s[v, :point] = jittered
      temp_eq = cost(s)
      # Accept this change, or undo it.
      jump_anyway = anneal && (rand() < cooling_schedule(epochs, epoch))
      if temp_eq < orig_eq || jump_anyway
        s[v, :point] = jittered
      else
        s[v, :point] = original
      end
    end
    epoch % debug_epochs == 0 && @debug "Cost at epoch $(epoch): $(cost(s))"
    cost(s)
  end
end

end # module MeshOptimization

