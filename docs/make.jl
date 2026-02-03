using Documenter
using CombinatorialSpaces

makedocs(
  sitename = "CombinatorialSpaces.jl",
  format = Documenter.HTML(
    size_threshold_ignore = [
      "meshes.md",
      "mesh_decomposition.md"
    ]
  ),
  modules = [CombinatorialSpaces],
  checkdocs = :exports,
  warnonly = true,
  pages = [
    "simplicial_sets.md",
    "discrete_exterior_calculus.md",
    "combinatorial_maps.md",
    "grid_laplace.md",
    "meshes.md",
    "euler.md",
    "mg_benchmarks.md",
    "mesh_decomposition.md"
  ]
)

deploydocs(
  target = "build",
  repo = "github.com/AlgebraicJulia/CombinatorialSpaces.jl.git",
  branch = "gh-pages",
  devbranch = "main"
)
