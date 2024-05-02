using Documenter
using CombinatorialSpaces

makedocs(
  sitename = "CombinatorialSpaces.jl",
  format = Documenter.HTML(
    size_threshold_ignore = [
      "meshes.md"
    ]
  ),
  modules = [CombinatorialSpaces],
  checkdocs = :exports,
  warnonly = true,
  pages = [
    "simplicial_sets.md",
    "discrete_exterior_calculus.md",
    "fast_dec.md",
    "combinatorial_maps.md",
    "meshes.md"
  ]
)

deploydocs(
  target = "build",
  repo = "github.com/AlgebraicJulia/CombinatorialSpaces.jl.git",
  branch = "gh-pages",
  devbranch = "main"
)
