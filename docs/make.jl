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
    "combinatorial_maps.md",
    "meshes.md",
    "euler.md"
  ]
)

deploydocs(
  target = "build",
  repo = "github.com/AlgebraicJulia/CombinatorialSpaces.jl.git",
  branch = "gh-pages",
  devbranch = "main"
)
