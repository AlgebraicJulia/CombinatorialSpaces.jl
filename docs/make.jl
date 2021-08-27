using Documenter
using CombinatorialSpaces

makedocs(
  sitename = "CombinatorialSpaces.jl",
  format = Documenter.HTML(),
  modules = [CombinatorialSpaces],
  checkdocs = :exports,
  pages = [
    "simplicial_sets.md",
    "discrete_exterior_calculus.md",
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
