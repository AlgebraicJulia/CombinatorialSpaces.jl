using Documenter
using CombinatorialSpaces

makedocs(
  sitename = "CombinatorialSpaces",
  format = Documenter.HTML(),
  modules = [CombinatorialSpaces]
)

deploydocs(
  target = "build",
  repo = "github.com/AlgebraicJulia/CombinatorialSpaces.jl.git",
  branch = "gh-pages"
)
