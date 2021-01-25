using Documenter
using CombinatorialSpaces

makedocs(
    sitename = "CombinatorialSpaces",
    format = Documenter.HTML(),
    modules = [CombinatorialSpaces]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
