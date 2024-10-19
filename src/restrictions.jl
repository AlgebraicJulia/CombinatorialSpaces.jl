using CombinatorialSpaces, CombinatorialSpaces.SimplicialSets

export restrict, mask!

"""    restrict(sd::HasDeltaSet,  

restrict a form to a subset of the points.

`sd`: the mesh,
`func`: a function that chooses the submesh indices corresponding to the boundary
`form`: the vector you want to restrict 
"""
restrict(sd::HasDeltaSet, func::Function, form) = form[func(sd)]

restrict(indices, form) = form[indices]

"""    mask(sd::HasDeltaSet, func::Function, form, values)

Masks a form to values on a subset of the points.

# Arguments:
`sd`: the mesh
`func`: function that chooses the submesh indices corresponding to the boundary
`form`: the vector you want to restrict and 
`values:` is the vector you want to replace with
"""
mask!(sd::HasDeltaSet, func::Function, form, values) = setindex!(form, values, func(sd))

mask!(indices, form, values) = setindex!(form, values, indices)
