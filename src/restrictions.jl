export restrict, mask!

"""restrict a form to a subset of the points.

- sd is the mesh,
- func is a function that chooses the submesh indices corresponding to the boundary
- form is the vector you want to restrict 
"""
restrict(sd, func::Function, form) = form[func(sd)]

restrict(indices, form) = form[indices]

"""mask a form to values on a subset of the points.

- sd is the mesh,
- form is the vector you want to restrict and 
- values is the vector you want to replace with
- func is a function that chooses the submesh indices corresponding to the boundary
"""
mask!(sd, func::Function, form, values) = setindex!(form, values, func(sd))

mask!(indices, form, values) = setindex!(form, values, indices)
