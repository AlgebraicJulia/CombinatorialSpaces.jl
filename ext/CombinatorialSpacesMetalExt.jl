module CombinatorialSpacesMetalExt

using CombinatorialSpaces
using Metal
import CombinatorialSpaces: cache_wedge

# Wedge Product
#--------------

# Cast all but the last argument to MtlArray.
# Metal does not support Float64, and Float16 is not likely useful. Use Float32.
function cache_wedge(::Type{Tuple{m,n}}, sd, ::Type{Val{:Metal}}) where {m,n}
  wc = cache_wedge(Tuple{m,n}, sd, Val{:CPU})
  if wc[2] isa Matrix
    (MtlArray(wc[1]), MtlArray(Matrix{Float32}(wc[2])), wc[3])
  else
    (MtlArray.(wc[1:end-1])..., wc[end])
  end
end

end

