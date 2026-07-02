using SparseArrays
using ComponentArrays

const HAS_CUDA   = Base.find_package("CUDA")   !== nothing
const HAS_AMDGPU = Base.find_package("AMDGPU") !== nothing

if HAS_CUDA
    import CUDA
    import CUDA.CUSPARSE

    CUDA.allowscalar(false)

    const USE_CUDA   = CUDA.functional()
    const USE_AMDGPU = false
    println("CUDA is functional: $USE_CUDA")

elseif HAS_AMDGPU
    import AMDGPU
    import AMDGPU.rocSPARSE

    AMDGPU.allowscalar(false)

    const USE_CUDA   = false
    const USE_AMDGPU = AMDGPU.functional()
    println("AMDGPU is functional: $USE_AMDGPU")

else
    const USE_CUDA   = false
    const USE_AMDGPU = false
    println("No GPU package found. Running on CPU.")
end

# ── to_device ────────────────────────────────────────────────────────────────
function to_device(arr::AbstractVector{T}) where T
    USE_CUDA   && return CUDA.CuVector{T}(arr)
    USE_AMDGPU && return AMDGPU.ROCVector{T}(arr)
    return arr
end

function to_device(mat::AbstractMatrix{T}) where T
    USE_CUDA   && return CUDA.CUSPARSE.CuSparseMatrixCSC{T}(mat)
    USE_AMDGPU && return AMDGPU.rocSPARSE.ROCSparseMatrixCSC{T}(mat)
    return SparseMatrixCSC{T}(mat)
end

function to_device(ca::ComponentVector)
    return ComponentArray(map(to_device, NamedTuple(ca)))
end