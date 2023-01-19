module DiffRast

import ChainRulesCore: rrule

using Atomix: @atomic, @atomicreplace
using CImGui
using CImGui.ImGuiGLFWBackend.LibGLFW
using ChainRulesCore
using GL
using KAUtils
using KernelAbstractions
using ModernGL
using NNlib
using Preferences
using StaticArrays

const BACKEND = KAUtils.BACKEND
const DEVICE = KAUtils.DEVICE

@static if BACKEND == "CUDA"
    using NNlibCUDA
    using CUDA
    using CUDAKernels
elseif BACKEND == "ROC"
    using AMDGPU
    using ROCKernels
end

# Hack to add UInt32 literal, e.g.: 2u32 ≡ UInt32(2).
struct U32 end
Base.:*(n::Number, ::U32) = UInt32(n)
const u32::U32 = U32()

# Convenience functions.
to_vec2f0(x) = SVector{2, Float32}(x[1], x[2])

to_vec4f0(x) = SVector{4, Float32}(x[1], x[2], x[3], x[4])

to_vec3u32(x) = SVector{3, UInt32}(x[1], x[2], x[3])

to_vec4u32(x) = SVector{4, UInt32}(x[1], x[2], x[3], x[4])

same_sign(v1, v2) = sign(v1) == sign(v2)

GL.init(4, 4)

include("shaders.jl")
include("rasterization.jl")
include("interpolation.jl")

include("antialias_hash.jl")
include("antialias_topology.jl")
include("antialias.jl")

include("utils.jl")

end
