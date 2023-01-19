function interpolate(colors::C, rasterization::R; indices::I) where {
    C <: AbstractArray{Float32, 3},
    R <: AbstractArray{Float32, 4},
    I <: AbstractMatrix{UInt32},
}
    _, width, height, depth = size(rasterization)
    n_attributes = size(colors, 1)

    dev = device_from_type(C)
    out = zeros(dev, Float32, (n_attributes, width, height, depth))
    wait(interpolate_kernel!(dev)(
        out, colors, rasterization, indices, Val{n_attributes}();
        ndrange=(width, height, depth)))
    out
end

@kernel function interpolate_kernel!(
    out, @Const(colors), @Const(rasterization), @Const(indices), ::Val{C},
) where C
    px, py, pz = @index(Global, NTuple)
    @uniform n_vertices = size(colors, 2)
    @uniform n_triangles = size(indices, 2)

    r = to_vec4f0(@view(rasterization[:, px, py, pz]))
    triangle_idx::Int32 = Int32(r[4])

    valid_triangle = 1 ≤ triangle_idx ≤ n_triangles
    if valid_triangle
        v_idx = to_vec3u32(@view(indices[:, triangle_idx])) .+ one(UInt32)
        valid_indices = all(1 .≤ v_idx .≤ n_vertices)
        # Don't do anything in case of corrupted indices.
        if valid_indices
            # Barycentric coordiantes.
            b = SVector{3, Float32}(r[1], r[2], 1f0 - r[1] - r[2])
            # Interpolate and write attributes.
            for i in 1:C
                out[i, px, py, pz] =
                    b[1] * colors[i, v_idx[1], pz] +
                    b[2] * colors[i, v_idx[2], pz] +
                    b[3] * colors[i, v_idx[3], pz]
            end
        end
    end
end

function ∇interpolate(Δ::R, colors::C, rasterization::R; indices::I) where {
    C <: AbstractArray{Float32, 3},
    R <: AbstractArray{Float32, 4},
    I <: AbstractMatrix{UInt32},
}
    dev = device_from_type(C)
    n_attributes = size(colors, 1)

    ∇rasterization = zeros(dev, Float32, size(rasterization))
    ∇attributes = zeros(dev, Float32, size(colors))

    _, width, height, depth = size(rasterization)
    wait(∇interpolate_kernel!(dev)(
        ∇rasterization, ∇attributes, Δ,
        colors, rasterization, indices, Val{n_attributes}();
        ndrange=(width, height, depth)))

    # println(">> ∇interpolation")
    # display(∇rasterization); println()
    # display(∇attributes); println()

    ∇rasterization, ∇attributes
end

function rrule(
    ::typeof(interpolate), colors::C, rasterization::R; indices::I,
) where {
    C <: AbstractArray{Float32, 3},
    R <: AbstractArray{Float32, 4},
    I <: AbstractMatrix{UInt32},
}
    out = interpolate(colors, rasterization; indices)
    function interpolate_pullback(Δ)
        ∇rasterization, ∇attributes = ∇interpolate(
            Δ, colors, rasterization; indices)
        NoTangent(), ∇attributes, ∇rasterization
    end
    out, interpolate_pullback
end

@kernel function ∇interpolate_kernel!(
    ∇rasterization, ∇attributes, @Const(Δ),
    @Const(colors), @Const(rasterization), @Const(indices), ::Val{C},
) where C
    px, py, pz = @index(Global, NTuple)
    @uniform n_vertices = size(colors, 2)
    @uniform n_triangles = size(indices, 2)

    r = to_vec4f0(@view(rasterization[:, px, py, pz]))
    triangle_idx::UInt32 = r[4]
    valid_triangle = 1 ≤ triangle_idx ≤ n_triangles

    if valid_triangle
        v_idx = to_vec3u32(@view(indices[:, triangle_idx])) .+ 1u32
        valid_indices = all(1 .≤ v_idx .≤ n_vertices)
        if valid_indices
            # Barycentric coordiantes.
            b = SVector{3, Float32}(r[1], r[2], 1f0 - r[1] - r[2])
            ∇b = zeros(MVector{2, Float32})
            # Loop over attributes & accumulate attribute gradients.
            for i in 1:C
                dy = Δ[i, px, py, pz]
                c1 = colors[i, v_idx[1], pz]
                c2 = colors[i, v_idx[2], pz]
                c3 = colors[i, v_idx[3], pz]
                ∇b[1] += dy * (c1 - c3)
                ∇b[2] += dy * (c2 - c3)
                @atomic ∇attributes[i, v_idx[1], pz] += dy * b[1]
                @atomic ∇attributes[i, v_idx[2], pz] += dy * b[2]
                @atomic ∇attributes[i, v_idx[3], pz] += dy * b[3]
            end

            ∇rasterization[1, px, py, pz] = ∇b[1]
            ∇rasterization[2, px, py, pz] = ∇b[2]
        end
    end
end
