struct Rasterizer
    fb::GL.Framebuffer
    vao::GL.VertexArray
    program::GL.ShaderProgram

    width::Int64
    height::Int64
end

function Rasterizer(;
    width::Integer, height::Integer,
    differential_bary::Bool = false, enable_z_modify::Bool = false,
)
    vertex_shader_code, geometry_shader_code, fragment_shader_code =
        get_rasterizer_shaders_code(; differential_bary, enable_z_modify)

    vertex_shader = GL.Shader(GL_VERTEX_SHADER, vertex_shader_code)
    geometry_shader = GL.Shader(GL_GEOMETRY_SHADER, geometry_shader_code)
    fragment_shader = GL.Shader(GL_FRAGMENT_SHADER, fragment_shader_code)
    program = GL.ShaderProgram(
        (vertex_shader, geometry_shader, fragment_shader))

    fb = GL.Framebuffer(Dict(
        GL_COLOR_ATTACHMENT0 => GL.TextureArray(
            width, height, 0; type=GL_FLOAT, # TODO fix texture write/load
            internal_format=GL_RGBA32F, data_format=GL_RGBA),
        GL_DEPTH_STENCIL_ATTACHMENT => GL.TextureArray(
            width, height, 0; type=GL_UNSIGNED_INT_24_8,
            internal_format=GL_DEPTH24_STENCIL8,
            data_format=GL_DEPTH_STENCIL),
    ))

    vb_layout = GL.BufferLayout([
        GL.BufferElement(SVector{4, Float32}, "position")])
    vb = GL.VertexBuffer(Float32[], vb_layout; usage=GL_DYNAMIC_DRAW)
    ib = GL.IndexBuffer(UInt32[]; usage=GL_DYNAMIC_DRAW)
    vao = GL.VertexArray(ib, vb)

    Rasterizer(fb, vao, program, width, height)
end

function GL.resize!(
    r::Rasterizer, positions::Array{Float32, 3}, indices::Matrix{UInt32},
)
    depth = size(positions, 3)
    for att in values(r.fb.attachments)
        GL.resize!(att; width=r.width, height=r.height, depth)
    end

    GL.set_data!(r.vao.vertex_buffer, positions)
    GL.set_data!(r.vao.index_buffer, indices)
end

function GL.delete!(r::Rasterizer)
    GL.delete!(r.fb)
    GL.delete!(r.vao)
    GL.delete!(r.program)
end

function (r::Rasterizer)(positions::P; indices::I) where {
    P <: AbstractArray{Float32, 3},
    I <: AbstractMatrix{UInt32},
}
    dev = device_from_type(P)
    positions_host = Array(positions)
    indices_host = Array(indices)

    GL.resize!(r, positions_host, indices_host)
    GL.bind(r.fb)

    GL.enable_depth()
    glDepthFunc(GL_LESS)
    glClearDepth(1.0)

    GL.set_viewport(r.width, r.height)
    GL.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
    GL.set_clear_color(0, 0, 0, 0)

    GL.bind(r.program)
    GL.bind(r.vao)
    GL.draw(r.vao)

    rasterization = GL.get_data(r.fb[GL_COLOR_ATTACHMENT0])
    to_device(dev, rasterization)
end

function ∇(
    r::Rasterizer, Δ::R, rasterization::R, positions::P; indices::I,
) where {
    R <: AbstractArray{Float32, 4},
    P <: AbstractArray{Float32, 3},
    I <: AbstractMatrix{UInt32},
}
    xys = SVector{2, Float32}(2f0 / r.width, 2f0 / r.height)
    xyo = SVector{2, Float32}(1f0 / r.width - 1f0, 1f0 / r.height - 1f0)

    dev = device_from_type(P)
    _, width, height, depth = size(rasterization)

    ∇positions = zeros(dev, Float32, size(positions))
    wait(∇rasterize_kernel!(dev)(
        ∇positions, Δ, rasterization, positions, indices, xys, xyo;
        ndrange=(width, height, depth)))
    ∇positions
end

function rrule(r::Rasterizer, positions::P; indices::I) where {
    P <: AbstractArray{Float32, 3},
    I <: AbstractMatrix{UInt32},
}
    rasterization = r(positions; indices)
    function rasterization_pullback(Δ)
        Tangent{Rasterizer}(), ∇(r, Δ, rasterization, positions; indices)
    end
    rasterization, rasterization_pullback
end

@kernel function ∇rasterize_kernel!(
    ∇positions, @Const(Δ), @Const(rasterization), @Const(positions),
    @Const(indices), xys::SVector{2, Float32}, xyo::SVector{2, Float32},
)
    px, py, pz = @index(Global, NTuple)
    @uniform n_vertices = size(positions, 2)
    @uniform n_triangles = size(indices, 2)

    dy = to_vec2f0(@view(Δ[:, px, py, pz]))
    triangle_idx::UInt32 = rasterization[4, px, py, pz]
    valid_triangle = 1 ≤ triangle_idx ≤ n_triangles

    if valid_triangle
        # Check if gradients are ±0.
        is_grad_zero = (
            (reinterpret(UInt32, dy[1]) | reinterpret(UInt32, dy[2])) << 1) == 0
        if !is_grad_zero
            v_idx = to_vec3u32(@view(indices[:, triangle_idx])) .+ 1u32
            valid_indices = all(1 .≤ v_idx .≤ n_vertices)
            if valid_indices
                p1 = to_vec4f0(@view(positions[:, v_idx[1], pz]))
                p2 = to_vec4f0(@view(positions[:, v_idx[2], pz]))
                p3 = to_vec4f0(@view(positions[:, v_idx[3], pz]))

                # Evaluate edge function.
                fxy = xys .* SVector{2, Float32}(px - 1, py - 1) .+ xyo

                p1x = p1[1] - fxy[1] * p1[4]
                p1y = p1[2] - fxy[2] * p1[4]
                p2x = p2[1] - fxy[1] * p2[4]
                p2y = p2[2] - fxy[2] * p2[4]
                p3x = p3[1] - fxy[1] * p3[4]
                p3y = p3[2] - fxy[2] * p3[4]

                a1 = p2x * p3y - p2y * p3x
                a2 = p3x * p1y - p3y * p1x
                a3 = p1x * p2y - p1y * p2x

                # Compute inverse area.
                at = a1 + a2 + a3
                inverse_area = 1f0 / (at + copysign(1f-6, at))
                # Perspective correction. Normalized barycentrics.
                b1 = a1 * inverse_area
                b2 = a2 * inverse_area
                # Position gradients.
                gb = dy .* inverse_area
                gbb = gb[1] * b1 + gb[2] * b2

                gp1x = gbb * (p3y - p2y) - gb[2] * p3y
                gp2x = gbb * (p1y - p3y) + gb[1] * p3y
                gp3x = gbb * (p2y - p1y) - gb[1] * p2y + gb[2] * p1y

                gp1y = gbb * (p2x - p3x) + gb[2] * p3x
                gp2y = gbb * (p3x - p1x) - gb[1] * p3x
                gp3y = gbb * (p1x - p2x) + gb[1] * p2x - gb[2] * p1x

                gp1w = -fxy[1] * gp1x - fxy[2] * gp1y
                gp2w = -fxy[1] * gp2x - fxy[2] * gp2y
                gp3w = -fxy[1] * gp3x - fxy[2] * gp3y

                @atomic ∇positions[1, v_idx[1], pz] += gp1x
                @atomic ∇positions[2, v_idx[1], pz] += gp1y
                @atomic ∇positions[4, v_idx[1], pz] += gp1w

                @atomic ∇positions[1, v_idx[2], pz] += gp2x
                @atomic ∇positions[2, v_idx[2], pz] += gp2y
                @atomic ∇positions[4, v_idx[2], pz] += gp2w

                @atomic ∇positions[1, v_idx[3], pz] += gp3x
                @atomic ∇positions[2, v_idx[3], pz] += gp3y
                @atomic ∇positions[4, v_idx[3], pz] += gp3w
            end
        end
    end
end
