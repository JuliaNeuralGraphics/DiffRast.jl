const AA_EDGE_MASK::UInt32 = 3
const AA_FLAG_DOWN_BIT::UInt32 = 2
const AA_FLAG_TRI_BIT::UInt32 = 3
const JENKINS_MAGIC::UInt32 = 0x9e3779b9

function antialias(
    interpolations::C, positions::P; rasterization::C, indices::E,
) where {
    C <: AbstractArray{Float32, 4},
    P <: AbstractArray{Float32, 3},
    E <: AbstractMatrix{UInt32},
}
    dev = device_from_type(C)
    _, width, height, depth = size(interpolations)

    work_counter = zeros(dev, UInt32, 2)
    # `* 2` for two work items per pixel: bottom and right neighbor.
    work_buffer = zeros(dev, SVector{4, UInt32}, width * height * depth * 2)

    topology_hash = construct_topology_hash(indices)

    ndrange = (width, height, depth)
    wait(antialias_discontinuity_kernel!(dev)(
        work_buffer, work_counter, rasterization; ndrange))

    output = copy(interpolations)
    n_linear_threads = 512
    CH = Val{size(output, 1)}()
    wait(antialias_analysis_kernel!(dev, n_linear_threads)(
        output, interpolations, work_buffer, work_counter, positions,
        rasterization, indices, topology_hash, UInt32(n_linear_threads), CH;
        ndrange=length(work_buffer)))

    output, work_buffer, work_counter
end

"""
Each work item consists of:
- px, py: pixel coordinate.
- pz_flags: high 16 bits are pz, low 16 bit - edge index and flags.
- α: Antialiasing α value. Zero if no AA.
"""
@kernel function antialias_discontinuity_kernel!(
    work_buffer::B, work_counter::C, @Const(rasterization),
) where {
    B <: AbstractVector{SVector{4, UInt32}},
    C <: AbstractVector{UInt32},
}
    px::UInt32, py::UInt32, pz::UInt32 = @index(Global, NTuple)
    @uniform width::UInt32 = size(rasterization, 2)
    @uniform height::UInt32 = size(rasterization, 3)

    tidx_1::UInt32 = rasterization[4, px, py, pz]
    tidx_2::UInt32 = rasterization[4, min(width, px + 1u32), py, pz]
    tidx_3::UInt32 = rasterization[4, px, min(height, py + 1u32), pz]

    count::UInt32 = 0u32
    tidx_1 != tidx_2 && (count  = 1u32;)
    tidx_1 != tidx_3 && (count += 1u32;)

    if count > 0
        s_temp = @localmem UInt32 1
        s_temp[1] = 0u32
        @synchronize()
        # Accumulate counters accross all threads.
        idx, _ = @atomic s_temp[1] + count
        @synchronize()
        # If this is the first time
        # - add s_temp to work counter;
        # - add whatever was in the work counter to `idx`.
        if idx == 0
            base, _ = @atomic work_counter[1] + s_temp[1]
            s_temp[1] = base
        end
        @synchronize()
        idx += s_temp[1]

        wbz = (pz - 1u32) << 16
        if tidx_1 != tidx_2
            idx += 1u32
            work_buffer[idx] = SVector{4, UInt32}(px, py, wbz, 0u32)
        end
        if tidx_1 != tidx_3
            idx += 1u32
            wbz = wbz + (1u32 << AA_FLAG_DOWN_BIT)
            work_buffer[idx] = SVector{4, UInt32}(px, py, wbz, 0u32)
        end
    end
end

function unpack_work_item(work_item::SVector{4, UInt32})
    px = work_item[1]
    py = work_item[2]
    pz = (work_item[3] >> 16) + 1u32
    down = (work_item[3] >> AA_FLAG_DOWN_BIT) & 1u32
    px, py, pz, down
end

function unpack_work_item_extra(work_item::SVector{4, UInt32})
    # Unpack α and flags.
    α = reinterpret(Float32, work_item[4])
    # Get the sign of δs from forward pass.
    δs_sign = (work_item[3] >> AA_FLAG_TRI_BIT) & 1u32
    # Reconstruct δs from forward pass by adding back the sign.
    # δs = reinterpret(Float32, reinterpret(UInt32, 1f0) | (δs_sign << 31))
    # Get delta index from max_idx3.
    di = work_item[3] & AA_EDGE_MASK
    α, δs_sign, di
end

@kernel function antialias_analysis_kernel!(
    output::O, interpolations::O, work_buffer::B, work_counter::C, positions,
    rasterization, indices::I, topology_hash, n_linear_threads::UInt32,
    ::Val{CH},
) where {
    O <: AbstractArray{Float32, 4},
    B <: AbstractVector{SVector{4, UInt32}},
    C <: AbstractVector{UInt32},
    I <: AbstractMatrix{UInt32},
    CH,
}
    i::UInt32 = @index(Local) # Need local index for `s_base` below.
    @uniform work_count::UInt32 = work_counter[1]
    @uniform n_vertices::UInt32 = size(positions, 2)
    @uniform n_triangles::UInt32 = size(indices, 2)
    @uniform half_res::SVector{2, Float32} = SVector{2, Float32}(
        size(rasterization, 2), size(rasterization, 3)) .* 0.5f0

    s_base = @localmem UInt32 1
    while true
        @synchronize
        if i == 1
            s_base[1], _ = @atomic work_counter[2] + n_linear_threads
        end
        @synchronize

        idx = s_base[1] + i # This is the Global index.
        idx > work_count && break

        work_item = work_buffer[idx]
        px, py, pz, down = unpack_work_item(work_item)
        is_down = down == 1u32

        pxx, pyy = is_down ? (px, py + 1u32) : (px + 1u32, py)
        tdepth_1       = rasterization[3, px, py, pz]
        tidx_1::UInt32 = rasterization[4, px, py, pz]
        tdepth_2       = rasterization[3, pxx, pyy, pz]
        tidx_2::UInt32 = rasterization[4, pxx, pyy, pz]

        # Select triangle idx based on background & depth.
        tidx = (tidx_1 ≥ 1) ? tidx_1 : tidx_2
        if (tidx_1 ≥ 1) && (tidx_2 ≥ 1)
            tidx = (tdepth_1 < tdepth_2) ? tidx_1 : tidx_2
        end

        fx, fy = px, py
        if tidx == tidx_2
            # Calculate w.r.t. neighboring pixel.
            fx += 1u32 - down # either +0 or +1
            fy += down
        end
        fxy = SVector{2, Float32}(fx - 1u32, fy - 1u32) .- half_res .+ 0.5f0

        valid_triangle = 1 ≤ tidx ≤ n_triangles
        valid_triangle || continue

        v_idx = to_vec3u32(@view(indices[:, tidx]))
        valid_indices = all(0 .≤ v_idx .< n_vertices)
        valid_indices || continue

        # Fetch opposite vertex indices.
        # Use vertex itself (always a silhouette) if no opposite vertex exists.
        # Vertex ids are 0 based here.
        op1 = evhash_find_vertex(topology_hash, v_idx[3], v_idx[2], v_idx[1])
        op2 = evhash_find_vertex(topology_hash, v_idx[1], v_idx[3], v_idx[2])
        op3 = evhash_find_vertex(topology_hash, v_idx[2], v_idx[1], v_idx[3])

        # Make vertices 1-based & fetch them.
        v_idx = v_idx .+ 1u32
        p1 = to_vec4f0(@view(positions[:, v_idx[1], pz]))
        p2 = to_vec4f0(@view(positions[:, v_idx[2], pz]))
        p3 = to_vec4f0(@view(positions[:, v_idx[3], pz]))
        o1 = op1 < 0 ? p1 : (to_vec4f0(@view(positions[:, op1 + 1, pz])))
        o2 = op2 < 0 ? p2 : (to_vec4f0(@view(positions[:, op2 + 1, pz])))
        o3 = op3 < 0 ? p3 : (to_vec4f0(@view(positions[:, op3 + 1, pz])))

        # Project vertices to pixel space.
        pxy1 = project_to_pixel(p1, fxy, half_res)
        pxy2 = project_to_pixel(p2, fxy, half_res)
        pxy3 = project_to_pixel(p3, fxy, half_res)
        oxy1 = project_to_pixel(o1, fxy, half_res)
        oxy2 = project_to_pixel(o2, fxy, half_res)
        oxy3 = project_to_pixel(o3, fxy, half_res)

        # If no matching sings - we're done.
        bb, a1, a2, a3 = compute_signs(pxy1, pxy2, pxy3, oxy1, oxy2, oxy3)
        if same_sign(a1, bb) || same_sign(a2, bb) || same_sign(a3, bb)
            if is_down # XY flip for horizontal edges.
                pxy1 = reverse(pxy1)
                pxy2 = reverse(pxy2)
                pxy3 = reverse(pxy3)
            end

            δxy1 = pxy3 .- pxy2
            δxy2 = pxy1 .- pxy3
            δxy3 = pxy2 .- pxy1

            # Check if an edge crosses between us and the neighbor pixel.
            δs = (tidx == tidx_1) ? 1f0 : -1f0
            δ1 = δs * (pxy2[1] * δxy1[2] - pxy2[2] * δxy1[1])
            δ2 = δs * (pxy3[1] * δxy2[2] - pxy3[2] * δxy2[1])
            δ3 = δs * (pxy1[1] * δxy3[2] - pxy1[2] * δxy3[1])

            if same_sign(pxy2[2], pxy3[2])
                δ1 = -maxintfloat(Float32)
                δxy1 = SVector{2, Float32}(δxy1[1], 1f0)
            end
            if same_sign(pxy3[2], pxy1[2])
                δ2 = -maxintfloat(Float32)
                δxy2 = SVector{2, Float32}(δxy2[1], 1f0)
            end
            if same_sign(pxy1[2], pxy2[2])
                δ3 = -maxintfloat(Float32)
                δxy3 = SVector{2, Float32}(δxy3[1], 1f0)
            end

            δc = -maxintfloat(Float32)
            di = max_idx3(δ1, δ2, δ3, δxy1[2], δxy2[2], δxy3[2])
            if (di == 0) && same_sign(a1, bb) && (abs(δxy1[2]) > abs(δxy1[1]))
                δc = δ1 / δxy1[2]
            end
            if (di == 1) && same_sign(a2, bb) && (abs(δxy2[2]) > abs(δxy2[1]))
                δc = δ2 / δxy2[2]
            end
            if (di == 2) && same_sign(a3, bb) && (abs(δxy3[2]) > abs(δxy3[1]))
                δc = δ3 / δxy3[2]
            end

            # Adjust output image if a suitable edge was found.
            ϵ = 0.0625f0 # Expect no more than 1/16 pixel inaccuracy.
            if -ϵ < δc < (1f0 + ϵ)
                δc = clamp(δc, 0f0, 1f0)
                α = δs * (0.5f0 - δc)
                out_px, out_py = (α > 0) ? (px, py) : (pxx, pyy)
                for j in 1:CH
                    antialiased_color = α * (
                        interpolations[j, pxx, pyy, pz] -
                        interpolations[j, px, py, pz])
                    @atomic output[j, out_px, out_py, pz] += antialiased_color
                end

                # Update work buffer for the gradient calculation.
                flags::UInt32 = (pz - 1u32) << 16
                flags |= di
                flags |= down << AA_FLAG_DOWN_BIT
                # Store the sign of δs. If it is -1, we store 1, otherwise 0.
                flags |= (reinterpret(UInt32, δs) >> 31) << AA_FLAG_TRI_BIT
                work_buffer[idx] = SVector{4, UInt32}(
                    work_item[1], work_item[2], flags, reinterpret(UInt32, α))
            end
        end
    end
end

function ∇antialias(
    Δ::O, interpolations::O, positions::P; rasterization::O,
    indices::I, work_buffer::B, work_counter::C,
) where {
    O <: AbstractArray{Float32, 4},
    B <: AbstractVector{SVector{4, UInt32}},
    C <: AbstractVector{UInt32},
    P <: AbstractArray{Float32, 3},
    I <: AbstractMatrix{UInt32},
}
    @assert size(Δ) == size(interpolations)
    @assert size(rasterization)[2:end] == size(interpolations)[2:end]
    @assert ndims(positions) == 3 && size(positions, 1) == 4
    @assert ndims(indices) == 2 && size(indices, 1) == 3

    dev = device_from_type(B)
    ∇interpolations = copy(Δ) # Using Δ as a base.
    ∇positions = zeros(dev, Float32, size(positions))

    # Clear 2nd slot in counter.
    work_counter = to_device(dev, UInt32[Array(work_counter)[1], 0u32])

    n_linear_threads = 512
    CH = Val{size(interpolations, 1)}()
    wait(∇antialias_kernel!(dev, n_linear_threads)(
        ∇interpolations, ∇positions, Δ, work_buffer, work_counter,
        interpolations, positions, rasterization, indices,
        UInt32(n_linear_threads), CH; ndrange=length(work_buffer)))

    ∇interpolations, ∇positions
end

@kernel function ∇antialias_kernel!(
    ∇interpolations, ∇positions, Δ, work_buffer::B, work_counter::C,
    interpolations, positions::P, rasterization::R, indices::I,
    n_linear_threads::UInt32, ::Val{CH},
) where {
    B <: AbstractVector{SVector{4, UInt32}},
    P <: AbstractArray{Float32, 3},
    C <: AbstractVector{UInt32},
    R <: AbstractArray{Float32, 4},
    I <: AbstractMatrix{UInt32},
    CH,
}
    i::UInt32 = @index(Local)
    @uniform work_count::UInt32 = work_counter[1]
    @uniform n_vertices::UInt32 = size(positions, 2)
    @uniform n_triangles::UInt32 = size(indices, 2)
    @uniform half_res::SVector{2, Float32} = SVector{2, Float32}(
        size(rasterization, 2), size(rasterization, 3)) .* 0.5f0

    s_base = @localmem UInt32 1
    while true
        @synchronize
        if i == 1
            s_base[1], _ = @atomic work_counter[2] + n_linear_threads
        end
        @synchronize

        idx = s_base[1] + i
        (idx > work_count) && break

        # Read work item filled by forward pass.
        work_item = work_buffer[idx]
        # If α is 0, then no antialiasing occurred, skip.
        (work_item[4] == 0) && continue

        px, py, pz, down = unpack_work_item(work_item)
        α, δs_sign, di = unpack_work_item_extra(work_item)

        is_down = down == 1u32
        is_α_positive = α > 0
        is_δs_negative = δs_sign == 1

        pxx, pyy = is_down ? (px, py + 1u32) : (px + 1u32, py)
        tidx::UInt32 = rasterization[4,
            (is_δs_negative ? pxx : px),
            (is_δs_negative ? pyy : py), pz]

        fx, fy = px, py
        if is_δs_negative
            # Calculate w.r.t. neighboring pixel.
            fx += 1u32 - down # either +0 or +1
            fy += down
        end
        fxy = SVector{2, Float32}(fx - 1u32, fy - 1u32) .- half_res .+ 0.5f0

        valid_triangle = 1 ≤ tidx ≤ n_triangles
        valid_triangle || continue

        position_ω = 0f0

        # Loop over channels and accumulate gradients.
        for j in 1:CH
            δy = Δ[j, (is_α_positive ? px : pxx), (is_α_positive ? py : pyy), pz]
            if !iszero(δy)
                # Update position gradient weight.
                position_ω += δy * (
                    interpolations[j, pxx, pyy, pz] -
                    interpolations[j, px, py, pz])
                # Update color gradients.
                v = α * δy
                @atomic ∇interpolations[j, px, py, pz] += -v
                @atomic ∇interpolations[j, pxx, pyy, pz] += v
            end
        end

        iszero(position_ω) && continue

        # Fetch vertex indices of the active edge and their positions.
        i1 = (di < 2) ? (di + 1u32) : 0u32
        i2 = (i1 < 2) ? (i1 + 1u32) : 0u32
        v_idx1 = indices[(i1 + 1u32), tidx] + 1u32
        v_idx2 = indices[(i2 + 1u32), tidx] + 1u32

        (
            (v_idx1 < 1) || (v_idx1 > n_vertices) ||
            (v_idx2 < 1) || (v_idx2 > n_vertices)) &&
            continue

        p1 = to_vec4f0(@view(positions[:, v_idx1, pz]))
        p2 = to_vec4f0(@view(positions[:, v_idx2, pz]))

        # Project vertices to pixel space.
        l_half_res = copy(half_res)
        if is_down # XY flip for horizontal edges.
            p1 = SVector{4, Float32}(p1[2], p1[1], p1[3], p1[4])
            p2 = SVector{4, Float32}(p2[2], p2[1], p2[3], p2[4])
            fxy = reverse(fxy)
            l_half_res = reverse(l_half_res)
        end

        pxy1 = project_to_pixel(p1, fxy, l_half_res)
        pxy2 = project_to_pixel(p2, fxy, l_half_res)

        # Gradient calculation.
        δx = pxy2[1] - pxy1[1]
        δy = pxy2[2] - pxy1[2]
        δb = pxy1[1] * δy - pxy1[2] * δx

        iδy = 1f0 / (δy + copysign(1f-3, δy))
        # Calculate position gradients.
        δby = δb * iδy
        w1 = 1f0 / p1[4]
        w2 = 1f0 / p2[4]
        iw1 = -w1 * iδy * position_ω
        iw2 =  w2 * iδy * position_ω

        gp1x = iw1 * l_half_res[1] * pxy2[2]
        gp2x = iw2 * l_half_res[1] * pxy1[2]
        gp1y = iw1 * l_half_res[2] * (δby - pxy2[1])
        gp2y = iw2 * l_half_res[2] * (δby - pxy1[1])
        gp1w = -(p1[1] * gp1x + p1[2] * gp1y) * w1
        gp2w = -(p2[1] * gp2x + p2[2] * gp2y) * w2

        if is_down
            gp1x, gp1y = gp1y, gp1x
            gp2x, gp2y = gp2y, gp2x
        end

        # Kill position gradients if α was saturated.
        if abs(α) ≥ 0.5f0
            gp1x = gp1y = gp1w = 0f0
            gp2x = gp2y = gp2w = 0f0
        end

        @atomic ∇positions[1, v_idx1, pz] += gp1x
        @atomic ∇positions[2, v_idx1, pz] += gp1y
        @atomic ∇positions[4, v_idx1, pz] += gp1w

        @atomic ∇positions[1, v_idx2, pz] += gp2x
        @atomic ∇positions[2, v_idx2, pz] += gp2y
        @atomic ∇positions[4, v_idx2, pz] += gp2w
    end
end

function compute_signs(
    pxy1::SVector{2, Float32}, pxy2::SVector{2, Float32}, pxy3::SVector{2, Float32},
    oxy1::SVector{2, Float32}, oxy2::SVector{2, Float32}, oxy3::SVector{2, Float32},
)
    # Signs to kill non-silhouette edges.
    bb = # Triangle itself.
        (pxy2[1] - pxy1[1]) * (pxy3[2] - pxy1[2]) - # 2\/3
        (pxy3[1] - pxy1[1]) * (pxy2[2] - pxy1[2])   # 3/\2
    a1 = # Wings.
        (pxy2[1] - oxy1[1]) * (pxy3[2] - oxy1[2]) - # 2\/3
        (pxy3[1] - oxy1[1]) * (pxy2[2] - oxy1[2])   # 3/\2
    a2 = # Wings.
        (pxy3[1] - oxy2[1]) * (pxy1[2] - oxy2[2]) - # 3\/1
        (pxy1[1] - oxy2[1]) * (pxy3[2] - oxy2[2])   # 1/\3
    a3 = # Wings.
        (pxy1[1] - oxy3[1]) * (pxy2[2] - oxy3[2]) - # 1\/2
        (pxy2[1] - oxy3[1]) * (pxy1[2] - oxy3[2])   # 2/\1
    bb, a1, a2, a3
end

function project_to_pixel(
    vertex::SVector{4, Float32}, fxy::SVector{2, Float32},
    half_res::SVector{2, Float32},
)
    w = 1f0 / vertex[4]
    to_vec2f0(vertex) .* w .* half_res .- fxy
end

function max_idx3(n1, n2, n3, d1, d2, d3)
    g31 = rational_gt(n3, n1, d3, d1)
    g32 = rational_gt(n3, n2, d3, d2)
    g31 && g32 && return 2u32
    rational_gt(n2, n1, d2, d1) && return 1u32
    return 0u32
end

function rational_gt(n1, n2, d1, d2)
    ((n1 * d2) > (n2 * d1)) == same_sign(d1, d2)
end

function rrule(
    ::typeof(antialias), interpolations::C, positions::P;
    rasterization::C, indices::E,
) where {
    C <: AbstractArray{Float32, 4},
    P <: AbstractArray{Float32, 3},
    E <: AbstractMatrix{UInt32},
}
    antialiased, work_buffer, work_counter = antialias(
        interpolations, positions; rasterization, indices)
    function antialias_pullback(Δ)
        ∇interpolations, ∇positions = ∇antialias(
            Δ, interpolations, positions; rasterization,
            indices, work_buffer, work_counter)
        NoTangent(), ∇interpolations, ∇positions
    end
    antialiased, antialias_pullback
end
