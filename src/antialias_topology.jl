function construct_topology_hash(
    indices::I; hash_elements_per_triangle::Int64 = 8,
) where I <: AbstractMatrix{UInt32}
    ndims(indices) == 2 || error(
        "`indices` has `$(ndims(indices))` dimensions, but must be `2`.")
    ((size(indices, 1) == 3) && (size(indices, 2) > 0)) || error(
        "`indices` has `$(size(indices))` size, but must be of `[3, > 0]`.")
    (hash_elements_per_triangle % 2 == 0) || error(
        "`hash_elements_per_triangle` is `$hash_elements_per_triangle`, " *
        "but must be a power of 2.")

    # Number of triangles accomodated by hash. Must be power of two.
    n_alloc_triangles = 64
    n_triangles = size(indices, 2)
    while n_alloc_triangles < n_triangles
        n_alloc_triangles <<= 1
    end

    dev = device_from_type(I)
    ev_hash = zeros(dev, UInt32,
        (4, hash_elements_per_triangle * n_alloc_triangles))
    wait(antialias_mesh_kernel!(dev)(
        ev_hash, indices; ndrange=n_triangles))
    ev_hash
end

@kernel function antialias_mesh_kernel!(
    ev_hash::E, @Const(indices),
) where E <: AbstractMatrix{UInt32}
    i = @index(Global)
    # Must start from 0.
    v1, v2, v3 = indices[1, i], indices[2, i], indices[3, i]
    same_idx = (v1 == v2) || (v2 == v3) || (v1 == v3)
    if !same_idx
        evhash_insert_vertex!(ev_hash, v2, v3, v1)
        evhash_insert_vertex!(ev_hash, v3, v1, v2)
        evhash_insert_vertex!(ev_hash, v1, v2, v3)
    end
end
