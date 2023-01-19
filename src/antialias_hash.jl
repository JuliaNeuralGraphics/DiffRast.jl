struct HashIndex
    idx::UInt32
    skip::UInt32
    mask::UInt32
end

function HashIndex(hash_size::UInt32, key::UInt64)
    mask = hash_size - 1u32
    idx = (key & 0xffffffff) % UInt32
    skip = (key >> 32) % UInt32
    idx, skip, _ = jenkins_mix(idx, skip, JENKINS_MAGIC)
    idx &= mask
    skip &= mask
    skip |= 1u32
    HashIndex(idx, skip, mask)
end

function next(h::HashIndex)
    HashIndex((h.idx + h.skip) & h.mask, h.skip, h.mask)
end

get(h::HashIndex) = h.idx + 1u32

function jenkins_mix(a::UInt32, b::UInt32, c::UInt32)
    a -= b; a -= c; a ⊻= (c >> 13);
    b -= c; b -= a; b ⊻= (a << 8);
    c -= a; c -= b; c ⊻= (b >> 13);
    a -= b; a -= c; a ⊻= (c >> 12);
    b -= c; b -= a; b ⊻= (a << 16);
    c -= a; c -= b; c ⊻= (b >> 5);
    a -= b; a -= c; a ⊻= (c >> 3);
    b -= c; b -= a; b ⊻= (a << 10);
    c -= a; c -= b; c ⊻= (b >> 15);
    a, b, c
end

function hash_insert!(ev_hash, key::UInt64, value::UInt32)
    idx = HashIndex(UInt32(size(ev_hash, 2)), key)
    while true
        # Reinterpret UInt32 array to UInt64.
        # `key` will then take two positions in the `ev_hash`.
        uev_hash = reinterpret(UInt64, ev_hash)
        prev, _ = @atomicreplace uev_hash[1, get(idx)] 0 => key
        ((prev == 0) || (prev == key)) && break
        idx = next(idx)
    end

    # Write value to either `3` or `4` slots.
    prev, _ = @atomicreplace ev_hash[3, get(idx)] 0 => value
    if (prev != 0) && (prev != value) # If `3` is occupied already.
        @atomicreplace ev_hash[4, get(idx)] 0 => value
    end
    nothing
end

function hash_find(ev_hash, key::UInt64)
    idx = HashIndex(UInt32(size(ev_hash, 2)), key)
    while true
        entry = to_vec4u32(@view(ev_hash[:, get(idx)]))
        k = UInt64(entry[1]) | (UInt64(entry[2]) << 32)
        (k == key || k == 0) && return (entry[3], entry[4])
        idx = next(idx)
    end
end

function evhash_insert_vertex!(ev_hash, va::UInt32, vb::UInt32, vn::UInt32)
    if va != vb
        v0::UInt64 = min(va, vb) + 1u32
        v1::UInt64 = max(va, vb) + 1u32
        key = v0 | (v1 << 32)
        hash_insert!(ev_hash, key, vn + 1u32)
    end
end

function evhash_find_vertex(ev_hash, va::UInt32, vb::UInt32, vr::UInt32)::Int64
    va == vb && return -1

    v0::UInt64 = min(va, vb) + 1u32
    v1::UInt64 = max(va, vb) + 1u32
    key = v0 | (v1 << 32) # Hash key.

    # TODO can hash find return 0?
    vn = Int64.(hash_find(ev_hash, key)) .- 1u32
    vn[1] == vr && return vn[2]
    vn[2] == vr && return vn[1]
    -1
end
