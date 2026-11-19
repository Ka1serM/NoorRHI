public struct AccelerationStructureHandle {
    public uint low;
    public uint high;
}

public struct RayHit {
    public float t;
    public float2 barycentrics;
    public uint instanceIndex;
    public uint instanceId;
    public uint primitiveIndex;
    public uint geometryIndex;
    public bool frontFace;
}

public struct RayCandidate {
    public float t;
    public float2 barycentrics;
    public uint instanceIndex;
    public uint instanceId;
    public uint primitiveIndex;
    public uint geometryIndex;
    public bool frontFace;
    public float3 objectRayOrigin;
    public float3 objectRayDirection;
}

public enum HitVerdict : uint {
    Ignore = 0,
    Accept = 1,
    AcceptAndEnd = 2,
}

//! Called for non-opaque triangle candidates only, in no fixed order. On the software path a filter must not trace rays, because all traversals share one stack.
public interface IHitFilter {
    [mutating] HitVerdict filter(RayCandidate candidate);
}

#if !defined(NOORRHI_RAYTRACING_FALLBACK)

RaytracingAccelerationStructure noorrhi_scene(AccelerationStructureHandle scene) {
    return RaytracingAccelerationStructure(uint64_t(scene.low) | uint64_t(scene.high) << 32);
}

struct NoorRHIAcceptAllHits : IHitFilter {
    [mutating] HitVerdict filter(RayCandidate candidate) { return HitVerdict.Accept; }
}

public bool traceRayInline<F : IHitFilter>(AccelerationStructureHandle scene, uint flags, uint mask, RayDesc ray, inout F filter, out RayHit hit) {
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(noorrhi_scene(scene), flags, mask, ray);
    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE)
            continue;
        RayCandidate candidate;
        candidate.t = query.CandidateTriangleRayT();
        candidate.barycentrics = query.CandidateTriangleBarycentrics();
        candidate.instanceIndex = query.CandidateInstanceIndex();
        candidate.instanceId = query.CandidateInstanceID();
        candidate.primitiveIndex = query.CandidatePrimitiveIndex();
        candidate.geometryIndex = query.CandidateGeometryIndex();
        candidate.frontFace = query.CandidateTriangleFrontFace();
        candidate.objectRayOrigin = query.CandidateObjectRayOrigin();
        candidate.objectRayDirection = query.CandidateObjectRayDirection();
        const HitVerdict verdict = filter.filter(candidate);
        if (verdict == HitVerdict.Ignore)
            continue;
        query.CommitNonOpaqueTriangleHit();
        if (verdict == HitVerdict.AcceptAndEnd)
            query.Abort();
    }
    hit = {};
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return false;
    hit.t = query.CommittedRayT();
    hit.barycentrics = query.CommittedTriangleBarycentrics();
    hit.instanceIndex = query.CommittedInstanceIndex();
    hit.instanceId = query.CommittedInstanceID();
    hit.primitiveIndex = query.CommittedPrimitiveIndex();
    hit.geometryIndex = query.CommittedGeometryIndex();
    hit.frontFace = query.CommittedTriangleFrontFace();
    return true;
}

public bool traceRayInline(AccelerationStructureHandle scene, uint flags, uint mask, RayDesc ray, out RayHit hit) {
    NoorRHIAcceptAllHits filter;
    return traceRayInline(scene, flags, mask, ray, filter, hit);
}

public void traceRay<P>(AccelerationStructureHandle scene, uint flags, uint mask, uint hitGroupOffset, uint hitGroupStride, uint missIndex, RayDesc ray,
                        inout P payload) {
    TraceRay(noorrhi_scene(scene), flags, mask, hitGroupOffset, hitGroupStride, missIndex, ray, payload);
}

public uint3 launchIndex() { return DispatchRaysIndex(); }
public uint3 launchSize() { return DispatchRaysDimensions(); }
public float hitT() { return RayTCurrent(); }
public float rayTMin() { return RayTMin(); }
public uint hitInstanceIndex() { return InstanceIndex(); }
public uint hitInstanceId() { return InstanceID(); }
public uint hitPrimitiveIndex() { return PrimitiveIndex(); }
public uint hitGeometryIndex() { return GeometryIndex(); }
public bool hitFrontFace() { return HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE; }
public float3 worldRayOrigin() { return WorldRayOrigin(); }
public float3 worldRayDirection() { return WorldRayDirection(); }
public float3x4 objectToWorld() { return ObjectToWorld3x4(); }
public float3x4 worldToObject() { return WorldToObject3x4(); }
public void ignoreHit() { IgnoreHit(); }
public void acceptHitAndEndSearch() { AcceptHitAndEndSearch(); }

#else

//! The TLAS needs at most 64 entries. A BLAS path holds one entry per wide level, and RayTracingBuild::clusterLevels + 1 bounds the levels.
static const uint noorrhi_stack_size = 64 + 128;
static uint noorrhi_stack[noorrhi_stack_size];
static const uint noorrhi_node_words = 4;
static const uint noorrhi_wide_node_words = 6;
static const uint noorrhi_instance_words = 7;
static const uint noorrhi_triangle_words = 3;
static const uint noorrhi_none = 0xffffffffu;

static const uint noorrhi_empty_child = 0xffffffffu;
static const uint noorrhi_verdict_ignore = 0;
static const uint noorrhi_verdict_accept = 1;
static const uint noorrhi_verdict_end = 2;

static const uint noorrhi_instance_cull_disable = 1;
static const uint noorrhi_instance_front_counter_clockwise = 2;
static const uint noorrhi_instance_force_opaque = 4;
static const uint noorrhi_instance_force_no_opaque = 8;

//! Slang allows the built-in type only in ray tracing stages, so WGSL compute rejects it.
public struct BuiltInTriangleIntersectionAttributes {
    public float2 barycentrics;
}

struct NoorRHICandidate {
    float t;
    float2 barycentrics;
    uint instanceIndex;
    uint instanceId;
    uint primitiveIndex;
    uint geometryIndex;
    uint hitGroupBase;
    bool frontFace;
    float3 objectRayOrigin;
    float3 objectRayDirection;
}

interface INoorRHIHitFilter {
    [mutating] uint filter(NoorRHICandidate candidate, RayDesc ray);
}

struct NoorRHIAcceptAll : INoorRHIHitFilter {
    [mutating] uint filter(NoorRHICandidate candidate, RayDesc ray) { return noorrhi_verdict_accept; }
}

struct NoorRHIPublicFilter<F : IHitFilter> : INoorRHIHitFilter {
    F inner;

    [mutating] uint filter(NoorRHICandidate candidate, RayDesc ray) {
        RayCandidate publicCandidate;
        publicCandidate.t = candidate.t;
        publicCandidate.barycentrics = candidate.barycentrics;
        publicCandidate.instanceIndex = candidate.instanceIndex;
        publicCandidate.instanceId = candidate.instanceId;
        publicCandidate.primitiveIndex = candidate.primitiveIndex;
        publicCandidate.geometryIndex = candidate.geometryIndex;
        publicCandidate.frontFace = candidate.frontFace;
        publicCandidate.objectRayOrigin = candidate.objectRayOrigin;
        publicCandidate.objectRayDirection = candidate.objectRayDirection;
        return uint(inner.filter(publicCandidate));
    }
}

Ptr<uint4> noorrhi_words(uint low, uint high) {
    Ptr<uint4> pointer;
    pointer.low = low;
    pointer.high = high;
    return pointer;
}

float noorrhi_safe(float value) {
    return abs(value) > 1e-20 ? value : (value < 0.0 ? -1e-20 : 1e-20);
}

bool noorrhi_box(uint4 low, uint4 high, float3 origin, float3 inverse, float tMin, float tMax, out float tNear) {
    const float3 first = (asfloat(low.xyz) - origin) * inverse;
    const float3 second = (asfloat(high.xyz) - origin) * inverse;
    const float3 near = min(first, second);
    const float3 far = max(first, second);
    tNear = max(max(near.x, near.y), max(near.z, tMin));
    return low.w != noorrhi_empty_child && tNear <= min(min(far.x, far.y), min(far.z, tMax));
}

float3 noorrhi_transform_point(Ptr<uint4> rows, float3 point) {
    const float4 x = asfloat(rows[0]);
    const float4 y = asfloat(rows[1]);
    const float4 z = asfloat(rows[2]);
    return float3(dot(x.xyz, point) + x.w, dot(y.xyz, point) + y.w, dot(z.xyz, point) + z.w);
}

float3 noorrhi_transform_vector(Ptr<uint4> rows, float3 vector) {
    return float3(dot(asfloat(rows[0]).xyz, vector), dot(asfloat(rows[1]).xyz, vector), dot(asfloat(rows[2]).xyz, vector));
}

struct NoorRHITraversal {
    float tMax;
    bool found;
    bool done;
    NoorRHICandidate hit;
}

struct NoorRHIInstanceRay {
    float3 origin;
    float3 direction;
    float3 inverse;
    uint octant;
    uint item;
    uint4 info;
}

uint noorrhi_byte(uint2 plane, uint slot) {
    return (plane[slot >> 2] >> ((slot & 3) * 8)) & 0xffu;
}

float noorrhi_scale(uint exponents, uint axis) {
    return asfloat(((exponents >> (axis * 8)) & 0xffu) << 23);
}

//! A wide node is 6 words: origin and exponents, 8 child descriptors, then the quantized low and high planes of 8 boxes, 8 bytes per axis.
uint noorrhi_wide_hits(Ptr<uint4> node, NoorRHIInstanceRay instanceRay, float tMin, float tMax) {
    const uint4 header = node[0];
    const uint4 descriptors0 = node[1];
    const uint4 descriptors1 = node[2];
    const uint4 planes0 = node[3];
    const uint4 planes1 = node[4];
    const uint4 planes2 = node[5];
    const float3 scale = float3(noorrhi_scale(header.w, 0), noorrhi_scale(header.w, 1), noorrhi_scale(header.w, 2));
    const float3 slope = scale * instanceRay.inverse;
    const float3 offset = (asfloat(header.xyz) - instanceRay.origin) * instanceRay.inverse;
    const uint octant = instanceRay.octant;
    const uint2 nearX = (octant & 1) != 0 ? planes1.zw : planes0.xy;
    const uint2 farX = (octant & 1) != 0 ? planes0.xy : planes1.zw;
    const uint2 nearY = (octant & 2) != 0 ? planes2.xy : planes0.zw;
    const uint2 farY = (octant & 2) != 0 ? planes0.zw : planes2.xy;
    const uint2 nearZ = (octant & 4) != 0 ? planes2.zw : planes1.xy;
    const uint2 farZ = (octant & 4) != 0 ? planes1.xy : planes2.zw;
    uint hits = 0;
    [unroll]
    for (uint slot = 0; slot < 8; ++slot) {
        const uint descriptor = slot < 4 ? descriptors0[slot] : descriptors1[slot - 4];
        const float3 near = float3(noorrhi_byte(nearX, slot), noorrhi_byte(nearY, slot), noorrhi_byte(nearZ, slot)) * slope + offset;
        const float3 far = float3(noorrhi_byte(farX, slot), noorrhi_byte(farY, slot), noorrhi_byte(farZ, slot)) * slope + offset;
        const float entry = max(max(near.x, near.y), max(near.z, tMin));
        const float exit = min(min(far.x, far.y), min(far.z, tMax));
        if (descriptor != noorrhi_none && entry <= exit)
            hits |= 1u << (slot ^ octant);
    }
    return hits;
}

void noorrhi_test_triangles<F : INoorRHIHitFilter>(Ptr<uint4> triangles, uint first, uint count, NoorRHIInstanceRay instanceRay, RayDesc ray, uint flags,
                                                   inout F filter, inout NoorRHITraversal traversal) {
    const uint instanceFlags = instanceRay.info.w >> 24;
    for (uint primitive = first; primitive < first + count; ++primitive) {
        const Ptr<uint4> triangle = triangles + primitive * noorrhi_triangle_words;
        const uint4 word0 = triangle[0];
        const uint4 word1 = triangle[1];
        const uint4 word2 = triangle[2];
        const float3 vertex = asfloat(word0.xyz);
        const float3 edge1 = asfloat(word1.xyz);
        const float3 edge2 = asfloat(word2.xyz);
        const float3 p = cross(instanceRay.direction, edge2);
        const float determinant = dot(edge1, p);
        if (determinant == 0.0)
            continue;
        const float reciprocal = 1.0 / determinant;
        const float3 s = instanceRay.origin - vertex;
        const float u = dot(s, p) * reciprocal;
        if (u < 0.0 || u > 1.0)
            continue;
        const float3 q = cross(s, edge1);
        const float v = dot(instanceRay.direction, q) * reciprocal;
        if (v < 0.0 || u + v > 1.0)
            continue;
        const float t = dot(edge2, q) * reciprocal;
        if (t < ray.TMin || t > traversal.tMax)
            continue;
        const bool frontFace = (determinant > 0.0) != ((instanceFlags & noorrhi_instance_front_counter_clockwise) != 0);
        if ((instanceFlags & noorrhi_instance_cull_disable) == 0) {
            if ((flags & RAY_FLAG_CULL_BACK_FACING_TRIANGLES) != 0 && !frontFace)
                continue;
            if ((flags & RAY_FLAG_CULL_FRONT_FACING_TRIANGLES) != 0 && frontFace)
                continue;
        }
        bool opaque = (word2.w & 1u) != 0;
        if ((instanceFlags & noorrhi_instance_force_opaque) != 0)
            opaque = true;
        if ((instanceFlags & noorrhi_instance_force_no_opaque) != 0)
            opaque = false;
        if ((flags & RAY_FLAG_FORCE_OPAQUE) != 0)
            opaque = true;
        if ((flags & RAY_FLAG_FORCE_NON_OPAQUE) != 0)
            opaque = false;
        if ((flags & RAY_FLAG_CULL_OPAQUE) != 0 && opaque)
            continue;
        if ((flags & RAY_FLAG_CULL_NON_OPAQUE) != 0 && !opaque)
            continue;
        NoorRHICandidate candidate;
        candidate.t = t;
        candidate.barycentrics = float2(u, v);
        candidate.instanceIndex = instanceRay.item;
        candidate.instanceId = instanceRay.info.z & 0x00ffffffu;
        candidate.primitiveIndex = word0.w;
        candidate.geometryIndex = word1.w;
        candidate.hitGroupBase = instanceRay.info.w & 0x00ffffffu;
        candidate.frontFace = frontFace;
        candidate.objectRayOrigin = instanceRay.origin;
        candidate.objectRayDirection = instanceRay.direction;
        const uint verdict = opaque ? noorrhi_verdict_accept : filter.filter(candidate, ray);
        if (verdict == noorrhi_verdict_ignore)
            continue;
        traversal.hit = candidate;
        traversal.tMax = t;
        traversal.found = true;
        if (verdict == noorrhi_verdict_end || (flags & RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH) != 0) {
            traversal.done = true;
            return;
        }
    }
}

//! Children are visited in the order of their slot index xor the ray octant, which the builder made front to back. A group is a node index above an 8-bit hit mask.
void noorrhi_traverse_blas<F : INoorRHIHitFilter>(NoorRHIInstanceRay instanceRay, RayDesc ray, uint flags, inout F filter,
                                                  inout NoorRHITraversal traversal, uint base) {
    const Ptr<uint4> blas = noorrhi_words(instanceRay.info.x, instanceRay.info.y);
    const Ptr<uint4> nodes = blas + 4u;
    const Ptr<uint4> triangles = blas + (blas[0].z >> 4);
    uint depth = base;
    uint group = noorrhi_wide_hits(nodes, instanceRay, ray.TMin, traversal.tMax);
    for (;;) {
        if ((group & 0xffu) == 0) {
            if (depth == base)
                return;
            group = noorrhi_stack[--depth];
            continue;
        }
        const uint slot = firstbitlow(group) ^ instanceRay.octant;
        group &= group - 1;
        const uint descriptor = nodes[(group >> 8) * noorrhi_wide_node_words + 1 + (slot >> 2)][slot & 3];
        const uint reference = descriptor >> 3;
        const uint count = descriptor & 7u;
        if (count == 0) {
            if ((group & 0xffu) != 0)
                noorrhi_stack[depth++] = group;
            group = reference << 8 | noorrhi_wide_hits(nodes + reference * noorrhi_wide_node_words, instanceRay, ray.TMin, traversal.tMax);
            continue;
        }
        noorrhi_test_triangles(triangles, reference, count, instanceRay, ray, flags, filter, traversal);
        if (traversal.done)
            return;
    }
}

float3 noorrhi_inverse(float3 direction) {
    return 1.0 / float3(noorrhi_safe(direction.x), noorrhi_safe(direction.y), noorrhi_safe(direction.z));
}

bool noorrhi_traverse<F : INoorRHIHitFilter>(AccelerationStructureHandle scene, uint flags, uint mask, RayDesc ray, inout F filter,
                                             out NoorRHICandidate hit) {
    const Ptr<uint4> tlas = noorrhi_words(scene.low, scene.high);
    const Ptr<uint4> tlasNodes = tlas + 4u;
    const Ptr<uint4> instances = tlas + (tlas[0].z >> 4);
    const float3 worldInverse = noorrhi_inverse(ray.Direction);
    NoorRHITraversal traversal = {};
    traversal.tMax = ray.TMax;
    uint depth = 0;
    uint node = 0;
    for (;;) {
        const Ptr<uint4> entry = tlasNodes + node * noorrhi_node_words;
        const uint4 low0 = entry[0];
        const uint4 high0 = entry[1];
        const uint4 low1 = entry[2];
        const uint4 high1 = entry[3];
        float near0;
        float near1;
        const bool hit0 = noorrhi_box(low0, high0, ray.Origin, worldInverse, ray.TMin, traversal.tMax, near0);
        const bool hit1 = noorrhi_box(low1, high1, ray.Origin, worldInverse, ray.TMin, traversal.tMax, near1);
        for (uint side = 0; side < 2; ++side) {
            const uint4 low = side == 0 ? low0 : low1;
            const uint4 high = side == 0 ? high0 : high1;
            if (!(side == 0 ? hit0 : hit1) || high.w == 0)
                continue;
            for (uint item = low.w; item < low.w + high.w; ++item) {
                const Ptr<uint4> instance = instances + item * noorrhi_instance_words;
                NoorRHIInstanceRay instanceRay;
                instanceRay.info = instance[6];
                if (((instanceRay.info.z >> 24) & mask) == 0)
                    continue;
                instanceRay.item = item;
                instanceRay.origin = noorrhi_transform_point(instance, ray.Origin);
                instanceRay.direction = noorrhi_transform_vector(instance, ray.Direction);
                instanceRay.inverse = noorrhi_inverse(instanceRay.direction);
                instanceRay.octant = (instanceRay.direction.x < 0.0 ? 1u : 0u) | (instanceRay.direction.y < 0.0 ? 2u : 0u)
                                     | (instanceRay.direction.z < 0.0 ? 4u : 0u);
                noorrhi_traverse_blas(instanceRay, ray, flags, filter, traversal, depth);
                if (traversal.done) {
                    hit = traversal.hit;
                    return true;
                }
            }
        }
        const bool inner0 = hit0 && high0.w == 0;
        const bool inner1 = hit1 && high1.w == 0;
        if (inner0 && inner1) {
            const bool secondFirst = near1 < near0;
            node = secondFirst ? low1.w : low0.w;
            noorrhi_stack[depth++] = secondFirst ? low0.w : low1.w;
            continue;
        }
        if (inner0 || inner1) {
            node = inner0 ? low0.w : low1.w;
            continue;
        }
        if (depth == 0)
            break;
        node = noorrhi_stack[--depth];
    }
    hit = traversal.hit;
    return traversal.found;
}

RayHit noorrhi_ray_hit(NoorRHICandidate candidate) {
    RayHit hit;
    hit.t = candidate.t;
    hit.barycentrics = candidate.barycentrics;
    hit.instanceIndex = candidate.instanceIndex;
    hit.instanceId = candidate.instanceId;
    hit.primitiveIndex = candidate.primitiveIndex;
    hit.geometryIndex = candidate.geometryIndex;
    hit.frontFace = candidate.frontFace;
    return hit;
}

public bool traceRayInline(AccelerationStructureHandle scene, uint flags, uint mask, RayDesc ray, out RayHit hit) {
    hit = {};
    NoorRHIAcceptAll filter;
    NoorRHICandidate candidate;
    if (!noorrhi_traverse(scene, flags, mask, ray, filter, candidate))
        return false;
    hit = noorrhi_ray_hit(candidate);
    return true;
}

public bool traceRayInline<F : IHitFilter>(AccelerationStructureHandle scene, uint flags, uint mask, RayDesc ray, inout F filter, out RayHit hit) {
    hit = {};
    NoorRHIPublicFilter<F> adapter;
    adapter.inner = filter;
    NoorRHICandidate candidate;
    const bool found = noorrhi_traverse(scene, flags, mask, ray, adapter, candidate);
    filter = adapter.inner;
    if (!found)
        return false;
    hit = noorrhi_ray_hit(candidate);
    return true;
}

#if defined(NOORRHI_RAYTRACING_LAUNCH)

public interface IRayPayload {
    [mutating] void noorrhiMiss(uint shader);
    [mutating] void noorrhiClosestHit(uint shader, BuiltInTriangleIntersectionAttributes attributes);
    [mutating] void noorrhiAnyHit(uint shader, BuiltInTriangleIntersectionAttributes attributes);
}

struct NoorRHIHitState {
    float t;
    float tMin;
    uint instanceIndex;
    uint instanceId;
    uint primitiveIndex;
    uint geometryIndex;
    bool frontFace;
    float3 origin;
    float3 direction;
    uint instanceLow;
    uint instanceHigh;
}

static uint3 noorrhi_launch_index;
static NoorRHIHitState noorrhi_hit;
static uint noorrhi_verdict;

uint4 noorrhi_launch(uint index) { return noorrhi_root_words(index); }

public bool beginRayLaunch(uint3 id) {
    if (any(id >= noorrhi_launch(0).xyz))
        return false;
    noorrhi_launch_index = id;
    return true;
}

public uint3 launchIndex() { return noorrhi_launch_index; }
public uint3 launchSize() { return noorrhi_launch(0).xyz; }
public float hitT() { return noorrhi_hit.t; }
public float rayTMin() { return noorrhi_hit.tMin; }
public uint hitInstanceIndex() { return noorrhi_hit.instanceIndex; }
public uint hitInstanceId() { return noorrhi_hit.instanceId; }
public uint hitPrimitiveIndex() { return noorrhi_hit.primitiveIndex; }
public uint hitGeometryIndex() { return noorrhi_hit.geometryIndex; }
public bool hitFrontFace() { return noorrhi_hit.frontFace; }
public float3 worldRayOrigin() { return noorrhi_hit.origin; }
public float3 worldRayDirection() { return noorrhi_hit.direction; }
public void ignoreHit() { noorrhi_verdict = noorrhi_verdict_ignore; }
public void acceptHitAndEndSearch() { noorrhi_verdict = noorrhi_verdict_end; }

float3x4 noorrhi_rows(uint first) {
    const Ptr<uint4> rows = noorrhi_words(noorrhi_hit.instanceLow, noorrhi_hit.instanceHigh) + first;
    return float3x4(asfloat(rows[0]), asfloat(rows[1]), asfloat(rows[2]));
}

public float3x4 worldToObject() { return noorrhi_rows(0); }
public float3x4 objectToWorld() { return noorrhi_rows(3); }

void noorrhi_begin_ray(RayDesc ray) {
    noorrhi_hit = {};
    noorrhi_hit.tMin = ray.TMin;
    noorrhi_hit.origin = ray.Origin;
    noorrhi_hit.direction = ray.Direction;
}

BuiltInTriangleIntersectionAttributes noorrhi_set_hit(AccelerationStructureHandle scene, NoorRHICandidate candidate) {
    const Ptr<uint4> tlas = noorrhi_words(scene.low, scene.high);
    const Ptr<uint4> instance = tlas + (tlas[0].z >> 4) + candidate.instanceIndex * noorrhi_instance_words;
    noorrhi_hit.t = candidate.t;
    noorrhi_hit.instanceIndex = candidate.instanceIndex;
    noorrhi_hit.instanceId = candidate.instanceId;
    noorrhi_hit.primitiveIndex = candidate.primitiveIndex;
    noorrhi_hit.geometryIndex = candidate.geometryIndex;
    noorrhi_hit.frontFace = candidate.frontFace;
    noorrhi_hit.instanceLow = instance.low;
    noorrhi_hit.instanceHigh = instance.high;
    BuiltInTriangleIntersectionAttributes attributes;
    attributes.barycentrics = candidate.barycentrics;
    return attributes;
}

uint2 noorrhi_hit_group(NoorRHICandidate candidate, uint offset, uint stride) {
    const uint record = candidate.hitGroupBase + offset + candidate.geometryIndex * stride;
    if (record >= noorrhi_launch(1).x)
        return uint2(noorrhi_none, noorrhi_none);
    const uint4 tables = noorrhi_launch(2);
    Ptr<uint2> groups;
    groups.low = tables.z;
    groups.high = tables.w;
    return groups[record];
}

struct NoorRHIAnyHitFilter<P : IRayPayload> : INoorRHIHitFilter {
    P payload;
    AccelerationStructureHandle scene;
    uint offset;
    uint stride;

    [mutating] uint filter(NoorRHICandidate candidate, RayDesc ray) {
        const uint shader = noorrhi_hit_group(candidate, offset, stride).y;
        if (shader == noorrhi_none)
            return noorrhi_verdict_accept;
        const BuiltInTriangleIntersectionAttributes attributes = noorrhi_set_hit(scene, candidate);
        noorrhi_verdict = noorrhi_verdict_accept;
        payload.noorrhiAnyHit(shader, attributes);
        return noorrhi_verdict;
    }
}

public void traceRay<P : IRayPayload>(AccelerationStructureHandle scene, uint flags, uint mask, uint hitGroupOffset, uint hitGroupStride, uint missIndex,
                                      RayDesc ray, inout P payload) {
    noorrhi_begin_ray(ray);
    NoorRHIAnyHitFilter<P> filter;
    filter.payload = payload;
    filter.scene = scene;
    filter.offset = hitGroupOffset;
    filter.stride = hitGroupStride;
    NoorRHICandidate candidate;
    const bool found = noorrhi_traverse(scene, flags, mask, ray, filter, candidate);
    payload = filter.payload;
    if (found) {
        if ((flags & RAY_FLAG_SKIP_CLOSEST_HIT_SHADER) != 0)
            return;
        const uint shader = noorrhi_hit_group(candidate, hitGroupOffset, hitGroupStride).x;
        if (shader == noorrhi_none)
            return;
        const BuiltInTriangleIntersectionAttributes attributes = noorrhi_set_hit(scene, candidate);
        payload.noorrhiClosestHit(shader, attributes);
        return;
    }
    const uint4 counts = noorrhi_launch(0);
    if (missIndex >= counts.w)
        return;
    const uint4 tables = noorrhi_launch(2);
    Ptr<uint> misses;
    misses.low = tables.x;
    misses.high = tables.y;
    const uint shader = misses[missIndex];
    if (shader != noorrhi_none)
        payload.noorrhiMiss(shader);
}

#endif

#endif
