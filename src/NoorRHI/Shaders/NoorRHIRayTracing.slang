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

#if !defined(NOORRHI_RAYTRACING_FALLBACK)

RaytracingAccelerationStructure noorrhi_scene(AccelerationStructureHandle scene) {
    return RaytracingAccelerationStructure(uint64_t(scene.low) | uint64_t(scene.high) << 32);
}

public bool traceRayInline(AccelerationStructureHandle scene, uint flags, uint mask, RayDesc ray, out RayHit hit) {
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(noorrhi_scene(scene), flags, mask, ray);
    while (query.Proceed())
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            query.CommitNonOpaqueTriangleHit();
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

static const uint noorrhi_stack_size = 96;
static const uint noorrhi_node_words = 4;
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
    uint primitiveIndex;
    uint geometryIndex;
    uint hitGroupBase;
    bool frontFace;
}

interface INoorRHIHitFilter {
    [mutating] uint filter(NoorRHICandidate candidate, RayDesc ray);
}

struct NoorRHIAcceptAll : INoorRHIHitFilter {
    [mutating] uint filter(NoorRHICandidate candidate, RayDesc ray) { return noorrhi_verdict_accept; }
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

bool noorrhi_traverse<F : INoorRHIHitFilter>(AccelerationStructureHandle scene, uint flags, uint mask, RayDesc ray, inout F filter,
                                             out NoorRHICandidate hit) {
    hit = {};
    const Ptr<uint4> tlas = noorrhi_words(scene.low, scene.high);
    const Ptr<uint4> tlasNodes = tlas + 4u;
    const Ptr<uint4> instances = tlas + (tlas[0].z >> 4);
    const float3 worldInverse = 1.0 / float3(noorrhi_safe(ray.Direction.x), noorrhi_safe(ray.Direction.y), noorrhi_safe(ray.Direction.z));
    float tMax = ray.TMax;
    bool found = false;
    uint stack[noorrhi_stack_size] = {};
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
        const bool hit0 = noorrhi_box(low0, high0, ray.Origin, worldInverse, ray.TMin, tMax, near0);
        const bool hit1 = noorrhi_box(low1, high1, ray.Origin, worldInverse, ray.TMin, tMax, near1);
        for (uint side = 0; side < 2; ++side) {
            const uint4 low = side == 0 ? low0 : low1;
            const uint4 high = side == 0 ? high0 : high1;
            if (!(side == 0 ? hit0 : hit1) || high.w == 0)
                continue;
            for (uint item = low.w; item < low.w + high.w; ++item) {
                const Ptr<uint4> instance = instances + item * noorrhi_instance_words;
                const uint4 info = instance[6];
                if (((info.z >> 24) & mask) == 0)
                    continue;
                const uint instanceFlags = info.w >> 24;
                const float3 origin = noorrhi_transform_point(instance, ray.Origin);
                const float3 direction = noorrhi_transform_vector(instance, ray.Direction);
                const float3 inverse = 1.0 / float3(noorrhi_safe(direction.x), noorrhi_safe(direction.y), noorrhi_safe(direction.z));
                const Ptr<uint4> blas = noorrhi_words(info.x, info.y);
                const Ptr<uint4> blasNodes = blas + 4u;
                const Ptr<uint4> triangles = blas + (blas[0].z >> 4);
                const uint base = depth;
                uint blasNode = 0;
                for (;;) {
                    const Ptr<uint4> blasEntry = blasNodes + blasNode * noorrhi_node_words;
                    const uint4 blasLow0 = blasEntry[0];
                    const uint4 blasHigh0 = blasEntry[1];
                    const uint4 blasLow1 = blasEntry[2];
                    const uint4 blasHigh1 = blasEntry[3];
                    float blasNear0;
                    float blasNear1;
                    const bool blasHit0 = noorrhi_box(blasLow0, blasHigh0, origin, inverse, ray.TMin, tMax, blasNear0);
                    const bool blasHit1 = noorrhi_box(blasLow1, blasHigh1, origin, inverse, ray.TMin, tMax, blasNear1);
                    for (uint blasSide = 0; blasSide < 2; ++blasSide) {
                        const uint4 leafLow = blasSide == 0 ? blasLow0 : blasLow1;
                        const uint4 leafHigh = blasSide == 0 ? blasHigh0 : blasHigh1;
                        if (!(blasSide == 0 ? blasHit0 : blasHit1) || leafHigh.w == 0)
                            continue;
                        for (uint primitive = leafLow.w; primitive < leafLow.w + leafHigh.w; ++primitive) {
                            const Ptr<uint4> triangle = triangles + primitive * noorrhi_triangle_words;
                            const uint4 word0 = triangle[0];
                            const uint4 word1 = triangle[1];
                            const uint4 word2 = triangle[2];
                            const float3 vertex = asfloat(word0.xyz);
                            const float3 edge1 = asfloat(word1.xyz);
                            const float3 edge2 = asfloat(word2.xyz);
                            const float3 p = cross(direction, edge2);
                            const float determinant = dot(edge1, p);
                            if (determinant == 0.0)
                                continue;
                            const float reciprocal = 1.0 / determinant;
                            const float3 s = origin - vertex;
                            const float u = dot(s, p) * reciprocal;
                            if (u < 0.0 || u > 1.0)
                                continue;
                            const float3 q = cross(s, edge1);
                            const float v = dot(direction, q) * reciprocal;
                            if (v < 0.0 || u + v > 1.0)
                                continue;
                            const float t = dot(edge2, q) * reciprocal;
                            if (t < ray.TMin || t > tMax)
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
                            candidate.instanceIndex = item;
                            candidate.primitiveIndex = word0.w;
                            candidate.geometryIndex = word1.w;
                            candidate.hitGroupBase = info.w & 0x00ffffffu;
                            candidate.frontFace = frontFace;
                            const uint verdict = opaque ? noorrhi_verdict_accept : filter.filter(candidate, ray);
                            if (verdict == noorrhi_verdict_ignore)
                                continue;
                            hit = candidate;
                            tMax = t;
                            found = true;
                            if (verdict == noorrhi_verdict_end || (flags & RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH) != 0)
                                return true;
                        }
                    }
                    const bool inner0 = blasHit0 && blasHigh0.w == 0;
                    const bool inner1 = blasHit1 && blasHigh1.w == 0;
                    if (inner0 && inner1) {
                        const bool secondFirst = blasNear1 < blasNear0;
                        blasNode = secondFirst ? blasLow1.w : blasLow0.w;
                        stack[depth++] = secondFirst ? blasLow0.w : blasLow1.w;
                        continue;
                    }
                    if (inner0 || inner1) {
                        blasNode = inner0 ? blasLow0.w : blasLow1.w;
                        continue;
                    }
                    if (depth == base)
                        break;
                    blasNode = stack[--depth];
                }
            }
        }
        const bool inner0 = hit0 && high0.w == 0;
        const bool inner1 = hit1 && high1.w == 0;
        if (inner0 && inner1) {
            const bool secondFirst = near1 < near0;
            node = secondFirst ? low1.w : low0.w;
            stack[depth++] = secondFirst ? low0.w : low1.w;
            continue;
        }
        if (inner0 || inner1) {
            node = inner0 ? low0.w : low1.w;
            continue;
        }
        if (depth == 0)
            break;
        node = stack[--depth];
    }
    return found;
}

public bool traceRayInline(AccelerationStructureHandle scene, uint flags, uint mask, RayDesc ray, out RayHit hit) {
    hit = {};
    NoorRHIAcceptAll filter;
    NoorRHICandidate candidate;
    if (!noorrhi_traverse(scene, flags, mask, ray, filter, candidate))
        return false;
    const Ptr<uint4> tlas = noorrhi_words(scene.low, scene.high);
    hit.t = candidate.t;
    hit.barycentrics = candidate.barycentrics;
    hit.instanceIndex = candidate.instanceIndex;
    hit.instanceId = (tlas + (tlas[0].z >> 4) + candidate.instanceIndex * noorrhi_instance_words)[6].z & 0x00ffffffu;
    hit.primitiveIndex = candidate.primitiveIndex;
    hit.geometryIndex = candidate.geometryIndex;
    hit.frontFace = candidate.frontFace;
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
    noorrhi_hit.instanceId = instance[6].z & 0x00ffffffu;
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
