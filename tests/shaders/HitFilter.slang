#include "RayTracing.h"

static const uint filterAcceptAll = 0u;
static const uint filterIgnoreAll = 1u;
static const uint filterIgnoreFirstLayer = 2u;
static const uint filterAcceptAndEnd = 3u;
static const uint filterObjectRay = 4u;

struct TestFilter : IHitFilter {
    uint mode;
    uint calls;

    [mutating] HitVerdict filter(RayCandidate candidate) {
        ++calls;
        if (mode == filterIgnoreAll)
            return HitVerdict.Ignore;
        if (mode == filterIgnoreFirstLayer)
            return candidate.instanceId == 0u ? HitVerdict.Ignore : HitVerdict.Accept;
        if (mode == filterAcceptAndEnd)
            return HitVerdict.AcceptAndEnd;
        if (mode == filterObjectRay) {
            const bool matches = abs(candidate.objectRayOrigin.z - candidate.t) < 1e-3f && all(candidate.objectRayDirection == float3(0.0f, 0.0f, -1.0f));
            return matches ? HitVerdict.Accept : HitVerdict.Ignore;
        }
        return HitVerdict.Accept;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void hitFilterMain(uint3 id : SV_DispatchThreadID) {
    const RayTracing::FilterArguments arguments = rootArguments<RayTracing::FilterArguments>();
    RayDesc ray;
    ray.Origin = float3(0.0f, 0.0f, 1.0f);
    ray.Direction = float3(0.0f, 0.0f, -1.0f);
    ray.TMin = 0.001f;
    ray.TMax = 10.0f;
    TestFilter filter;
    filter.mode = arguments.mode;
    filter.calls = 0u;
    RayHit hit;
    uint value = 0u;
    if (traceRayInline(arguments.scene, arguments.flags, arguments.mask, ray, filter, hit))
        value = 1u | hit.instanceId << 1 | uint(round(hit.t * 2.0f)) << 4;
    arguments.result[0] = value | filter.calls << 8;
}
