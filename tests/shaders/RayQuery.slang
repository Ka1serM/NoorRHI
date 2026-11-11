#include "RayTracing.h"
#include "Grid.slang"

[shader("compute")]
[numthreads(8, 8, 1)]
void rayQueryMain(uint3 id : SV_DispatchThreadID) {
    const RayTracing::Arguments arguments = rootArguments<RayTracing::Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.width)
        return;
    const float2 uv = (float2(id.xy) + 0.5f) / float(arguments.width) * 2.0f - 1.0f;
    RayDesc ray;
    ray.Origin = float3(uv, 1.0f);
    ray.Direction = float3(0.0f, 0.0f, -1.0f);
    ray.TMin = 0.001f;
    ray.TMax = 10.0f;
    RayHit hit;
    uint value = 0u;
    if (traceRayInline(arguments.scene, RAY_FLAG_NONE, 0xff, ray, hit))
        value = abs(hit.t - 1.0f) > 1e-3f ? 3u : (hit.frontFace ? 2u : 1u);
    arguments.result[id.y * arguments.width + id.x] = value;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void gridQueryMain(uint3 id : SV_DispatchThreadID) {
    const RayTracing::Arguments arguments = rootArguments<RayTracing::Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.width)
        return;
    RayHit hit;
    uint value = 0u;
    if (traceRayInline(arguments.scene, RAY_FLAG_NONE, 0xff, makeGridRay(id.xy, arguments.width), hit))
        value = encodeGridHit(hit.t, hit.instanceId, hit.primitiveIndex);
    arguments.result[id.y * arguments.width + id.x] = value;
}
