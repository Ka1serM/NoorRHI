#include "RayTracing.h"

struct Payload {
    uint hit;
};

RayDesc makeRay(uint2 pixel, uint2 size) {
    const float2 uv = (float2(pixel) + 0.5f) / float2(size) * 2.0f - 1.0f;
    RayDesc ray;
    ray.Origin = float3(uv, 1.0f);
    ray.Direction = float3(0.0f, 0.0f, -1.0f);
    ray.TMin = 0.001f;
    ray.TMax = 10.0f;
    return ray;
}

uint classifyHit(float t, bool frontFace) {
    if (abs(t - 1.0f) > 1e-3f)
        return 3u;
    return frontFace ? 2u : 1u;
}

[shader("raygeneration")]
void rayGenMain() {
    const RayTracing::Arguments arguments = rootArguments<RayTracing::Arguments>();
    const uint2 pixel = launchIndex().xy;
    Payload payload;
    payload.hit = 0u;
    traceRay(arguments.scene, RAY_FLAG_NONE, 0xff, 0, 0, 0, makeRay(pixel, launchSize().xy), payload);
    arguments.result[pixel.y * arguments.width + pixel.x] = payload.hit;
}

[shader("miss")]
void missMain(inout Payload payload) {
    payload.hit = 0u;
}

[shader("closesthit")]
void closestHitMain(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes) {
    payload.hit = classifyHit(hitT(), hitFrontFace());
}
