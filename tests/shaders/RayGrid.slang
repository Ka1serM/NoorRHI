#include "RayTracing.h"
#include "Grid.slang"

struct GridPayload {
    uint value;
};

[shader("raygeneration")]
void gridRayGenMain() {
    const RayTracing::Arguments arguments = rootArguments<RayTracing::Arguments>();
    const uint2 pixel = launchIndex().xy;
    GridPayload payload;
    payload.value = 0u;
    traceRay(arguments.scene, RAY_FLAG_NONE, 0xff, 0, 0, 0, makeGridRay(pixel, arguments.width), payload);
    arguments.result[pixel.y * arguments.width + pixel.x] = payload.value;
}

[shader("miss")]
void gridMissMain(inout GridPayload payload) {
    payload.value = 0u;
}

[shader("closesthit")]
void gridClosestHitMain(inout GridPayload payload, BuiltInTriangleIntersectionAttributes attributes) {
    payload.value = encodeGridHit(hitT(), hitInstanceId(), hitPrimitiveIndex());
}
