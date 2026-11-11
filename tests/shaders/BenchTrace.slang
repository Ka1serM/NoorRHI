#include "Bench.h"
#include "BenchRays.slang"

struct BenchPayload {
    float t;
};

[shader("raygeneration")]
void benchRayGenMain() {
    const Bench::TraceArguments arguments = rootArguments<Bench::TraceArguments>();
    const uint2 pixel = launchIndex().xy;
    BenchPayload payload;
    payload.t = 0.0f;
    traceRay(arguments.scene, RAY_FLAG_NONE, 0xff, 0, 0, 0, makeBenchRay(pixel, arguments), payload);
    arguments.result[pixel.y * arguments.width + pixel.x] = payload.t;
}

[shader("miss")]
void benchMissMain(inout BenchPayload payload) {
    payload.t = 0.0f;
}

[shader("closesthit")]
void benchClosestHitMain(inout BenchPayload payload, BuiltInTriangleIntersectionAttributes attributes) {
    payload.t = hitT();
}
