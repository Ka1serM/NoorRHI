#include "Bench.h"
#include "BenchRays.slang"

[shader("compute")]
[numthreads(8, 8, 1)]
void benchQueryMain(uint3 id : SV_DispatchThreadID) {
    const Bench::TraceArguments arguments = rootArguments<Bench::TraceArguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    RayHit hit;
    float t = 0.0f;
    if (traceRayInline(arguments.scene, RAY_FLAG_NONE, 0xff, makeBenchRay(id.xy, arguments), hit))
        t = hit.t;
    arguments.result[id.y * arguments.width + id.x] = t;
}
