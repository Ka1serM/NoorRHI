uint hashBench(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float randomBench(uint seed) {
    return float(hashBench(seed) >> 8) * (1.0f / 16777216.0f);
}

RayDesc makeBenchRay(uint2 pixel, Bench::TraceArguments arguments) {
    RayDesc ray;
    ray.TMin = 0.001f;
    ray.TMax = 100.0f;
    if (arguments.incoherent != 0u) {
        const uint seed = (pixel.y * arguments.width + pixel.x) * 4u;
        ray.Origin = float3(randomBench(seed) * arguments.extent, 0.3f + randomBench(seed + 1u) * 0.7f, randomBench(seed + 2u) * arguments.extent);
        const float z = randomBench(seed + 3u) * 2.0f - 1.0f;
        const float angle = randomBench(seed ^ 0x9e3779b9u) * 6.2831853f;
        const float radius = sqrt(max(0.0f, 1.0f - z * z));
        ray.Direction = float3(radius * cos(angle), z, radius * sin(angle));
        return ray;
    }
    const float2 uv = (float2(pixel) + 0.5f) / float2(float(arguments.width), float(arguments.height)) * 2.0f - 1.0f;
    const float3 forward = normalize(float3(0.0f, -0.35f, 1.0f));
    const float3 right = float3(1.0f, 0.0f, 0.0f);
    const float3 up = cross(forward, right);
    const float aspect = float(arguments.width) / float(arguments.height);
    ray.Origin = float3(arguments.extent * 0.5f, 1.5f, -1.0f);
    ray.Direction = normalize(forward + right * (uv.x * aspect * 0.6f) - up * (uv.y * 0.6f));
    return ray;
}
