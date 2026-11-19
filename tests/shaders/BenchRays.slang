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
        const float3 cell = float3(randomBench(seed), randomBench(seed + 1u), randomBench(seed + 2u));
        ray.Origin = arguments.originLow.xyz + cell * (arguments.originHigh.xyz - arguments.originLow.xyz);
        const float z = randomBench(seed + 3u) * 2.0f - 1.0f;
        const float angle = randomBench(seed ^ 0x9e3779b9u) * 6.2831853f;
        const float radius = sqrt(max(0.0f, 1.0f - z * z));
        ray.Direction = float3(radius * cos(angle), z, radius * sin(angle));
        return ray;
    }
    const float2 uv = (float2(pixel) + 0.5f) / float2(float(arguments.width), float(arguments.height)) * 2.0f - 1.0f;
    const float3 forward = normalize(arguments.forward.xyz);
    const float3 right = normalize(cross(float3(0.0f, 1.0f, 0.0f), forward));
    const float3 up = cross(forward, right);
    const float aspect = float(arguments.width) / float(arguments.height);
    ray.Origin = arguments.eye.xyz;
    ray.Direction = normalize(forward + right * (uv.x * aspect * 0.6f) - up * (uv.y * 0.6f));
    return ray;
}
