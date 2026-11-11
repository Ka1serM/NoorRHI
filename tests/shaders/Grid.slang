static const uint gridShiftedInstance = 7u;
static const uint gridWrongDistance = 0xffffffffu;

RayDesc makeGridRay(uint2 pixel, uint width) {
    const float cell = 2.0f / float(width);
    RayDesc ray;
    ray.Origin = float3(-1.0f + (float(pixel.x) + 0.25f) * cell, -1.0f + (float(pixel.y) + 0.75f) * cell, 1.0f);
    ray.Direction = float3(0.0f, 0.0f, -1.0f);
    ray.TMin = 0.001f;
    ray.TMax = 10.0f;
    return ray;
}

uint encodeGridHit(float t, uint instanceId, uint primitive) {
    const float expected = instanceId == gridShiftedInstance ? 1.5f : 1.0f;
    if (abs(t - expected) > 1e-3f)
        return gridWrongDistance;
    return instanceId << 24 | (primitive + 1u);
}
