#include "Sdf.h"

struct Varyings {
    float4 position : SV_Position;
};

[shader("vertex")]
Varyings vertMain(uint index : SV_VertexID) {
    const float2 corner = float2(float((index << 1u) & 2u), float(index & 2u));
    Varyings output;
    output.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

float smoothMin(float a, float b, float k) {
    const float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return lerp(b, a, h) - k * h * (1.0 - h);
}

float sphere(float3 p, float radius) {
    return length(p) - radius;
}

float torus(float3 p, float2 radii) {
    return length(float2(length(p.xz) - radii.x, p.y)) - radii.y;
}

float roundBox(float3 p, float3 halfSize, float radius) {
    const float3 q = abs(p) - halfSize;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - radius;
}

float3 rotateY(float3 p, float angle) {
    const float c = cos(angle);
    const float s = sin(angle);
    return float3(c * p.x + s * p.z, p.y, c * p.z - s * p.x);
}

float3 rotateX(float3 p, float angle) {
    const float c = cos(angle);
    const float s = sin(angle);
    return float3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

float2 scene(float3 p, float time) {
    const float floor = p.y + 1.0;
    const float3 orbit = float3(sin(time * 0.9) * 1.4, 0.2 + 0.4 * sin(time * 1.7), cos(time * 0.9) * 1.4);
    const float blob = smoothMin(sphere(p - orbit, 0.45), sphere(p - float3(0.0, 0.1, 0.0), 0.6), 0.5);
    const float ring = torus(rotateX(rotateY(p, time * 0.5), time * 0.7), float2(1.1, 0.12));
    const float box = roundBox(rotateY(p - float3(0.0, -0.75, 0.0), time * 0.3), float3(1.4, 0.05, 1.4), 0.05);
    const float shapes = smoothMin(smoothMin(blob, ring, 0.25), box, 0.2);
    return shapes < floor ? float2(shapes, 1.0) : float2(floor, 0.0);
}

float3 normalAt(float3 p, float time) {
    const float2 e = float2(0.001, -0.001);
    return normalize(e.xyy * scene(p + e.xyy, time).x + e.yyx * scene(p + e.yyx, time).x + e.yxy * scene(p + e.yxy, time).x
                     + e.xxx * scene(p + e.xxx, time).x);
}

float softShadow(float3 origin, float3 direction, float time) {
    float light = 1.0;
    float t = 0.02;
    for (int step = 0; step < 48 && t < 8.0; ++step) {
        const float distance = scene(origin + direction * t, time).x;
        if (distance < 0.0005)
            return 0.0;
        light = min(light, 10.0 * distance / t);
        t += clamp(distance, 0.01, 0.3);
    }
    return clamp(light, 0.0, 1.0);
}

float ambientOcclusion(float3 p, float3 normal, float time) {
    float occlusion = 0.0;
    float weight = 1.0;
    for (int step = 1; step <= 5; ++step) {
        const float offset = 0.03 * float(step * step);
        occlusion += weight * (offset - scene(p + normal * offset, time).x);
        weight *= 0.7;
    }
    return clamp(1.0 - 1.5 * occlusion, 0.0, 1.0);
}

float3 shade(float3 origin, float3 direction, float time) {
    const float3 sky = lerp(float3(0.75, 0.82, 0.95), float3(0.25, 0.4, 0.75), clamp(direction.y * 1.5, 0.0, 1.0));
    float t = 0.0;
    float2 hit = float2(0.0, -1.0);
    for (int step = 0; step < 128 && t < 30.0; ++step) {
        hit = scene(origin + direction * t, time);
        if (hit.x < 0.0005 * t)
            break;
        t += hit.x;
    }
    if (t >= 30.0)
        return sky;

    const float3 p = origin + direction * t;
    const float3 normal = normalAt(p, time);
    const float3 sun = normalize(float3(0.6, 0.8, -0.4));
    const float checker = fmod(floor(p.x * 2.0) + floor(p.z * 2.0), 2.0) == 0.0 ? 0.35 : 0.25;
    const float3 albedo = hit.y > 0.5 ? 0.5 + 0.5 * cos(float3(0.0, 2.0, 4.0) + p.y * 2.0 + time * 0.4) : float3(checker);
    const float diffuse = max(dot(normal, sun), 0.0) * softShadow(p + normal * 0.002, sun, time);
    const float occlusion = ambientOcclusion(p, normal, time);
    const float specular = pow(max(dot(reflect(direction, normal), sun), 0.0), 32.0) * diffuse;
    const float3 color = albedo * (float3(1.3, 1.15, 0.95) * diffuse + sky * 0.35 * occlusion) + specular * 0.6;
    return lerp(color, sky, 1.0 - exp(-0.002 * t * t));
}

[shader("fragment")]
float4 fragMain(Varyings input) : SV_Target0 {
    const Sdf::Arguments arguments = rootArguments<Sdf::Arguments>();
    const float2 uv = (input.position.xy * 2.0 - arguments.resolution) / arguments.resolution.y * float2(1.0, -1.0);
    const float angle = arguments.time * 0.2;
    const float3 origin = float3(4.5 * sin(angle), 1.6, 4.5 * cos(angle));
    const float3 forward = normalize(-origin);
    const float3 right = normalize(cross(forward, float3(0.0, 1.0, 0.0)));
    const float3 up = cross(right, forward);
    const float3 direction = normalize(forward * 1.8 + right * uv.x + up * uv.y);
    const float3 color = shade(origin, direction, arguments.time);
    return float4(pow(color / (1.0 + color), 1.0 / 2.2), 1.0);
}
