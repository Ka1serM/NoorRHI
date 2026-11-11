#include "Gradient.h"

[Differentiable]
float splat(float4 gaussian, no_diff float2 position) {
    const float2 offset = position - gaussian.xy;
    return gaussian.w * exp(-dot(offset, offset) / (2.0 * gaussian.z * gaussian.z));
}

void accumulate(Gradient::Arguments arguments, uint2 pixel) {
    const float2 position = float2(pixel) + 0.5;
    float value = 0.0;
    for (uint index = 0; index < arguments.count; ++index)
        value += splat(arguments.gaussians[index], position);
    const float residual = 2.0 * (value - arguments.target[pixel.y * arguments.width + pixel.x]);
    for (uint index = 0; index < arguments.count; ++index) {
        var gaussian = diffPair(arguments.gaussians[index]);
        bwd_diff(splat)(gaussian, position, residual);
        const AtomicPtr<float> gradient = arguments.gradients + index * 4;
        atomicAdd(gradient, gaussian.d.x);
        atomicAdd(gradient + 1, gaussian.d.y);
        atomicAdd(gradient + 2, gaussian.d.z);
        atomicAdd(gradient + 3, gaussian.d.w);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 id : SV_DispatchThreadID) {
    const Gradient::Arguments arguments = rootArguments<Gradient::Arguments>();
    if (id.x < arguments.width && id.y < arguments.height)
        accumulate(arguments, id.xy);
}

[shader("vertex")]
float4 vertMain(uint vertexId : SV_VertexID) : SV_Position {
    return float4(float2((vertexId << 1) & 2, vertexId & 2) * 2.0 - 1.0, 0.0, 1.0);
}

[shader("fragment")]
float4 fragMain(float4 position : SV_Position) : SV_Target0 {
    accumulate(rootArguments<Gradient::Arguments>(), uint2(position.xy));
    return float4(0.0);
}
