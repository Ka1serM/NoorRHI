#include "Bench.h"

struct TinyVertex {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
TinyVertex tinyVertexMain(uint vertex : SV_VulkanVertexID) {
    const Bench::DrawArguments arguments = rootArguments<Bench::DrawArguments>();
    const uint cell = arguments.index % 10000u;
    const float2 origin = float2(float(cell % 100u), float(cell / 100u)) * 0.02f - 1.0f;
    const float2 corner = float2(vertex == 1u ? 0.02f : 0.0f, vertex == 2u ? 0.02f : 0.0f);
    TinyVertex output;
    output.position = float4(origin + corner, 0.0f, 1.0f);
    output.color = arguments.color;
    return output;
}

[shader("fragment")]
float4 tinyFragmentMain(TinyVertex input) : SV_Target {
    return input.color;
}
