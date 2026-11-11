#include "Raster.h"

struct Varyings {
    float4 position : SV_Position;
};

[shader("vertex")]
Varyings vertMain(uint index : SV_VertexID) {
    float2 corner = float2(float((index << 1u) & 2u), float(index & 2u));
    Varyings output;
    output.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

[shader("vertex")]
Varyings cornerMain(uint index : SV_VertexID) {
    const float2 corners[3] = {float2(-1.0, -1.0), float2(0.0, -1.0), float2(-1.0, 0.0)};
    Varyings output;
    output.position = float4(corners[index], 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 colorMain(Varyings input) : SV_Target0 {
    return rootArguments<Raster::Arguments>().color;
}

[shader("fragment")]
float4 textureMain(Varyings input) : SV_Target0 {
    const Raster::Arguments arguments = rootArguments<Raster::Arguments>();
    return arguments.texture.Sample(arguments.sampler, input.position.xy / arguments.targetSize * arguments.uvScale);
}
