#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Raster {

using namespace NoorRHI;

struct Arguments {
    float4 color;
    TextureHandle texture;
    SamplerHandle sampler;
    float2 uvScale;
    float2 targetSize;
};

}
