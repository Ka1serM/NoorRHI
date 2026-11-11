#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace StorageImage {

using namespace NoorRHI;

struct Arguments {
    StorageTextureHandle target;
    TextureHandle source;
    SamplerHandle sampler;
    uint size;
};

}
