#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Scene {

using namespace NoorRHI;

struct Instance {
    uint2 cell;
    uint material;
    uint visible;
};

struct Material {
    TextureHandle texture;
    uint padding;
};

struct DrawArguments {
    Ptr<float2> corners;
    Ptr<Instance> instances;
    Ptr<Material> materials;
    SamplerHandle sampler;
    float cellSize;
};

struct CullArguments {
    Ptr<Instance> instances;
    Ptr<DrawArgs> draws;
    Ptr<DrawIndexedArgs> indexedDraws;
    AtomicPtr<uint> count;
    uint instanceCount;
    uint indexed;
};

}
