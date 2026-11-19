#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Bench {

using namespace NoorRHI;

struct StreamArguments {
    Ptr<float4> source;
    Ptr<float4> destination;
    uint count;
};

struct GemmArguments {
    Ptr<half> a;
    Ptr<half> b;
    Ptr<float> c;
    uint size;
};

struct DrawArguments {
    float4 color;
    uint index;
};

struct TraceArguments {
    Ptr<float> result;
    AccelerationStructureHandle scene;
    float4 eye;
    float4 forward;
    float4 originLow;
    float4 originHigh;
    uint width;
    uint height;
    uint incoherent;
    uint padding;
};

}
