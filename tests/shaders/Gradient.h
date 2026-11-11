#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Gradient {

using namespace NoorRHI;

struct Arguments {
    Ptr<float4> gaussians;
    Ptr<float> target;
    AtomicPtr<float> gradients;
    uint width;
    uint height;
    uint count;
};

}
