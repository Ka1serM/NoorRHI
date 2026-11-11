#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Network {

using namespace NoorRHI;

static const uint inputs = 4;
static const uint hidden = 8;
static const uint parameterCount = hidden * inputs + hidden + hidden + 1;

struct Arguments {
    Ptr<float4> samples;
    Ptr<float> targets;
    Ptr<float> parameters;
    AtomicPtr<float> gradients;
    uint count;
};

}
