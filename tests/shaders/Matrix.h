#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Matrix {

using namespace NoorRHI;

struct Arguments {
    Ptr<half> a;
    Ptr<half> b;
    Ptr<float> bias;
    Ptr<float> result;
};

struct TiledArguments {
    Ptr<half> a;
    Ptr<half> b;
    Ptr<float> result;
    uint size;
};

struct FloatArguments {
    Ptr<float> a;
    Ptr<float> b;
    Ptr<float> bias;
    Ptr<float> result;
};

}
