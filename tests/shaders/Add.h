#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Add {

using namespace NoorRHI;

struct Arguments {
    Ptr<float> a;
    Ptr<float> b;
    Ptr<float> result;
    uint count;
};

}
