#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace RayTracing {

using namespace NoorRHI;

struct Arguments {
    Ptr<uint> result;
    AccelerationStructureHandle scene;
    uint width;
};

struct FilterArguments {
    Ptr<uint> result;
    AccelerationStructureHandle scene;
    uint mode;
    uint flags;
    uint mask;
};

}
