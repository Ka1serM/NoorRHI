#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Count {

using namespace NoorRHI;

struct Arguments {
    Ptr<uint> values;
    Ptr<uint> slots;
    AtomicPtr<uint> counters;
    uint count;
    uint threshold;
};

}
