#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace RayTracingBuild {

using namespace NoorRHI;

struct GatherArguments {
    Ptr<float> positions;
    Ptr<uint> indices;
    Ptr<uint4> triangles;
    Ptr<float4> boxes;
    AtomicPtr<uint> bounds;
    uint stride;
    uint count;
    uint first;
    uint geometryIndex;
    uint opaque;
};

struct BuildArguments {
    Ptr<uint4> storage;
    Ptr<uint4> source;
    Ptr<float4> leafBoxes;
    Ptr<float4> tree;
    Ptr<uint> keys;
    Ptr<uint> values;
    Ptr<uint> sortedKeys;
    Ptr<uint> sortedValues;
    Ptr<uint> histogram;
    AtomicPtr<uint> bounds;
    uint count;
    uint leafBase;
    uint primitiveOffset;
    uint topLevel;
    uint shift;
    uint blockCount;
    uint level;
    uint padding;
};

}
