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
    uint shift;
    uint blockCount;
    uint level;
    uint padding;
};

static const uint clusterRadius = 16;
static const uint clusterRounds = 64;
static const uint pairingRounds = 23;
static const uint finishSize = 256;
static const uint finishIterations = 32;
static const uint finishPairingRounds = 8;
static const uint clusterLevels = clusterRounds + pairingRounds + finishIterations + finishPairingRounds;
static const uint wideCounter = 6;
static const uint nodeCounter = 7;
static const uint triangleCounter = 8;
static const uint stateCounts = 0;
static const uint stateRoot = 2;
static const uint stateRoundSizes = 4;
static const uint stateWideBegin = 12;
static const uint stateWideEnd = 13;
static const uint stateWideSize = 16;
static const uint stateWords = 20;

struct ClusterArguments {
    Ptr<uint4> storage;
    Ptr<uint4> source;
    Ptr<float4> leafBoxes;
    Ptr<uint> values;
    Ptr<uint4> nodes;
    Ptr<uint> nodeCounts;
    Ptr<uint> clusters;
    Ptr<uint> neighbors;
    Ptr<uint> slots;
    Ptr<uint> blockSums;
    Ptr<uint> state;
    Ptr<uint> pending;
    AtomicPtr<uint> counters;
    uint count;
    uint primitiveOffset;
    uint round;
    uint pairing;
};

}
