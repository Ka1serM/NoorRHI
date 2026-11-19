#include "RayTracingBuild.h"

static const uint groupSize = 128;
static const uint collapseGroupSize = 64;
static const uint wideWidth = 8;
static const uint wideLeafSize = 3;
static const uint wideNodeWords = 6;
static const uint emptyReference = 0xffffffffu;
static const uint noDigit = 16;
static const float emptyLow = 1e30;
static const float emptyHigh = -1e30;

groupshared float3 reduceLow[groupSize];
groupshared float3 reduceHigh[groupSize];
groupshared uint4 rankScan[groupSize];
groupshared uint sumScan[groupSize];
groupshared float3 clusterLow[groupSize + 2 * RayTracingBuild::clusterRadius];
groupshared float3 clusterHigh[groupSize + 2 * RayTracingBuild::clusterRadius];
groupshared float3 finishLow[RayTracingBuild::finishSize];
groupshared float3 finishHigh[RayTracingBuild::finishSize];
groupshared uint finishNodes[RayTracingBuild::finishSize];
groupshared uint finishNeighbors[RayTracingBuild::finishSize];
groupshared uint uniformValue;

uint orderFloat(float value) {
    const uint bits = asuint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

float unorderFloat(uint bits) {
    return asfloat((bits & 0x80000000u) != 0 ? bits & 0x7fffffffu : ~bits);
}

bool isEmpty(float3 low, float3 high) {
    return any(low > high);
}

void sanitize(inout float3 low, inout float3 high) {
    if (all(abs(low) < 1e30) && all(abs(high) < 1e30) && !isEmpty(low, high))
        return;
    low = float3(emptyLow);
    high = float3(emptyHigh);
}

void storeBox(Ptr<float4> boxes, uint index, float3 low, float3 high) {
    boxes[index * 2] = float4(low, 0.0);
    boxes[index * 2 + 1] = float4(high, 0.0);
}

void reduceCenters(uint thread, float3 low, float3 high, AtomicPtr<uint> bounds) {
    const bool empty = isEmpty(low, high);
    const float3 center = (low + high) * 0.5;
    reduceLow[thread] = empty ? float3(emptyLow) : center;
    reduceHigh[thread] = empty ? float3(emptyHigh) : center;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = groupSize / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            reduceLow[thread] = min(reduceLow[thread], reduceLow[thread + stride]);
            reduceHigh[thread] = max(reduceHigh[thread], reduceHigh[thread + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread != 0 || isEmpty(reduceLow[0], reduceHigh[0]))
        return;
    atomicMin(bounds, orderFloat(reduceLow[0].x));
    atomicMin(bounds + 1, orderFloat(reduceLow[0].y));
    atomicMin(bounds + 2, orderFloat(reduceLow[0].z));
    atomicMax(bounds + 3, orderFloat(reduceHigh[0].x));
    atomicMax(bounds + 4, orderFloat(reduceHigh[0].y));
    atomicMax(bounds + 5, orderFloat(reduceHigh[0].z));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void resetBounds() {
    const RayTracingBuild::BuildArguments arguments = rootArguments<RayTracingBuild::BuildArguments>();
    for (uint axis = 0; axis < 3; ++axis) {
        atomicStore(arguments.bounds + axis, orderFloat(emptyLow));
        atomicStore(arguments.bounds + 3 + axis, orderFloat(emptyHigh));
    }
}

[shader("compute")]
[numthreads(groupSize, 1, 1)]
void gatherTriangles(uint3 id : SV_DispatchThreadID, uint thread : SV_GroupIndex) {
    const RayTracingBuild::GatherArguments arguments = rootArguments<RayTracingBuild::GatherArguments>();
    float3 low = float3(emptyLow);
    float3 high = float3(emptyHigh);
    if (id.x < arguments.count) {
        float3 corners[3];
        for (uint corner = 0; corner < 3; ++corner) {
            const uint vertex = arguments.indices[id.x * 3 + corner] * arguments.stride;
            corners[corner] = float3(arguments.positions[vertex], arguments.positions[vertex + 1], arguments.positions[vertex + 2]);
        }
        const uint primitive = arguments.first + id.x;
        const Ptr<uint4> output = arguments.triangles + primitive * 3;
        output[0] = uint4(asuint(corners[0]), id.x);
        output[1] = uint4(asuint(corners[1] - corners[0]), arguments.geometryIndex);
        output[2] = uint4(asuint(corners[2] - corners[0]), arguments.opaque);
        low = min(corners[0], min(corners[1], corners[2]));
        high = max(corners[0], max(corners[1], corners[2]));
        sanitize(low, high);
        storeBox(arguments.boxes, primitive, low, high);
    }
    reduceCenters(thread, low, high, arguments.bounds);
}

[shader("compute")]
[numthreads(groupSize, 1, 1)]
void prepareInstances(uint3 id : SV_DispatchThreadID, uint thread : SV_GroupIndex) {
    const RayTracingBuild::BuildArguments arguments = rootArguments<RayTracingBuild::BuildArguments>();
    float3 low = float3(emptyLow);
    float3 high = float3(emptyHigh);
    if (id.x < arguments.count) {
        const Ptr<uint4> record = arguments.source + id.x * 4;
        const uint4 rows[3] = { record[0], record[1], record[2] };
        const uint4 info = record[3];
        const float3x3 linear = float3x3(asfloat(rows[0].xyz), asfloat(rows[1].xyz), asfloat(rows[2].xyz));
        const float3 translation = float3(asfloat(rows[0].w), asfloat(rows[1].w), asfloat(rows[2].w));
        const float3 cofactor0 = cross(linear[1], linear[2]);
        const float3 cofactor1 = cross(linear[2], linear[0]);
        const float3 cofactor2 = cross(linear[0], linear[1]);
        const float determinant = dot(linear[0], cofactor0);
        const bool valid = determinant != 0.0 && (info.z | info.w) != 0;
        const float3x3 inverse = valid ? transpose(float3x3(cofactor0, cofactor1, cofactor2)) / determinant : float3x3(0.0);
        const float3 inverseTranslation = -mul(inverse, translation);

        const Ptr<uint4> instance = arguments.storage + arguments.primitiveOffset + id.x * 7;
        instance[0] = asuint(float4(inverse[0], inverseTranslation.x));
        instance[1] = asuint(float4(inverse[1], inverseTranslation.y));
        instance[2] = asuint(float4(inverse[2], inverseTranslation.z));
        instance[3] = rows[0];
        instance[4] = rows[1];
        instance[5] = rows[2];
        instance[6] = uint4(info.z, info.w, info.x, info.y);

        if (valid) {
            Ptr<float4> header;
            header.low = info.z;
            header.high = info.w;
            const float3 boxLow = header[1].xyz;
            const float3 boxHigh = header[2].xyz;
            if (!isEmpty(boxLow, boxHigh)) {
                const float3 center = mul(linear, (boxLow + boxHigh) * 0.5) + translation;
                const float3 extent = mul(abs(linear), (boxHigh - boxLow) * 0.5);
                low = center - extent;
                high = center + extent;
                sanitize(low, high);
            }
        }
        storeBox(arguments.leafBoxes, id.x, low, high);
    }
    reduceCenters(thread, low, high, arguments.bounds);
}

uint expandBits(uint value) {
    value = (value * 0x00010001u) & 0xff0000ffu;
    value = (value * 0x00000101u) & 0x0f00f00fu;
    value = (value * 0x00000011u) & 0xc30c30c3u;
    value = (value * 0x00000005u) & 0x49249249u;
    return value;
}

[shader("compute")]
[numthreads(groupSize, 1, 1)]
void computeMortonCodes(uint3 id : SV_DispatchThreadID) {
    const RayTracingBuild::BuildArguments arguments = rootArguments<RayTracingBuild::BuildArguments>();
    if (id.x >= arguments.count)
        return;
    const float3 low = arguments.leafBoxes[id.x * 2].xyz;
    const float3 high = arguments.leafBoxes[id.x * 2 + 1].xyz;
    const float3 sceneLow = float3(unorderFloat(atomicLoad(arguments.bounds)), unorderFloat(atomicLoad(arguments.bounds + 1)),
                                   unorderFloat(atomicLoad(arguments.bounds + 2)));
    const float3 sceneHigh = float3(unorderFloat(atomicLoad(arguments.bounds + 3)), unorderFloat(atomicLoad(arguments.bounds + 4)),
                                    unorderFloat(atomicLoad(arguments.bounds + 5)));
    float3 position = float3(0.0);
    if (!isEmpty(low, high) && !isEmpty(sceneLow, sceneHigh))
        position = clamp(((low + high) * 0.5 - sceneLow) / max(sceneHigh - sceneLow, float3(1e-30)), 0.0, 1.0);
    const uint3 cell = min(uint3(position * 1024.0), uint3(1023));
    arguments.keys[id.x] = expandBits(cell.x) << 2 | expandBits(cell.y) << 1 | expandBits(cell.z);
    arguments.values[id.x] = id.x;
}

uint scanGroup(uint thread, uint value) {
    sumScan[thread] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1; offset < groupSize; offset <<= 1) {
        const uint previous = thread >= offset ? sumScan[thread - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        sumScan[thread] += previous;
        GroupMemoryBarrierWithGroupSync();
    }
    return sumScan[thread];
}

uint scanSegments(uint thread, Ptr<uint> values, uint total) {
    const uint segment = (total + groupSize - 1) / groupSize;
    const uint begin = min(thread * segment, total);
    const uint end = min(begin + segment, total);
    uint sum = 0;
    for (uint index = begin; index < end; ++index)
        sum += values[index];
    uint running = scanGroup(thread, sum) - sum;
    for (uint index = begin; index < end; ++index) {
        const uint count = values[index];
        values[index] = running;
        running += count;
    }
    return sumScan[groupSize - 1];
}

uint4 getDigitMask(uint digit) {
    uint4 mask = uint4(0);
    if (digit < noDigit)
        mask[digit >> 2] = 1u << ((digit & 3) * 8);
    return mask;
}

uint getDigitCount(uint4 counts, uint digit) {
    return (counts[digit >> 2] >> ((digit & 3) * 8)) & 0xffu;
}

uint4 scanDigits(uint thread, uint digit) {
    rankScan[thread] = getDigitMask(digit);
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1; offset < groupSize; offset <<= 1) {
        const uint4 previous = thread >= offset ? rankScan[thread - offset] : uint4(0);
        GroupMemoryBarrierWithGroupSync();
        rankScan[thread] += previous;
        GroupMemoryBarrierWithGroupSync();
    }
    return rankScan[thread];
}

uint readDigit(RayTracingBuild::BuildArguments arguments, uint index) {
    return index < arguments.count ? (arguments.keys[index] >> arguments.shift) & 15u : noDigit;
}

[shader("compute")]
[numthreads(groupSize, 1, 1)]
void countDigits(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const RayTracingBuild::BuildArguments arguments = rootArguments<RayTracingBuild::BuildArguments>();
    scanDigits(thread, readDigit(arguments, group.x * groupSize + thread));
    if (thread < noDigit)
        arguments.histogram[thread * arguments.blockCount + group.x] = getDigitCount(rankScan[groupSize - 1], thread);
}

[shader("compute")]
[numthreads(groupSize, 1, 1)]
void scanDigitCounts(uint thread : SV_GroupIndex) {
    const RayTracingBuild::BuildArguments arguments = rootArguments<RayTracingBuild::BuildArguments>();
    const uint total = noDigit * arguments.blockCount;
    scanSegments(thread, arguments.histogram, total);
}

[shader("compute")]
[numthreads(groupSize, 1, 1)]
void scatterDigits(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const RayTracingBuild::BuildArguments arguments = rootArguments<RayTracingBuild::BuildArguments>();
    const uint index = group.x * groupSize + thread;
    const uint digit = readDigit(arguments, index);
    const uint4 ranks = scanDigits(thread, digit);
    if (digit == noDigit)
        return;
    const uint target = arguments.histogram[digit * arguments.blockCount + group.x] + getDigitCount(ranks, digit) - 1;
    arguments.sortedKeys[target] = arguments.keys[index];
    arguments.sortedValues[target] = arguments.values[index];
}

[shader("compute")]
[numthreads(groupSize, 1, 1)]
void placeLeaves(uint3 id : SV_DispatchThreadID) {
    const RayTracingBuild::BuildArguments arguments = rootArguments<RayTracingBuild::BuildArguments>();
    if (id.x >= arguments.leafBase)
        return;
    const uint node = arguments.leafBase + id.x;
    if (id.x >= arguments.count) {
        storeBox(arguments.tree, node, float3(emptyLow), float3(emptyHigh));
        return;
    }
    const uint primitive = arguments.values[id.x];
    arguments.tree[node * 2] = arguments.leafBoxes[primitive * 2];
    arguments.tree[node * 2 + 1] = arguments.leafBoxes[primitive * 2 + 1];
}

[shader("compute")]
[numthreads(groupSize, 1, 1)]
void reduceTreeLevel(uint3 id : SV_DispatchThreadID) {
    const RayTracingBuild::BuildArguments arguments = rootArguments<RayTracingBuild::BuildArguments>();
    if (id.x >= arguments.level)
        return;
    const uint node = arguments.level + id.x;
    arguments.tree[node * 2] = min(arguments.tree[node * 4], arguments.tree[node * 4 + 2]);
    arguments.tree[node * 2 + 1] = max(arguments.tree[node * 4 + 1], arguments.tree[node * 4 + 3]);
}

uint countLeadingZeros(uint value) {
    return value == 0 ? 32u : 31u - firstbithigh(value);
}

int getCommonPrefix(RayTracingBuild::BuildArguments arguments, int first, int second) {
    if (second < 0 || second >= int(arguments.count))
        return -1;
    const uint firstKey = arguments.keys[first];
    const uint secondKey = arguments.keys[second];
    if (firstKey == secondKey)
        return 32 + int(countLeadingZeros(uint(first ^ second)));
    return int(countLeadingZeros(firstKey ^ secondKey));
}

void queryRange(RayTracingBuild::BuildArguments arguments, uint first, uint last, out float3 low, out float3 high) {
    low = float3(emptyLow);
    high = float3(emptyHigh);
    uint left = first + arguments.leafBase;
    uint right = last + arguments.leafBase + 1;
    while (left < right) {
        if ((left & 1) != 0) {
            low = min(low, arguments.tree[left * 2].xyz);
            high = max(high, arguments.tree[left * 2 + 1].xyz);
            ++left;
        }
        if ((right & 1) != 0) {
            --right;
            low = min(low, arguments.tree[right * 2].xyz);
            high = max(high, arguments.tree[right * 2 + 1].xyz);
        }
        left >>= 1;
        right >>= 1;
    }
}

void writeChild(RayTracingBuild::BuildArguments arguments, uint node, uint side, float3 low, float3 high, uint reference, uint count) {
    const Ptr<uint4> child = arguments.storage + 4 + node * 4 + side * 2;
    child[0] = uint4(asuint(low), reference);
    child[1] = uint4(asuint(high), count);
}

void writeChildRange(RayTracingBuild::BuildArguments arguments, uint node, uint side, uint first, uint last, uint inner) {
    float3 low;
    float3 high;
    queryRange(arguments, first, last, low, high);
    if (first != last) {
        writeChild(arguments, node, side, low, high, inner, 0);
        return;
    }
    writeChild(arguments, node, side, low, high, arguments.values[first], 1);
}

[shader("compute")]
[numthreads(groupSize, 1, 1)]
void writeNodes(uint3 id : SV_DispatchThreadID) {
    const RayTracingBuild::BuildArguments arguments = rootArguments<RayTracingBuild::BuildArguments>();
    if (id.x == 0) {
        arguments.storage[0] = uint4(max(arguments.count, 2u) - 1, arguments.count, arguments.primitiveOffset * 16, 0);
        arguments.storage[1] = asuint(float4(arguments.tree[2].xyz, 0.0));
        arguments.storage[2] = asuint(float4(arguments.tree[3].xyz, 0.0));
        arguments.storage[3] = uint4(0);
    }
    if (arguments.count == 1) {
        if (id.x != 0)
            return;
        writeChildRange(arguments, 0, 0, 0, 0, 0);
        writeChild(arguments, 0, 1, float3(emptyLow), float3(emptyHigh), emptyReference, 0);
        return;
    }
    if (id.x + 1 >= arguments.count)
        return;
    const int node = int(id.x);
    const int direction = getCommonPrefix(arguments, node, node + 1) > getCommonPrefix(arguments, node, node - 1) ? 1 : -1;
    const int minimum = getCommonPrefix(arguments, node, node - direction);
    int maximumLength = 2;
    while (getCommonPrefix(arguments, node, node + maximumLength * direction) > minimum)
        maximumLength <<= 1;
    int length = 0;
    for (int step = maximumLength >> 1; step > 0; step >>= 1)
        if (getCommonPrefix(arguments, node, node + (length + step) * direction) > minimum)
            length += step;
    const int other = node + length * direction;
    const int prefix = getCommonPrefix(arguments, node, other);
    int split = 0;
    int divisor = 2;
    int step = (length + 1) >> 1;
    for (;;) {
        if (getCommonPrefix(arguments, node, node + (split + step) * direction) > prefix)
            split += step;
        if (step == 1)
            break;
        divisor <<= 1;
        step = (length + divisor - 1) / divisor;
    }
    const int gamma = node + split * direction + min(direction, 0);
    const uint first = uint(min(node, other));
    const uint last = uint(max(node, other));
    writeChildRange(arguments, uint(node), 0, first, uint(gamma), uint(gamma));
    writeChildRange(arguments, uint(node), 1, uint(gamma) + 1, last, uint(gamma) + 1);
}

struct WideChild {
    float3 low;
    float3 high;
    uint node;
    uint primitives;
}

float getHalfArea(float3 low, float3 high) {
    const float3 extent = max(high - low, float3(0.0));
    return extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
}

void loadNodeBox(RayTracingBuild::ClusterArguments arguments, uint node, out float3 low, out float3 high) {
    low = asfloat(arguments.nodes[node * 2].xyz);
    high = asfloat(arguments.nodes[node * 2 + 1].xyz);
}

uint getClusterCount(RayTracingBuild::ClusterArguments arguments) {
    return arguments.state[RayTracingBuild::stateCounts + (arguments.round & 1)];
}

//! WGSL accepts a barrier only in uniform control flow, and values from read-write buffers and groupshared memory are not uniform there.
uint shareUniform(uint thread, uint value) {
    if (thread == 0)
        uniformValue = value;
    GroupMemoryBarrierWithGroupSync();
    return workgroupUniformLoad(uniformValue);
}

Ptr<uint> getClusters(RayTracingBuild::ClusterArguments arguments, uint parity) {
    return arguments.clusters + parity * arguments.count;
}

void writeRoundSize(RayTracingBuild::ClusterArguments arguments, uint parity, uint clusters) {
    const uint offset = RayTracingBuild::stateRoundSizes + parity * 4;
    arguments.state[RayTracingBuild::stateCounts + parity] = clusters;
    arguments.state[offset] = clusters > RayTracingBuild::finishSize ? (clusters + groupSize - 1) / groupSize : 0;
    arguments.state[offset + 1] = 1;
    arguments.state[offset + 2] = 1;
}

[shader("compute")]
[numthreads(groupSize, 1, 1)]
void beginClusters(uint3 id : SV_DispatchThreadID) {
    const RayTracingBuild::ClusterArguments arguments = rootArguments<RayTracingBuild::ClusterArguments>();
    if (id.x == 0) {
        writeRoundSize(arguments, 0, arguments.count);
        atomicStore(arguments.counters + RayTracingBuild::nodeCounter, arguments.count);
    }
    if (id.x >= arguments.count)
        return;
    const uint primitive = arguments.values[id.x];
    arguments.nodes[id.x * 2] = uint4(asuint(arguments.leafBoxes[primitive * 2].xyz), primitive);
    arguments.nodes[id.x * 2 + 1] = uint4(asuint(arguments.leafBoxes[primitive * 2 + 1].xyz), emptyReference);
    arguments.nodeCounts[id.x] = 1;
    arguments.clusters[id.x] = id.x;
}

//! PLOC (Meister and Bittner 2018): each cluster picks the neighbor within the radius of the Morton order whose union box has the smallest area.
[shader("compute")]
[numthreads(groupSize, 1, 1)]
void findNeighbors(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const RayTracingBuild::ClusterArguments arguments = rootArguments<RayTracingBuild::ClusterArguments>();
    const uint clusters = getClusterCount(arguments);
    const Ptr<uint> current = getClusters(arguments, arguments.round & 1);
    const int first = int(group.x * groupSize) - int(RayTracingBuild::clusterRadius);
    for (uint entry = thread; entry < groupSize + 2 * RayTracingBuild::clusterRadius; entry += groupSize) {
        const int index = first + int(entry);
        float3 low = float3(emptyLow);
        float3 high = float3(emptyHigh);
        if (index >= 0 && index < int(clusters))
            loadNodeBox(arguments, current[index], low, high);
        clusterLow[entry] = low;
        clusterHigh[entry] = high;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint index = group.x * groupSize + thread;
    if (index >= clusters)
        return;
    if (arguments.pairing != 0) {
        arguments.neighbors[index] = (index ^ 1) < clusters ? index ^ 1 : emptyReference;
        return;
    }
    const float3 low = clusterLow[thread + RayTracingBuild::clusterRadius];
    const float3 high = clusterHigh[thread + RayTracingBuild::clusterRadius];
    float bestArea = 0.0;
    uint best = emptyReference;
    for (uint offset = 0; offset <= 2 * RayTracingBuild::clusterRadius; ++offset) {
        const int other = int(index + offset) - int(RayTracingBuild::clusterRadius);
        if (offset == RayTracingBuild::clusterRadius || other < 0 || other >= int(clusters))
            continue;
        const float area = getHalfArea(min(low, clusterLow[thread + offset]), max(high, clusterHigh[thread + offset]));
        if (best != emptyReference && area >= bestArea)
            continue;
        bestArea = area;
        best = uint(other);
    }
    arguments.neighbors[index] = best;
}

uint writeMergedNode(RayTracingBuild::ClusterArguments arguments, uint first, uint second, float3 low, float3 high) {
    const uint node = atomicAdd(arguments.counters + RayTracingBuild::nodeCounter, 1u);
    arguments.nodes[node * 2] = uint4(asuint(low), first);
    arguments.nodes[node * 2 + 1] = uint4(asuint(high), second);
    arguments.nodeCounts[node] = arguments.nodeCounts[first] + arguments.nodeCounts[second];
    return node;
}

uint mergeCluster(RayTracingBuild::ClusterArguments arguments, Ptr<uint> current, uint index) {
    const uint neighbor = arguments.neighbors[index];
    const uint cluster = current[index];
    if (neighbor == emptyReference || arguments.neighbors[neighbor] != index)
        return cluster;
    if (neighbor < index)
        return emptyReference;
    const uint other = current[neighbor];
    float3 low;
    float3 high;
    float3 otherLow;
    float3 otherHigh;
    loadNodeBox(arguments, cluster, low, high);
    loadNodeBox(arguments, other, otherLow, otherHigh);
    return writeMergedNode(arguments, cluster, other, min(low, otherLow), max(high, otherHigh));
}

[shader("compute")]
[numthreads(groupSize, 1, 1)]
void mergeClusters(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const RayTracingBuild::ClusterArguments arguments = rootArguments<RayTracingBuild::ClusterArguments>();
    const uint clusters = getClusterCount(arguments);
    const uint index = group.x * groupSize + thread;
    const uint value = index < clusters ? mergeCluster(arguments, getClusters(arguments, arguments.round & 1), index) : emptyReference;
    const uint kept = scanGroup(thread, value != emptyReference ? 1u : 0u);
    if (index < clusters)
        arguments.slots[index] = value;
    if (thread == groupSize - 1)
        arguments.blockSums[group.x] = kept;
}

[shader("compute")]
[numthreads(groupSize, 1, 1)]
void scanClusters(uint thread : SV_GroupIndex) {
    const RayTracingBuild::ClusterArguments arguments = rootArguments<RayTracingBuild::ClusterArguments>();
    const uint next = (arguments.round & 1) ^ 1;
    const uint clusters = shareUniform(thread, getClusterCount(arguments));
    if (clusters <= RayTracingBuild::finishSize) {
        for (uint index = thread; index < clusters; index += groupSize)
            getClusters(arguments, next)[index] = getClusters(arguments, next ^ 1)[index];
        if (thread == 0)
            writeRoundSize(arguments, next, clusters);
        return;
    }
    const uint total = scanSegments(thread, arguments.blockSums, (clusters + groupSize - 1) / groupSize);
    if (thread == 0)
        writeRoundSize(arguments, next, total);
}

[shader("compute")]
[numthreads(groupSize, 1, 1)]
void compactClusters(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const RayTracingBuild::ClusterArguments arguments = rootArguments<RayTracingBuild::ClusterArguments>();
    const uint next = (arguments.round & 1) ^ 1;
    const uint clusters = getClusterCount(arguments);
    const uint index = group.x * groupSize + thread;
    const uint value = index < clusters ? arguments.slots[index] : emptyReference;
    const uint kept = value != emptyReference ? 1u : 0u;
    const uint offset = scanGroup(thread, kept) - kept;
    if (kept == 0)
        return;
    getClusters(arguments, next)[arguments.blockSums[group.x] + offset] = value;
}

uint findFinishNeighbor(uint slot, uint clusters, bool pairing) {
    if (pairing)
        return (slot ^ 1) < clusters ? slot ^ 1 : emptyReference;
    const float3 low = finishLow[slot];
    const float3 high = finishHigh[slot];
    float bestArea = 0.0;
    uint best = emptyReference;
    for (uint other = 0; other < clusters; ++other) {
        if (other == slot)
            continue;
        const float area = getHalfArea(min(low, finishLow[other]), max(high, finishHigh[other]));
        if (best != emptyReference && area >= bestArea)
            continue;
        bestArea = area;
        best = other;
    }
    return best;
}

//! One group clusters the last finishSize clusters against all others, so the top of the tree follows the geometry and not the Morton order.
[shader("compute")]
[numthreads(groupSize, 1, 1)]
void finishClusters(uint thread : SV_GroupIndex) {
    const RayTracingBuild::ClusterArguments arguments = rootArguments<RayTracingBuild::ClusterArguments>();
    const uint clusters = shareUniform(thread, getClusterCount(arguments));
    const Ptr<uint> current = getClusters(arguments, arguments.round & 1);
    for (uint slot = thread; slot < clusters; slot += groupSize) {
        float3 low;
        float3 high;
        finishNodes[slot] = current[slot];
        loadNodeBox(arguments, current[slot], low, high);
        finishLow[slot] = low;
        finishHigh[slot] = high;
    }
    GroupMemoryBarrierWithGroupSync();
    uint count = clusters;
    for (uint iteration = 0; count > 1; ++iteration) {
        const bool pairing = iteration >= RayTracingBuild::finishIterations;
        uint neighbors[2];
        for (uint part = 0; part < 2; ++part) {
            const uint slot = thread + part * groupSize;
            neighbors[part] = slot < count ? findFinishNeighbor(slot, count, pairing) : emptyReference;
        }
        for (uint part = 0; part < 2; ++part)
            finishNeighbors[thread + part * groupSize] = neighbors[part];
        GroupMemoryBarrierWithGroupSync();
        uint nodes[2];
        float3 lows[2];
        float3 highs[2];
        uint kept = 0;
        for (uint part = 0; part < 2; ++part) {
            const uint slot = thread + part * groupSize;
            const uint neighbor = neighbors[part];
            if (slot >= count || (neighbor != emptyReference && finishNeighbors[neighbor] == slot && neighbor < slot))
                continue;
            nodes[kept] = finishNodes[slot];
            lows[kept] = finishLow[slot];
            highs[kept] = finishHigh[slot];
            if (neighbor != emptyReference && finishNeighbors[neighbor] == slot) {
                lows[kept] = min(lows[kept], finishLow[neighbor]);
                highs[kept] = max(highs[kept], finishHigh[neighbor]);
                nodes[kept] = writeMergedNode(arguments, nodes[kept], finishNodes[neighbor], lows[kept], highs[kept]);
            }
            ++kept;
        }
        const uint offset = scanGroup(thread, kept) - kept;
        count = shareUniform(thread, sumScan[groupSize - 1]);
        for (uint part = 0; part < kept; ++part) {
            finishNodes[offset + part] = nodes[part];
            finishLow[offset + part] = lows[part];
            finishHigh[offset + part] = highs[part];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread == 0)
        arguments.state[RayTracingBuild::stateRoot] = finishNodes[0];
}

void appendChild(RayTracingBuild::ClusterArguments arguments, uint node, inout WideChild children[wideWidth], inout uint count) {
    WideChild child;
    child.node = node;
    child.primitives = arguments.nodeCounts[node];
    loadNodeBox(arguments, node, child.low, child.high);
    if (isEmpty(child.low, child.high))
        return;
    children[count++] = child;
}

float3 getSlotDirection(uint slot) {
    return float3((slot & 1) != 0 ? -1.0 : 1.0, (slot & 2) != 0 ? -1.0 : 1.0, (slot & 4) != 0 ? -1.0 : 1.0);
}

//! The product of an 8-bit cell and a power of two is exact, so the traversal decodes the same value with or without a fused multiply-add.
float decodeQuantized(float origin, float scale, uint cell) {
    return origin + float(cell) * scale;
}

float getQuantizationScale(float low, float high) {
    float scale = exp2(clamp(ceil(log2(max((high - low) / 255.0, 1e-37))), -126.0, 127.0));
    while (decodeQuantized(low, scale, 255) < high)
        scale *= 2.0;
    return scale;
}

uint quantizeLow(float value, float origin, float scale) {
    uint cell = uint(clamp(floor((value - origin) / scale), 0.0, 255.0));
    while (cell > 0 && decodeQuantized(origin, scale, cell) > value)
        --cell;
    return cell;
}

uint quantizeHigh(float value, float origin, float scale) {
    uint cell = uint(clamp(ceil((value - origin) / scale), 0.0, 255.0));
    while (cell < 255 && decodeQuantized(origin, scale, cell) < value)
        ++cell;
    return cell;
}

void packByte(inout uint packed[12], uint plane, uint slot, uint value) {
    const uint index = plane * wideWidth + slot;
    packed[index >> 2] |= value << ((index & 3) * 8);
}

void copyTriangle(RayTracingBuild::ClusterArguments arguments, uint primitive, uint slot) {
    const Ptr<uint4> target = arguments.storage + arguments.primitiveOffset + slot * 3;
    for (uint word = 0; word < 3; ++word)
        target[word] = arguments.source[primitive * 3 + word];
}

void writeLeafTriangles(RayTracingBuild::ClusterArguments arguments, uint root, uint first) {
    uint stack[wideLeafSize];
    uint depth = 0;
    uint node = root;
    uint written = 0;
    for (;;) {
        const uint4 low = arguments.nodes[node * 2];
        const uint4 high = arguments.nodes[node * 2 + 1];
        if (high.w != emptyReference) {
            stack[depth++] = high.w;
            node = low.w;
            continue;
        }
        copyTriangle(arguments, low.w, first + written++);
        if (depth == 0)
            return;
        node = stack[--depth];
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void beginCollapse() {
    const RayTracingBuild::ClusterArguments arguments = rootArguments<RayTracingBuild::ClusterArguments>();
    const uint root = arguments.state[RayTracingBuild::stateRoot];
    atomicStore(arguments.counters + RayTracingBuild::wideCounter, 1u);
    atomicStore(arguments.counters + RayTracingBuild::triangleCounter, 0u);
    arguments.pending[0] = root;
    arguments.state[RayTracingBuild::stateWideBegin] = 0;
    arguments.state[RayTracingBuild::stateWideEnd] = 1;
    arguments.state[RayTracingBuild::stateWideSize] = 1;
    arguments.state[RayTracingBuild::stateWideSize + 1] = 1;
    arguments.state[RayTracingBuild::stateWideSize + 2] = 1;
    float3 low;
    float3 high;
    loadNodeBox(arguments, root, low, high);
    arguments.storage[0] = uint4(1, arguments.count, arguments.primitiveOffset * 16, 0);
    arguments.storage[1] = asuint(float4(low, 0.0));
    arguments.storage[2] = asuint(float4(high, 0.0));
    arguments.storage[3] = uint4(0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void prepareCollapse() {
    const RayTracingBuild::ClusterArguments arguments = rootArguments<RayTracingBuild::ClusterArguments>();
    const uint begin = arguments.state[RayTracingBuild::stateWideEnd];
    const uint end = atomicLoad(arguments.counters + RayTracingBuild::wideCounter);
    arguments.state[RayTracingBuild::stateWideBegin] = begin;
    arguments.state[RayTracingBuild::stateWideEnd] = end;
    arguments.state[RayTracingBuild::stateWideSize] = (end - begin + collapseGroupSize - 1) / collapseGroupSize;
    const uint4 header = arguments.storage[0];
    arguments.storage[0] = uint4(end, header.yzw);
}

//! Each round turns the binary subtrees allocated in the previous round into wide nodes, expanding the largest child with more than wideLeafSize primitives first.
[shader("compute")]
[numthreads(collapseGroupSize, 1, 1)]
void collapseNodes(uint3 id : SV_DispatchThreadID) {
    const RayTracingBuild::ClusterArguments arguments = rootArguments<RayTracingBuild::ClusterArguments>();
    const uint wide = arguments.state[RayTracingBuild::stateWideBegin] + id.x;
    if (wide >= arguments.state[RayTracingBuild::stateWideEnd])
        return;
    WideChild children[wideWidth];
    uint count = 0;
    appendChild(arguments, arguments.pending[wide], children, count);
    while (count < wideWidth) {
        int best = -1;
        float bestArea = -1.0;
        for (uint index = 0; index < count; ++index) {
            const float area = getHalfArea(children[index].low, children[index].high);
            if (children[index].primitives > wideLeafSize && area > bestArea) {
                best = int(index);
                bestArea = area;
            }
        }
        if (best < 0)
            break;
        const uint4 split = arguments.nodes[children[best].node * 2];
        const uint other = arguments.nodes[children[best].node * 2 + 1].w;
        children[best] = children[--count];
        appendChild(arguments, split.w, children, count);
        appendChild(arguments, other, children, count);
    }

    float3 nodeLow = float3(emptyLow);
    float3 nodeHigh = float3(emptyHigh);
    uint innerCount = 0;
    uint leafTriangles = 0;
    for (uint index = 0; index < count; ++index) {
        nodeLow = min(nodeLow, children[index].low);
        nodeHigh = max(nodeHigh, children[index].high);
        if (children[index].primitives > wideLeafSize)
            ++innerCount;
        else
            leafTriangles += children[index].primitives;
    }
    if (count == 0) {
        nodeLow = float3(0.0);
        nodeHigh = float3(0.0);
    }
    const float3 center = (nodeLow + nodeHigh) * 0.5;

    uint slotChild[wideWidth];
    for (uint slot = 0; slot < wideWidth; ++slot)
        slotChild[slot] = emptyReference;
    uint assigned = 0;
    for (uint round = 0; round < count; ++round) {
        float bestCost = 1e38;
        uint bestChild = 0;
        uint bestSlot = 0;
        for (uint index = 0; index < count; ++index) {
            if ((assigned & (1u << index)) != 0)
                continue;
            const float3 offset = (children[index].low + children[index].high) * 0.5 - center;
            for (uint slot = 0; slot < wideWidth; ++slot) {
                const float cost = dot(offset, getSlotDirection(slot));
                if (slotChild[slot] != emptyReference || cost >= bestCost)
                    continue;
                bestCost = cost;
                bestChild = index;
                bestSlot = slot;
            }
        }
        slotChild[bestSlot] = bestChild;
        assigned |= 1u << bestChild;
    }

    uint innerTarget = innerCount != 0 ? atomicAdd(arguments.counters + RayTracingBuild::wideCounter, innerCount) : 0;
    uint triangleTarget = leafTriangles != 0 ? atomicAdd(arguments.counters + RayTracingBuild::triangleCounter, leafTriangles) : 0;
    const float3 scale = float3(getQuantizationScale(nodeLow.x, nodeHigh.x), getQuantizationScale(nodeLow.y, nodeHigh.y),
                                getQuantizationScale(nodeLow.z, nodeHigh.z));
    const uint exponents = (asuint(scale.x) >> 23) | (asuint(scale.y) >> 23) << 8 | (asuint(scale.z) >> 23) << 16;
    uint descriptors[wideWidth];
    uint packed[12];
    for (uint index = 0; index < 12; ++index)
        packed[index] = 0;
    for (uint slot = 0; slot < wideWidth; ++slot) {
        descriptors[slot] = emptyReference;
        if (slotChild[slot] == emptyReference)
            continue;
        const WideChild child = children[slotChild[slot]];
        if (child.primitives > wideLeafSize) {
            arguments.pending[innerTarget] = child.node;
            descriptors[slot] = innerTarget++ << 3;
        } else {
            writeLeafTriangles(arguments, child.node, triangleTarget);
            descriptors[slot] = triangleTarget << 3 | child.primitives;
            triangleTarget += child.primitives;
        }
        for (uint axis = 0; axis < 3; ++axis) {
            packByte(packed, axis, slot, quantizeLow(child.low[axis], nodeLow[axis], scale[axis]));
            packByte(packed, 3 + axis, slot, quantizeHigh(child.high[axis], nodeLow[axis], scale[axis]));
        }
    }
    const Ptr<uint4> node = arguments.storage + 4 + wide * wideNodeWords;
    node[0] = uint4(asuint(nodeLow), exponents);
    node[1] = uint4(descriptors[0], descriptors[1], descriptors[2], descriptors[3]);
    node[2] = uint4(descriptors[4], descriptors[5], descriptors[6], descriptors[7]);
    node[3] = uint4(packed[0], packed[1], packed[2], packed[3]);
    node[4] = uint4(packed[4], packed[5], packed[6], packed[7]);
    node[5] = uint4(packed[8], packed[9], packed[10], packed[11]);
}
