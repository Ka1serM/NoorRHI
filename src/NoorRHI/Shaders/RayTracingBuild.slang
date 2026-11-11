#include "RayTracingBuild.h"

static const uint groupSize = 128;
static const uint noDigit = 16;
static const float emptyLow = 1e30;
static const float emptyHigh = -1e30;

groupshared float3 reduceLow[groupSize];
groupshared float3 reduceHigh[groupSize];
groupshared uint4 rankScan[groupSize];
groupshared uint sumScan[groupSize];

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
    const uint segment = (total + groupSize - 1) / groupSize;
    const uint begin = min(thread * segment, total);
    const uint end = min(begin + segment, total);
    uint sum = 0;
    for (uint index = begin; index < end; ++index)
        sum += arguments.histogram[index];
    sumScan[thread] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1; offset < groupSize; offset <<= 1) {
        const uint previous = thread >= offset ? sumScan[thread - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        sumScan[thread] += previous;
        GroupMemoryBarrierWithGroupSync();
    }
    uint running = sumScan[thread] - sum;
    for (uint index = begin; index < end; ++index) {
        const uint count = arguments.histogram[index];
        arguments.histogram[index] = running;
        running += count;
    }
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
    if (arguments.topLevel != 0)
        return;
    const Ptr<uint4> target = arguments.storage + arguments.primitiveOffset + id.x * 3;
    for (uint word = 0; word < 3; ++word)
        target[word] = arguments.source[primitive * 3 + word];
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
    writeChild(arguments, node, side, low, high, arguments.topLevel != 0 ? arguments.values[first] : first, 1);
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
        writeChild(arguments, 0, 1, float3(emptyLow), float3(emptyHigh), 0xffffffffu, 0);
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
