module NoorRHI;

#if defined(NOORRHI_WEBGPU) && !defined(NOORRHI_RAYTRACING_FALLBACK)
#define NOORRHI_RAYTRACING_FALLBACK
#endif

#if defined(NOORRHI_RAYTRACING_LAUNCH)
static const uint noorrhi_root_offset = 48;
#else
static const uint noorrhi_root_offset = 0;
#endif

#if defined(NOORRHI_WEBGPU)

#include "NoorRHIWebGpu.slang"

T noorrhi_heap_load<T>(uint address) {
    if (T is half)
        return reinterpret<T>(half(f16tof32(noorrhi_heap_load<uint>(address & ~3u) >> (address & 2u) * 8u)));
    T value;
    __stage_switch {
    case compute: value = noorrhi_heap.Load<T>(address);
    case vertex: value = noorrhi_heap_read.Load<T>(address);
    case fragment: value = noorrhi_heap_read.Load<T>(address);
    }
    return value;
}

//! WGSL has no 16-bit stores, so a half store rewrites its whole word, and two invocations must not store the two halves of one word at the same time.
void noorrhi_heap_store<T>(uint address, T value) {
    if (T is half) {
        const uint shift = (address & 2u) * 8u;
        const uint word = noorrhi_heap.Load<uint>(address & ~3u) & ~(0xffffu << shift);
        noorrhi_heap.Store<uint>(address & ~3u, word | f32tof16(float(reinterpret<half>(value))) << shift);
        return;
    }
    noorrhi_heap.Store<T>(address, value);
}

public struct Ptr<T> {
    public uint low;
    public uint high;

    public __init(uint address) { low = address; high = 0; }

    public __subscript(uint index) -> T {
        get { return noorrhi_heap_load<T>(low + index * sizeof(T)); }
        [nonmutating] set { noorrhi_heap_store<T>(low + index * sizeof(T), newValue); }
    }
}

public Ptr<T> operator+<T>(Ptr<T> pointer, uint count) {
    return Ptr<T>(pointer.low + count * sizeof(T));
}
public Ptr<T> operator-<T>(Ptr<T> pointer, uint count) {
    return Ptr<T>(pointer.low - count * sizeof(T));
}
public __prefix T operator*<T>(Ptr<T> pointer) { return pointer[0]; }

public bool isNull<T>(Ptr<T> pointer) { return pointer.low == 0; }

//! The arguments come from a read-only binding, so WGSL treats them as uniform in compute shaders, where the heap is read-write.
T noorrhi_argument_load<T>(uint address) {
    if (T is half)
        return reinterpret<T>(half(f16tof32(noorrhi_arguments.Load<uint>(address & ~3u) >> (address & 2u) * 8u)));
    return noorrhi_arguments.Load<T>(address);
}

public T rootArguments<T>() {
    return noorrhi_argument_load<T>(noorrhi_root_offset);
}

uint4 noorrhi_root_words(uint index) {
    return noorrhi_argument_load<uint4>(index * 16u);
}

public struct AtomicPtr<T> {
    public uint low;
    public uint high;
}

public AtomicPtr<T> operator+<T>(AtomicPtr<T> pointer, uint count) {
    AtomicPtr<T> result = pointer;
    result.low += count * 4u;
    return result;
}

public bool isNull<T>(AtomicPtr<T> pointer) { return (pointer.low | pointer.high) == 0; }

public uint atomicLoad(AtomicPtr<uint> p) { return noorrhi_atomics[p.low >> 2].load(); }
public void atomicStore(AtomicPtr<uint> p, uint v) { noorrhi_atomics[p.low >> 2].store(v); }
public uint atomicAdd(AtomicPtr<uint> p, uint v) { return noorrhi_atomics[p.low >> 2].add(v); }
public uint atomicSub(AtomicPtr<uint> p, uint v) { return noorrhi_atomics[p.low >> 2].sub(v); }
public uint atomicMin(AtomicPtr<uint> p, uint v) { return noorrhi_atomics[p.low >> 2].min(v); }
public uint atomicMax(AtomicPtr<uint> p, uint v) { return noorrhi_atomics[p.low >> 2].max(v); }
public uint atomicAnd(AtomicPtr<uint> p, uint v) { return noorrhi_atomics[p.low >> 2].and(v); }
public uint atomicOr(AtomicPtr<uint> p, uint v) { return noorrhi_atomics[p.low >> 2].or(v); }
public uint atomicXor(AtomicPtr<uint> p, uint v) { return noorrhi_atomics[p.low >> 2].xor(v); }
public uint atomicExchange(AtomicPtr<uint> p, uint v) { return noorrhi_atomics[p.low >> 2].exchange(v); }
//! WGSL has only a weak compare-exchange, so it can fail spuriously and callers retry in a loop.
public uint atomicCompareExchange(AtomicPtr<uint> p, uint compare, uint v) {
    return noorrhi_atomics[p.low >> 2].compareExchange(compare, v);
}

public float atomicLoad(AtomicPtr<float> p) { return asfloat(noorrhi_atomics[p.low >> 2].load()); }
public void atomicStore(AtomicPtr<float> p, float v) { noorrhi_atomics[p.low >> 2].store(asuint(v)); }
//! WGSL has no float atomics, so the add retries a compare-exchange until no other invocation wrote in between.
public float atomicAdd(AtomicPtr<float> p, float v) {
    const uint index = p.low >> 2;
    uint expected = noorrhi_atomics[index].load();
    for (;;) {
        const uint found = noorrhi_atomics[index].compareExchange(expected, asuint(asfloat(expected) + v));
        if (found == expected)
            return asfloat(found);
        expected = found;
    }
}

public struct SamplerHandle {
    public uint value;
}

struct NoorRHITexture {
    uint binding;
    uint layer;
    uint2 origin;
    uint2 size;
    float maxLod;

    __init(uint handle) {
        uint4 entry = noorrhi_heap_load<uint4>(noorrhi_texture_table + handle * 16u);
        binding = entry.x & 0xffu;
        layer = entry.x >> 8u;
        origin = uint2(entry.y & 0xffffu, entry.y >> 16u);
        size = uint2(entry.z & 0xffffu, entry.z >> 16u);
        maxLod = float(max(entry.w, 1u) - 1u);
    }

    uint2 getSize(uint mip) { return max(size >> mip, uint2(1u)); }

    float2 getTexel(uint sampler, float2 uv, float lod) {
        uint modes = sampler >> noorrhi_sampler_mode_shift;
        float2 wrapped = float2(noorrhi_wrap(uv.x, modes & 3u), noorrhi_wrap(uv.y, (modes >> 2u) & 3u));
        float2 extent = float2(size);
        float2 margin = min(0.5 * exp2(ceil(clamp(lod, 0.0, maxLod))), extent * 0.5);
        return float2(origin) + clamp(wrapped * extent, margin, extent - margin);
    }
}

float noorrhi_wrap(float value, uint mode) {
    if (mode == 0u)
        return value - floor(value);
    if (mode == 1u) {
        float period = value - 2.0 * floor(value * 0.5);
        return period > 1.0 ? 2.0 - period : period;
    }
    return saturate(value);
}

public struct TextureHandle {
    public uint value;
    public float4 SampleLevel(SamplerHandle sampler, float2 uv, float lod) {
        NoorRHITexture texture = NoorRHITexture(value);
        float level = min(lod, texture.maxLod);
        return noorrhi_sample_level(sampler.value, texture.binding, texture.getTexel(sampler.value, uv, level), float(texture.layer), level);
    }
    public float4 SampleGrad(SamplerHandle sampler, float2 uv, float2 ddx, float2 ddy) {
        NoorRHITexture texture = NoorRHITexture(value);
        float2 extent = float2(texture.size);
        float2 dx = ddx * extent;
        float2 dy = ddy * extent;
        float lod = 0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1e-20));
        if (lod > texture.maxLod) {
            float shrink = exp2(texture.maxLod - lod);
            dx *= shrink;
            dy *= shrink;
            lod = texture.maxLod;
        }
        return noorrhi_sample_grad(sampler.value, texture.binding, texture.getTexel(sampler.value, uv, lod), float(texture.layer), dx, dy);
    }
    public float4 Sample(SamplerHandle sampler, float2 uv) { return SampleGrad(sampler, uv, ddx(uv), ddy(uv)); }
    public float4 Load(int2 coord, int mip = 0) {
        NoorRHITexture texture = NoorRHITexture(value);
        return noorrhi_load_pool(texture.binding, coord + int2(texture.origin >> uint(mip)), int(texture.layer), mip);
    }
    public uint2 GetDimensions(uint mip = 0) { return NoorRHITexture(value).getSize(mip); }
}

public struct UintTextureHandle {
    public uint value;
    public uint4 Load(int2 coord, int mip = 0) {
        NoorRHITexture texture = NoorRHITexture(value);
        return noorrhi_load_uint_pool(texture.binding, coord + int2(texture.origin >> uint(mip)), int(texture.layer), mip);
    }
    public uint2 GetDimensions(uint mip = 0) { return NoorRHITexture(value).getSize(mip); }
}

public struct StorageTextureHandle {
    public uint value;
    public __subscript(uint2 coord) -> float4 {
        get { return noorrhi_storage_load(value, coord); }
        [nonmutating] set { noorrhi_storage_store(value, coord, newValue); }
    }
    public uint2 GetDimensions() { return noorrhi_storage_dimensions(value); }
}

public struct UintStorageTextureHandle {
    public uint value;
    public __subscript(uint2 coord) -> uint4 {
        get { return noorrhi_storage_load_uint(value, coord); }
        [nonmutating] set { noorrhi_storage_store_uint(value, coord, newValue); }
    }
    public uint2 GetDimensions() { return noorrhi_storage_dimensions(value); }
}

#else

//! Slang's default bindless layout shares one binding between image types, which needs VK_EXT_mutable_descriptor_type.
export T getDescriptorFromHandle<T>(DescriptorHandle<T> handle) where T : IOpaqueDescriptor
{
    return defaultGetDescriptorFromHandle(handle, BindlessDescriptorOptions.None);
}

[[vk::push_constant]] ConstantBuffer<uint64_t> noorrhi_root;

T* noorrhi_pointer<T>(Ptr<T> pointer) { return (T*)(uint64_t(pointer.low) | uint64_t(pointer.high) << 32); }

public struct Ptr<T> {
    public uint low;
    public uint high;

    public __init(uint64_t address) { low = uint(address); high = uint(address >> 32); }

    public __subscript(uint index) -> T {
        get { return noorrhi_pointer(this)[index]; }
        [nonmutating] set { noorrhi_pointer(this)[index] = newValue; }
    }
}

public Ptr<T> operator+<T>(Ptr<T> pointer, uint count) {
    return Ptr<T>(uint64_t(noorrhi_pointer(pointer) + count));
}
public Ptr<T> operator-<T>(Ptr<T> pointer, uint count) {
    return Ptr<T>(uint64_t(noorrhi_pointer(pointer) - count));
}
public __prefix T operator*<T>(Ptr<T> pointer) { return pointer[0]; }

public bool isNull<T>(Ptr<T> pointer) { return (pointer.low | pointer.high) == 0; }

public T rootArguments<T>() {
    return *(T*)(noorrhi_root + noorrhi_root_offset);
}

uint4 noorrhi_root_words(uint index) {
    return ((uint4*)noorrhi_root)[index];
}

public struct AtomicPtr<T> {
    public uint low;
    public uint high;
}

Atomic<uint>* noorrhi_atomic(AtomicPtr<uint> pointer) { return (Atomic<uint>*)(uint64_t(pointer.low) | uint64_t(pointer.high) << 32); }

public AtomicPtr<T> operator+<T>(AtomicPtr<T> pointer, uint count) {
    uint64_t address = (uint64_t(pointer.low) | uint64_t(pointer.high) << 32) + uint64_t(count) * 4u;
    AtomicPtr<T> result;
    result.low = uint(address);
    result.high = uint(address >> 32);
    return result;
}

public bool isNull<T>(AtomicPtr<T> pointer) { return (pointer.low | pointer.high) == 0; }

public uint atomicLoad(AtomicPtr<uint> p) { return noorrhi_atomic(p)->load(); }
public void atomicStore(AtomicPtr<uint> p, uint v) { noorrhi_atomic(p)->store(v); }
public uint atomicAdd(AtomicPtr<uint> p, uint v) { return noorrhi_atomic(p)->add(v); }
public uint atomicSub(AtomicPtr<uint> p, uint v) { return noorrhi_atomic(p)->sub(v); }
public uint atomicMin(AtomicPtr<uint> p, uint v) { return noorrhi_atomic(p)->min(v); }
public uint atomicMax(AtomicPtr<uint> p, uint v) { return noorrhi_atomic(p)->max(v); }
public uint atomicAnd(AtomicPtr<uint> p, uint v) { return noorrhi_atomic(p)->and(v); }
public uint atomicOr(AtomicPtr<uint> p, uint v) { return noorrhi_atomic(p)->or(v); }
public uint atomicXor(AtomicPtr<uint> p, uint v) { return noorrhi_atomic(p)->xor(v); }
public uint atomicExchange(AtomicPtr<uint> p, uint v) { return noorrhi_atomic(p)->exchange(v); }
public uint atomicCompareExchange(AtomicPtr<uint> p, uint compare, uint v) { return noorrhi_atomic(p)->compareExchange(compare, v); }

Atomic<float>* noorrhi_atomic(AtomicPtr<float> pointer) { return (Atomic<float>*)(uint64_t(pointer.low) | uint64_t(pointer.high) << 32); }

public float atomicLoad(AtomicPtr<float> p) { return noorrhi_atomic(p)->load(); }
public void atomicStore(AtomicPtr<float> p, float v) { noorrhi_atomic(p)->store(v); }
public float atomicAdd(AtomicPtr<float> p, float v) { return noorrhi_atomic(p)->add(v); }

public struct SamplerHandle {
    public uint value;
    SamplerState resource() { return SamplerDescriptorHeap[NonUniformResourceIndex(value)]; }
}

public struct TextureHandle {
    public uint value;
    Texture2D<float4> resource() { return ResourceDescriptorHeap[NonUniformResourceIndex(value)]; }
    public float4 SampleLevel(SamplerHandle sampler, float2 uv, float lod) {
        return resource().SampleLevel(sampler.resource(), uv, lod);
    }
    public float4 SampleGrad(SamplerHandle sampler, float2 uv, float2 ddx, float2 ddy) {
        return resource().SampleGrad(sampler.resource(), uv, ddx, ddy);
    }
    public float4 Sample(SamplerHandle sampler, float2 uv) { return resource().Sample(sampler.resource(), uv); }
    public float4 Load(int2 coord, int mip = 0) { return resource().Load(int3(coord, mip)); }
    public uint2 GetDimensions(uint mip = 0) {
        uint width, height, levels;
        resource().GetDimensions(mip, width, height, levels);
        return uint2(width, height);
    }
}

public struct UintTextureHandle {
    public uint value;
    Texture2D<uint4> resource() { return ResourceDescriptorHeap[NonUniformResourceIndex(value)]; }
    public uint4 Load(int2 coord, int mip = 0) { return resource().Load(int3(coord, mip)); }
    public uint2 GetDimensions(uint mip = 0) {
        uint width, height, levels;
        resource().GetDimensions(mip, width, height, levels);
        return uint2(width, height);
    }
}

public struct StorageTextureHandle {
    public uint value;
    RWTexture2D<float4> resource() { return ResourceDescriptorHeap[NonUniformResourceIndex(value)]; }
    public __subscript(uint2 coord) -> float4 {
        get { return resource()[coord]; }
        [nonmutating] set { resource()[coord] = newValue; }
    }
    public uint2 GetDimensions() {
        uint width, height;
        resource().GetDimensions(width, height);
        return uint2(width, height);
    }
}

public struct UintStorageTextureHandle {
    public uint value;
    RWTexture2D<uint4> resource() { return ResourceDescriptorHeap[NonUniformResourceIndex(value)]; }
    public __subscript(uint2 coord) -> uint4 {
        get { return resource()[coord]; }
        [nonmutating] set { resource()[coord] = newValue; }
    }
    public uint2 GetDimensions() {
        uint width, height;
        resource().GetDimensions(width, height);
        return uint2(width, height);
    }
}

#endif

#if defined(NOORRHI_WEBGPU) || defined(NOORRHI_MATRIX_FALLBACK)

//! Without cooperative matrices, each group of eight lanes holds one matrix: lane l holds lines l, l + 8, and so on. A line is a row, except in MatrixB, where it is a column.
public struct NoorRHIMatrix<T : __BuiltinFloatingPointType, let Lines : int, let Length : int, let Use : int> {
    internal T values[(Lines + 7) / 8 * Length];

    public __init(T value) {
        [ForceUnroll] for (int index = 0; index < (Lines + 7) / 8 * Length; ++index)
            values[index] = value;
    }

    public static int getCount() { return (Lines + 7) / 8 * Length; }

    public __subscript(int index) -> T {
        get { return values[index]; }
        set { values[index] = newValue; }
    }

    public static This load(Ptr<T> pointer, uint stride) {
        This result = This(T(0.0));
        const uint lane = WaveGetLaneIndex() % 8;
        [ForceUnroll] for (int slot = 0; slot < (Lines + 7) / 8; ++slot) {
            const uint line = lane + slot * 8;
            if (Lines % 8 != 0 && line >= Lines)
                continue;
            [ForceUnroll] for (int index = 0; index < Length; ++index)
                result.values[slot * Length + index] = pointer[Use == 1 ? index * stride + line : line * stride + index];
        }
        return result;
    }

    public void store(Ptr<T> pointer, uint stride) {
        const uint lane = WaveGetLaneIndex() % 8;
        [ForceUnroll] for (int slot = 0; slot < (Lines + 7) / 8; ++slot) {
            const uint line = lane + slot * 8;
            if (Lines % 8 != 0 && line >= Lines)
                continue;
            [ForceUnroll] for (int index = 0; index < Length; ++index)
                pointer[Use == 1 ? index * stride + line : line * stride + index] = values[slot * Length + index];
        }
    }
}

public typealias MatrixA<T : __BuiltinFloatingPointType, let M : int, let N : int> = NoorRHIMatrix<T, M, N, 0>;
public typealias MatrixB<T : __BuiltinFloatingPointType, let M : int, let N : int> = NoorRHIMatrix<T, N, M, 1>;
public typealias MatrixAccumulator<T : __BuiltinFloatingPointType, let M : int, let N : int> = NoorRHIMatrix<T, M, N, 2>;

#define NOORRHI_MATRIX_OPERATOR(op) \
    public NoorRHIMatrix<T, L, N, U> operator op<T : __BuiltinFloatingPointType, let L : int, let N : int, let U : int>(NoorRHIMatrix<T, L, N, U> a, NoorRHIMatrix<T, L, N, U> b) { \
        [ForceUnroll] for (int index = 0; index < (L + 7) / 8 * N; ++index) \
            a.values[index] = a.values[index] op b.values[index]; \
        return a; \
    }
NOORRHI_MATRIX_OPERATOR(+)
NOORRHI_MATRIX_OPERATOR(-)
NOORRHI_MATRIX_OPERATOR(*)
NOORRHI_MATRIX_OPERATOR(/)
#undef NOORRHI_MATRIX_OPERATOR

public NoorRHIMatrix<T, L, N, U> operator*<T : __BuiltinFloatingPointType, let L : int, let N : int, let U : int>(NoorRHIMatrix<T, L, N, U> a, T scale) {
    [ForceUnroll] for (int index = 0; index < (L + 7) / 8 * N; ++index)
        a.values[index] = a.values[index] * scale;
    return a;
}

public __prefix NoorRHIMatrix<T, L, N, U> operator-<T : __BuiltinFloatingPointType, let L : int, let N : int, let U : int>(NoorRHIMatrix<T, L, N, U> a) {
    [ForceUnroll] for (int index = 0; index < (L + 7) / 8 * N; ++index)
        a.values[index] = -a.values[index];
    return a;
}

public MatrixAccumulator<C, M, N> multiplyAdd<A : __BuiltinFloatingPointType, B : __BuiltinFloatingPointType, C : __BuiltinFloatingPointType, let M : int, let N : int, let K : int>(
    MatrixA<A, M, K> a, MatrixB<B, K, N> b, MatrixAccumulator<C, M, N> c) {
    const uint group = WaveGetLaneIndex() & ~7u;
    [ForceUnroll] for (int column = 0; column < N; ++column)
        [ForceUnroll] for (int index = 0; index < K; ++index) {
            const C right = C(WaveReadLaneAt(b.values[column / 8 * K + index], group + column % 8).toFloat());
            [ForceUnroll] for (int slot = 0; slot < (M + 7) / 8; ++slot)
                c.values[slot * N + column] = c.values[slot * N + column] + C(a.values[slot * K + index].toFloat()) * right;
        }
    return c;
}

#else

public typealias MatrixA<T : __BuiltinFloatingPointType, let M : int, let N : int> = linalg.CoopMat<T, MemoryScope.Subgroup, M, N, linalg.CoopMatMatrixUse.MatrixA>;
public typealias MatrixB<T : __BuiltinFloatingPointType, let M : int, let N : int> = linalg.CoopMat<T, MemoryScope.Subgroup, M, N, linalg.CoopMatMatrixUse.MatrixB>;
public typealias MatrixAccumulator<T : __BuiltinFloatingPointType, let M : int, let N : int> =
    linalg.CoopMat<T, MemoryScope.Subgroup, M, N, linalg.CoopMatMatrixUse.MatrixAccumulator>;

//! Slang loads and stores cooperative matrices only through buffers and groupshared memory, so the device pointer access is written in SPIR-V.
public extension<T : __BuiltinFloatingPointType, let M : int, let N : int, let U : linalg.CoopMatMatrixUse> linalg.CoopMat<T, MemoryScope.Subgroup, M, N, U> {
    public static This load(Ptr<T> pointer, uint stride) {
        T* address = noorrhi_pointer(pointer);
        return spirv_asm { result:$$This = OpCooperativeMatrixLoadKHR $address $(0) $stride Aligned !16; };
    }
    public void store(Ptr<T> pointer, uint stride) {
        T* address = noorrhi_pointer(pointer);
        spirv_asm { OpCooperativeMatrixStoreKHR $address $this $(0) $stride Aligned !16; };
    }
    public static int getCount() { return GetLength(); }
}

public MatrixAccumulator<C, M, N> multiplyAdd<A : __BuiltinFloatingPointType, B : __BuiltinFloatingPointType, C : __BuiltinFloatingPointType, let M : int, let N : int, let K : int>(
    MatrixA<A, M, K> a, MatrixB<B, K, N> b, MatrixAccumulator<C, M, N> c) {
    return linalg.coopMatMulAdd<C, false>(a, b, c);
}

#endif

//! These match DispatchSize, DrawArgs and DrawIndexedArgs in Types.h, so a shader writes the commands of dispatchIndirect and the indirect draws.
public struct DispatchSize {
    public uint x;
    public uint y;
    public uint z;
}

public struct DrawArgs {
    public uint vertexCount;
    public uint instanceCount;
    public uint firstVertex;
    public uint firstInstance;
}

public struct DrawIndexedArgs {
    public uint indexCount;
    public uint instanceCount;
    public uint firstIndex;
    public int vertexOffset;
    public uint firstInstance;
}

#include "NoorRHIRayTracing.slang"
