# NoorRHI

A small, object-oriented Vulkan API. Buffers are device addresses; textures and
samplers are 32-bit indices into one bindless descriptor set the device owns and
fills automatically. There are no pipeline layouts, descriptor sets, resource
registries in the app, or upload batches.

Everything it needs is core Vulkan 1.3; see
[Requirements and verification](#requirements-and-verification).

## Building

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /your/prefix
```

Vulkan and [VulkanMemoryAllocator][vma] are the only dependencies. VMA comes
from an existing `GPUOpen::VulkanMemoryAllocator` target, then
`find_package(VulkanMemoryAllocator)`, then `-DNOORRHI_VMA_DIR=`, then a vendored copy
at `external/VulkanMemoryAllocator`.
`NOORRHI_BUILD_EXAMPLES` and `NOORRHI_BUILD_TESTS` default to `ON` for a standalone
build and `OFF` when this project is added as a subdirectory. Both need `slangc`
from the Vulkan SDK (set `VULKAN_SDK`). The tests also need Catch2 v3, found with
`find_package`, `-DNOORRHI_CATCH2_DIR=`, or `external/Catch2`.

[vma]: https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator

## Consuming

```cmake
find_package(NoorRHI REQUIRED)
target_link_libraries(my_app PRIVATE NoorRHI::NoorRHI)
```

Or vendor it and `add_subdirectory(external/NoorRHI)`, which defines the same
`NoorRHI::NoorRHI` target.

```cpp
noorrhi::Device device;
auto vertices = device.buffer<Vertex>(source.size());
vertices.upload(std::span<const Vertex>(source));

noorrhi::Shared<SceneData> scene(device);
scene.data.vertices = vertices.ptr();
scene.commit();

struct FrameData { noorrhi::GpuPtr<SceneData> scene; Camera camera; };
pipeline.launch(groups, FrameData{scene.ptr(), camera});
```

Buffer, image, sampler, and shared-data owners are move-only. Internal references
keep submitted Vulkan objects alive until completion. GPU pointers and texture
handles are non-owning: keep their owners alive while using them, including
objects referenced indirectly through another GPU record. Destroy resources
before their Device; a surface provider outlives the Swapchains created for it.

## Data and uploads

- `Buffer<T>`: `ptr()`, `upload(span, elementOffset)`, `download(span)`.
- `Image<T>`: full-level `upload(span)` and `download(span)`;
  `sampled_handle()` and `storage_handle()` for shaders.
  `handle()` is a checked host reference for render targets and interop.
- `Sampler`: `handle()` for shaders. There is no combined image/sampler; a
  shader pairs a sampled texture handle with a sampler handle.

## Textures and samplers

The device creates one bindless descriptor set at construction, with arrays
sized by `DeviceConfig::texture_descriptor_capacity` (default 16384) and
`DeviceConfig::sampler_descriptor_capacity` (default 256). A `Sampled` or
`Storage` image takes one texture slot per usage, and a sampler takes one
sampler slot. Handles are slot indices, so shaders that import `noorrhi.slang`
index them with Slang's heap syntax:

```slang
import noorrhi;

RWTexture2D<float4> output = ResourceDescriptorHeap[arguments.output];
Texture2D<float4> albedo = ResourceDescriptorHeap[arguments.albedo];
SamplerState linear = SamplerDescriptorHeap[arguments.sampler];
```

Slot 0 is reserved and never written, so a zero handle means "no
texture". Store handles as `uint` in shared records. A resource returns its
slots when it is retired, meaning after every submission that might still read
them has finished. Until then its handles stay valid for in-flight work.
Slots are then reused. The set never grows: running out throws
`ErrorCode::OutOfMemory`. Wrap a handle that can differ within a wave, such as
a per-hit material texture, in `NonUniformResourceIndex()`.

Every pipeline shares one layout: the bindless set and an 8-byte push constant
carrying the root argument pointer. The set is bound before every dispatch,
draw, and trace. That makes it safe to record
legacy descriptor-set commands, such as an ImGui backend's, into the same
frame command buffer.
- `Shared<T>`: public `data`, `ptr()`, `commit()`. Commit compares the
  trivially-copyable record and skips unchanged bytes.

Upload copies into temporary mapped memory before returning. The caller may
immediately reuse its source; staging lives until the GPU copy finishes.
Temporary duplicate memory is intentional. Download waits for its copy.
There is no fixed staging budget or global dirty-object list.

Small per-frame values, including NoorRay's camera, go directly in root arguments.
The backend copies these into mapped chunks and reuses a chunk once every submission
that may read it has completed by timeline value. Chunks are added as needed, so a
frame can record any amount of argument data; `DeviceConfig::argument_arena_bytes` is
the chunk size.

## Presentation

A `Device` renders into images and needs no window. Presentation is an optional
`Swapchain`, one per window, for hosts that show results directly:

```cpp
noorrhi::Device device({.presentation = &window});   // window: a SurfaceProvider
noorrhi::Swapchain swapchain(device, window, {.present_mode = noorrhi::PresentMode::LowLatency});

while (running) {
    swapchain.wait_until_ready();      // before sampling input
    poll_input_and_update();
    noorrhi::Frame frame = swapchain.begin_frame();
    if (!frame) continue;              // minimised or just rebuilt
    record_work_and_draw_into(frame.target());
    swapchain.present(std::move(frame));
}
```

`DeviceConfig::presentation` only adopts the window system's loader and instance
extensions and restricts GPU and queue selection to ones that can present to it.
Each `Swapchain` creates and owns its surface.

The swapchain is built for low latency:

- `PresentMode::LowLatency` (default) prefers MAILBOX, then FIFO_LATEST_READY
  (enabled when the device supports it), then FIFO. `Vsync` is FIFO and
  `Immediate` prefers IMMEDIATE. `active_present_mode()` reports the result;
  `set_present_mode()` switches on the next frame.
- `SwapchainDesc::max_frames_in_flight` (default 1) paces the CPU to the GPU, so
  frames never queue behind a busy GPU. `wait_until_ready()` performs that wait
  up front so input is sampled after it, not before.
- Resizes rebuild from the previous chain without waiting for the GPU. The old
  chain, its images and semaphores are retired and released once the frames
  that used them have finished.
- `present()` submits and queues the image without a CPU wait.

## Frames and synchronization

Prepare changed resources and commit shared data **before** `begin_frame()`.
Launch rendering and presentation between begin and present; they share one
submission. Acquire slots wait only when reused.
Timestamp reads are nonblocking and return the latest available measurement.

Uploads/readbacks during an open frame are rejected. Large scene replacement and
destruction may synchronize. Ordered submissions handle normal updates and TLAS
refits. Dropping a Frame discards it and rebuilds the acquired swapchain before
the next frame. The window provider reports live pixel dimensions.

The API is single-threaded per Device. Raw pointers cannot validate arbitrary
shader pointer graphs; callers remain responsible for pointer bounds, compatible
host/shader layouts, and owner lifetime. Vulkan interop additionally requires the
external API to finish using exported objects before their owners are destroyed.

## Requirements and verification

Vulkan 1.3 is mandatory, with buffer device addresses, timeline semaphores,
synchronization2, dynamic rendering, `scalarBlockLayout`, `shaderInt64`,
`shaderDrawParameters` and the descriptor indexing features behind the bindless
set (runtime arrays, partially bound and update-after-bind sampled and storage
images, non-uniform indexing). Devices without them are skipped at selection.
Ray tracing is checked separately and reported in `DeviceFeatures`.

Shaders are supplied as SPIR-V bytes. Compile them with `NOORRHI_SLANG_FLAGS`
(set by this project's CMake), which puts `shaders/noorrhi.slang` on the import
path and the bindless set at set 0. Any shader that indexes
`ResourceDescriptorHeap` or `SamplerDescriptorHeap` must `import noorrhi;`.
Validate modules with `spirv-val --target-env vulkan1.3`. Requested validation fails clearly if its
layer is absent.

`memory_report()` reports VMA allocation bytes, reserved block bytes, allocation
count, and argument-arena capacity, including pending allocations. Dedicated
external-memory images are outside these VMA totals.

GPU tests cover compute, images (including bindless reads and writes, and slot reuse
and exhaustion), ray tracing, shared updates, argument wraparound, and retirement. `noorrhi_presentation_test` additionally needs a desktop display and
tests resizing, present-mode switches and abandoned frames. Enable synchronization validation with
`VK_LAYER_VALIDATE_SYNC=1`.
