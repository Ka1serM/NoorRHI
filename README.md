# NoorRHI

A small C++23 GPU API with a Vulkan backend and a WebGPU backend. The same app code runs on the desktop and in the
browser. Shaders are written in Slang and use pointers and bindless handles.

```cpp
// shaders/Scale.h, included by C++ and by Slang
#include "NoorRHI/Shaders/ShaderTypes.h"
namespace Scale {
using namespace NoorRHI;
struct Arguments { Ptr<float> values; uint count; };
}
```

```slang
// shaders/Scale.slang
#include "Scale.h"
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const Scale::Arguments arguments = rootArguments<Scale::Arguments>();
    if (id.x < arguments.count)
        arguments.values[id.x] *= 2.0;
}
```

```cpp
#include "NoorRHI/NoorRHI.h"
#include "Scale.slang.h"

using namespace NoorRHI;

Device device;
ComputePipeline pipeline;
Buffer<float> values;
if (!device.open() || !pipeline.open(device, Scale::main) || !values.open(device, 1024))
    return 1;

CommandList list(device);
list.dispatch(pipeline, {1024}, Scale::Arguments{values.getPtr(), 1024});
std::expected<Token, Error> token = device.submit(std::move(list));
if (!token || !device.wait(*token))
    return 1;
```

```cmake
add_subdirectory(external/NoorRHI)
include(external/NoorRHI/cmake/NoorRHIShaders.cmake)
noorrhi_add_shaders(AppShaders shaders/Scale.slang)
target_link_libraries(app PRIVATE AppShaders)
```

## Synchronization

- Each command sees the writes of the commands before it in the list. Each list sees the writes of the lists that were
  submitted before it on the same queue. The list records only the barriers that this needs. Each barrier waits only
  for the stages that ran and blocks only the stages of the next command.
- Copies of memory that does not overlap run without a barrier between them.
- Put independent commands between `beginConcurrent()` and `endConcurrent()`. Then they have no barriers between them.
  A command in the group must not read what another command in the group writes.
- Draws in one render pass do not see the writes of other draws in the same pass.
- To use the results of a list on another queue, give its token to `submit` as a wait.

## Autodiff

Slang autodiff works on both backends. Mark a function `[Differentiable]` and call `bwd_diff` on it. To get the
gradients of weights in a buffer, read each weight through a `[Differentiable]` load whose `[BackwardDerivative]` adds
the gradient with `atomicAdd` to an `AtomicPtr<float>`. `tests/shaders/Network.slang` gets the gradients of a small MLP
this way.

## Portability

- Use `SV_VulkanVertexID` and `SV_VulkanInstanceID` in vertex shaders. They include the first vertex and the first
  instance on both backends. `SV_VertexID` and `SV_InstanceID` do not agree between SPIR-V and WGSL.
- On WebGPU, index buffers and indirect commands of draws without a count must be in `BufferMemory::Plain`. The count
  of a count draw can be in atomic memory.
- On WebGPU, a count draw runs a small compute pass before its render pass to clear the commands past the count.
- `Queue::Async` uses a dedicated compute queue when Vulkan has one. On WebGPU it is the same queue.
- `openAsync` compiles a pipeline in the background. A draw or dispatch with a pipeline that is not ready fails.
- `CommandList::download` copies into a `Readback`. `isReady()` does not block, and `read()` waits for the copy.

## Linux

Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) and set `VULKAN_SDK`. It gives `slangc` and the Vulkan
headers.

```sh
sudo dnf install clang lld libcxx-devel libcxx-static libcxxabi-static compiler-rt llvm-libunwind-devel \
    llvm-libunwind-static cmake ninja-build python3 wayland-devel libxkbcommon-devel
```

```sh
git submodule update --init --recursive
cmake --workflow --preset linux-release
cmake --workflow --preset linux-webgpu-release
```

## Windows

Install the Vulkan SDK and [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) with its `bin` folder in PATH.

```powershell
git submodule update --init --recursive
cmake --workflow --preset windows-release
cmake --workflow --preset windows-webgpu-release
```

## Web

The web build uses the Linux dependencies and an activated [emsdk](https://emscripten.org).

```sh
cmake --workflow --preset web-release
```
