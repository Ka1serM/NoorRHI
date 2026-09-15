# gpu API examples

These small programs use the public `gpu` API directly and compile their
shaders with Slang (`slangc` from the Vulkan SDK). Examples are built by default
in a standalone configure:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gpu_compute_example gpu_graphics_example gpu_raytracing_example
./build/examples/gpu_compute_example
./build/examples/gpu_graphics_example
./build/examples/gpu_raytracing_example
```

The compute example uploads two arrays and checks their sum. The graphics
example records a triangle into an off-screen color attachment. The ray
tracing example builds a BLAS and TLAS, launches a ray-generation shader, and
checks its output. The shaders contain no descriptor or layout bindings; only
the root pointer supplied by `gpu::Device` is used for compute and ray tracing.
Ray tracing reports a skip when the selected Vulkan device does not expose the
required feature.
