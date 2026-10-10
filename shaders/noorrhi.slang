// Every shader that reads textures or samplers imports this module. It maps
// ResourceDescriptorHeap[i] / SamplerDescriptorHeap[i] onto the device's
// bindless set (set 0, compile with -bindless-space-index 0) with one array per
// descriptor type, which is all core Vulkan 1.3 offers: samplers at binding 0,
// sampled images at binding 2, storage images at binding 3. Slang's default
// layout instead shares one binding between image types, which needs
// VK_EXT_mutable_descriptor_type.
//
// Wrap a handle that can differ between invocations of one wave (a per-hit
// material texture, say) in NonUniformResourceIndex(); Vulkan leaves a
// divergent index into a descriptor array undefined otherwise.
module noorrhi;

export T getDescriptorFromHandle<T>(DescriptorHandle<T> handle) where T : IOpaqueDescriptor
{
    return defaultGetDescriptorFromHandle(handle, BindlessDescriptorOptions.None);
}
