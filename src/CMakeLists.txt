file(GLOB_RECURSE noorrhi_sources CONFIGURE_DEPENDS NoorRHI/*.cpp)
if(NOORRHI_BACKEND STREQUAL "Vulkan")
    list(FILTER noorrhi_sources EXCLUDE REGEX "/NoorRHI/WebGpu/|/NoorRHI/Shaders/(SpirvModule|WgslTranslator)")
else()
    list(FILTER noorrhi_sources EXCLUDE REGEX "/NoorRHI/Vulkan/")
endif()

noorrhi_compile_shaders(NoorRHIBuildShaders NoorRHI/Shaders/RayTracingBuild.slang)

add_library(NoorRHI STATIC ${noorrhi_sources} $<TARGET_OBJECTS:NoorRHIBuildShaders>)
add_library(NoorRHI::NoorRHI ALIAS NoorRHI)
add_dependencies(NoorRHI NoorRHIBuildShaders)
target_include_directories(NoorRHI PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/NoorRHIBuildShaders")
target_include_directories(NoorRHI PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(NoorRHI PUBLIC cxx_std_23)
#! Dawn builds without RTTI, so the classes that derive from its platform interfaces must too.
target_compile_options(NoorRHI PRIVATE -fno-exceptions -fno-rtti -Wall -Wextra)
target_precompile_headers(NoorRHI PRIVATE <algorithm> <expected> <memory> <optional> <span> <string> <string_view> <vector>)

if(NOORRHI_BACKEND STREQUAL "Vulkan")
    target_compile_definitions(NoorRHI PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_EXCEPTIONS
                                              VULKAN_HPP_ASSERT_ON_RESULT=static_cast<void>)
    target_link_libraries(NoorRHI PUBLIC Vulkan::Headers)
    target_precompile_headers(NoorRHI PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/NoorRHI/Vulkan/Vulkan.h")
else()
    #! Tint carries Dawn's native webgpu_cpp.h, so the WebGPU bindings must come first on the include path.
    if(EMSCRIPTEN)
        target_link_libraries(NoorRHI PUBLIC noorrhi_webgpu)
        #! Tint recurses deeply, and iOS Safari gives a page about 300 MB, so the main stack is 16 MB and each worker per core gets 4 MB.
        target_link_options(NoorRHI INTERFACE -sASYNCIFY -sSTACK_SIZE=16MB -sDEFAULT_PTHREAD_STACK_SIZE=4MB -sINITIAL_MEMORY=32MB -sALLOW_MEMORY_GROWTH=1)
    else()
        target_link_libraries(NoorRHI PUBLIC dawn::webgpu_dawn)
    endif()
    target_link_libraries(NoorRHI PRIVATE tint_lang_spirv_reader tint_lang_wgsl_writer tint_lang_core_ir dawn_utils_config SPIRV-Headers)
    target_include_directories(NoorRHI SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../external/dawn")
    target_compile_definitions(NoorRHI PRIVATE SPV_ENABLE_UTILITY_CODE)
    #! Dawn requires NDEBUG outside Debug, and CompilerOptions.cmake removes it from RelWithDebInfo.
    set_source_files_properties(NoorRHI/Shaders/WgslTranslator.cpp PROPERTIES COMPILE_DEFINITIONS "$<$<NOT:$<CONFIG:Debug>>:NDEBUG>")
endif()
