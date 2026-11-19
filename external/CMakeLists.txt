include(FetchContent)

if(NOORRHI_BACKEND STREQUAL "Vulkan")
    #! The backend loads libvulkan at run time, so only the headers are needed.
    find_package(VulkanHeaders 1.4 CONFIG REQUIRED GLOBAL HINTS "$ENV{VULKAN_SDK}")
    if(NOT TARGET SPIRV-Headers)
        FetchContent_Declare(spirv_headers
            URL https://github.com/KhronosGroup/SPIRV-Headers/archive/6efe5ae6b7c6ad29da3f2cc3f164b9c3d72b78b1.tar.gz
            URL_HASH SHA256=d035de73b4085301bf921dc66997ce993fbf7c0d78f3308e67b6be2513a8b88c
            SYSTEM EXCLUDE_FROM_ALL)
        set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
        set(SPIRV_HEADERS_SKIP_INSTALL ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(spirv_headers)
    endif()
    return()
endif()

FetchContent_Populate(dawn
    URL https://github.com/google/dawn/archive/43af65b60452f3e51ea94db848b6bb8c053a1fd0.tar.gz
    URL_HASH SHA256=4fc51992e385a733cdd2f8eafbaee6b3372229a0b8152dd1d970a130060488a8)
set(dawn_SOURCE_DIR "${dawn_SOURCE_DIR}" PARENT_SCOPE)
#! The archive has no third-party code, which fetch_dawn_dependencies.py checks out.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python3_EXECUTABLE}" "${dawn_SOURCE_DIR}/tools/fetch_dawn_dependencies.py"
                        --directory "${dawn_SOURCE_DIR}" COMMAND_ERROR_IS_FATAL ANY)

if(EMSCRIPTEN)
    #! The emdawnwebgpu port in emsdk has no SetImmediates, so the bindings come from Dawn.
    set(DAWN_FETCH_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    #! Dawn's third-party code warns a lot and is not ours to fix.
    add_compile_options(-w)
    #! Dawn requires NDEBUG outside Debug, and CompilerOptions.cmake removes it from RelWithDebInfo.
    add_compile_definitions($<$<NOT:$<CONFIG:Debug>>:NDEBUG>)
    add_subdirectory("${dawn_SOURCE_DIR}" dawn EXCLUDE_FROM_ALL SYSTEM)
    add_library(noorrhi_webgpu INTERFACE)
    target_link_libraries(noorrhi_webgpu INTERFACE emdawnwebgpu_cpp)
elseif(NOT TARGET dawn::webgpu_dawn)
    set(DAWN_FETCH_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_PROTOBUF OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_NULL OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_DESKTOP_GL OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_OPENGLES OFF CACHE BOOL "" FORCE)
    set(DAWN_USE_GLFW OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_MONOLITHIC_LIBRARY STATIC CACHE STRING "" FORCE)
    set(TINT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_SPV_READER OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_GLSL_WRITER OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_GLSL_VALIDATOR OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_IR_BINARY OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF)
    #! Dawn's third-party code warns a lot and is not ours to fix.
    add_compile_options(-w)
    #! Dawn requires NDEBUG outside Debug, and CompilerOptions.cmake removes it from RelWithDebInfo.
    add_compile_definitions($<$<NOT:$<CONFIG:Debug>>:NDEBUG>)
    if(WIN32 AND NOT CMAKE_HOST_WIN32)
        #! Dawn includes <Windows.h>, and llvm-mingw has only windows.h, which a Linux file system does not match.
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/mingw-case/Windows.h" "#include <windows.h>\n")
        include_directories(SYSTEM "${CMAKE_CURRENT_BINARY_DIR}/mingw-case")
    endif()
    if(WIN32)
        #! Dawn calls std::getenv without <cstdlib>, which MSVC includes through other headers and libc++ does not.
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include cstdlib>")
        #! A C++ module must start its translation unit, so the forced <cstdlib> breaks Dawn's webgpu module.
        set(DAWN_SUPPORTS_CXX_MODULES OFF CACHE BOOL "" FORCE)
        #! Dawn's D3D11 and D3D12 backends need Windows SDK headers that llvm-mingw does not have, such as DXProgrammableCapture.h.
        set(DAWN_ENABLE_D3D11 OFF CACHE BOOL "" FORCE)
        set(DAWN_ENABLE_D3D12 OFF CACHE BOOL "" FORCE)
        set(DAWN_USE_WINDOWS_UI OFF CACHE BOOL "" FORCE)
    endif()
    add_subdirectory("${dawn_SOURCE_DIR}" dawn EXCLUDE_FROM_ALL SYSTEM)
endif()
