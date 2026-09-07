cmake_minimum_required(VERSION 3.16)
project(StableFluidsC++ LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_METAL "Build the experimental Apple Metal application" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer for CPU code" OFF)
include(CTest)
find_package(SFML 3 COMPONENTS Graphics Window System REQUIRED)
find_package(Threads REQUIRED)

function(fluid_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

# Absolute source asset path supports launching these build-tree executables
# from any directory. Reconfigure after moving the source checkout.
function(fluid_assets target)
    target_compile_definitions(${target} PRIVATE
        "FLUID_FONT_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/assets/fonts/RobotoMono.ttf\"")
endfunction()

add_library(fluid_core src/Grid.cpp src/Emitter.cpp)
target_include_directories(fluid_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fluid_core PUBLIC SFML::Graphics SFML::Window SFML::System Threads::Threads)
fluid_warnings(fluid_core)

if(ENABLE_TSAN)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(FATAL_ERROR "ThreadSanitizer requires Clang, AppleClang, or GCC")
    endif()
    add_library(tsan_flags INTERFACE)
    target_compile_options(tsan_flags INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(tsan_flags INTERFACE -fsanitize=thread)
    target_link_libraries(fluid_core PUBLIC tsan_flags)
endif()

add_executable(StableFluidsC++ src/main.cpp src/Slider.cpp src/Renderer.cpp)
target_link_libraries(StableFluidsC++ PRIVATE fluid_core)
fluid_warnings(StableFluidsC++)
fluid_assets(StableFluidsC++)

if(BUILD_TESTING)
    foreach(test cpu_parity projection_test scalar_boundary_test)
        add_executable(${test} tests/${test}.cpp)
        target_link_libraries(${test} PRIVATE fluid_core)
        fluid_warnings(${test})
        add_test(NAME ${test} COMMAND ${test})
    endforeach()
endif()

if(BUILD_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "BUILD_METAL requires an Apple platform; leave it OFF for CPU builds")
    endif()
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_program(XCRUN_EXECUTABLE xcrun)
    if(NOT METAL_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK OR NOT XCRUN_EXECUTABLE)
        message(FATAL_ERROR "BUILD_METAL requires Metal/Foundation frameworks and xcrun")
    endif()
    set(METAL_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/metal")
    set(FLUID_METALLIB "${METAL_OUTPUT_DIR}/FluidKernels.metallib")
    add_custom_command(
        OUTPUT "${FLUID_METALLIB}"
        BYPRODUCTS "${METAL_OUTPUT_DIR}/FluidKernels.air"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${METAL_OUTPUT_DIR}"
        COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal
            -fmodules-cache-path=${METAL_OUTPUT_DIR}/module-cache
            -c "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/FluidKernels.metal"
            -o "${METAL_OUTPUT_DIR}/FluidKernels.air"
        COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib
            "${METAL_OUTPUT_DIR}/FluidKernels.air" -o "${FLUID_METALLIB}"
        DEPENDS src/metal/FluidKernels.metal src/metal/SimplexNoiseCompute.metal
        VERBATIM)
    add_custom_target(fluid_shaders DEPENDS "${FLUID_METALLIB}")
    add_library(fluid_metal src/metal/MetalGrid.cpp src/metal/MetalContext.cpp
        src/metal/MetalImplementation.cpp src/metal/MetalRenderer.cpp)
    target_include_directories(fluid_metal PUBLIC include include/metal)
    target_include_directories(fluid_metal SYSTEM PUBLIC external/metal-cpp-main)
    target_link_libraries(fluid_metal PUBLIC SFML::Graphics SFML::Window SFML::System
        PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    fluid_warnings(fluid_metal)
    add_executable(StableFluidsMetal src/metal/MetalMain.cpp src/Slider.cpp src/Emitter.cpp)
    target_link_libraries(StableFluidsMetal PRIVATE fluid_metal)
    target_compile_definitions(StableFluidsMetal PRIVATE "FLUID_METALLIB_PATH=\"${FLUID_METALLIB}\"")
    add_dependencies(StableFluidsMetal fluid_shaders)
    fluid_warnings(StableFluidsMetal)
    fluid_assets(StableFluidsMetal)
    if(BUILD_TESTING)
        add_executable(metal_ownership_test tests/metal_ownership_test.cpp src/Emitter.cpp)
        target_link_libraries(metal_ownership_test PRIVATE fluid_metal)
        fluid_warnings(metal_ownership_test)
        add_dependencies(metal_ownership_test fluid_shaders)
        add_test(NAME metal_ownership_test COMMAND metal_ownership_test "${FLUID_METALLIB}")
        set_tests_properties(metal_ownership_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
