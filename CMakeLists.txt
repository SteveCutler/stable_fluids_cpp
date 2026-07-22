cmake_minimum_required(VERSION 3.16)

project(StableFluidsC++)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(
    SFML 
    COMPONENTS Graphics Window System 
    REQUIRED
)

# ------------------------------------------------------------
# Simulation code shared by the application and tests
# ------------------------------------------------------------

add_library(fluid_core
src/Grid.cpp
src/Emitter.cpp
)

target_include_directories(fluid_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(fluid_core
    PUBLIC
        SFML::Graphics
        SFML::Window
        SFML::System
)

# ------------------------------------------------------------
# Main application
# ------------------------------------------------------------


add_executable(StableFluidsC++ 
src/main.cpp
src/Slider.cpp
src/Renderer.cpp
)


target_include_directories(StableFluidsC++
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
)

target_link_libraries(StableFluidsC++
    PRIVATE
        fluid_core
)



# ------------------------------------------------------------
# CPU numerical parity test
# ------------------------------------------------------------

add_executable(cpu_parity
    Tests/cpu_parity.cpp
)

target_link_libraries(cpu_parity
    PRIVATE
        fluid_core
)

enable_testing()

add_test(
    NAME cpu_parity
    COMMAND cpu_parity
)


# ------------------------------------------------------------
# Compiler warnings
# ------------------------------------------------------------

if(MSVC)
    target_compile_options(fluid_core PRIVATE
        /W4
        /permissive-
    )

    target_compile_options(StableFluidsC++ PRIVATE
        /W4
        /permissive-
    )

    target_compile_options(cpu_parity PRIVATE
        /W4
        /permissive-
    )
else()
    target_compile_options(fluid_core PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )

    target_compile_options(StableFluidsC++ PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )

    target_compile_options(cpu_parity PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()