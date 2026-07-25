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
# Tsan flag setup
# ------------------------------------------------------------

option (ENABLE_TSAN "Enable ThreadSanitizer" OFF)

if(ENABLE_TSAN)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
        message(FATAL_ERROR
            "ThreadSanitizer requires Clang, AppleClang, or GCC"
            )
    endif()

    add_library(tsan_flags INTERFACE)

    target_compile_options(tsan_flags INTERFACE
        -fsanitize=thread
        -fno-omit-frame-pointer
        )

    target_link_options(tsan_flags INTERFACE
        -fsanitize=thread
        )

    target_link_libraries(fluid_core PUBLIC tsan_flags)
endif()



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


# ------------------------------------------------------------
# Projection test
# ------------------------------------------------------------

add_executable(projection_test
    Tests/projection_test.cpp
)

target_link_libraries(projection_test
    PRIVATE
        fluid_core
)

# ------------------------------------------------------------
# Register tests with CTest
# ------------------------------------------------------------

enable_testing()

add_test(
    NAME cpu_parity
    COMMAND cpu_parity
)

add_test(
    NAME projection_test
    COMMAND projection_test
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
    target_compile_options(projection_test PRIVATE
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

    target_compile_options(projection_test PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()