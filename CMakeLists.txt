cmake_minimum_required(VERSION 3.16)
project(StableFluidsC++)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SFML COMPONENTS Graphics Window System REQUIRED)

add_executable(StableFluidsC++ 
src/main.cpp
src/Grid.cpp
src/Slider.cpp
src/Renderer.cpp
src/Emitter.cpp
)

if(MSVC)
    target_compile_options(StableFluidsC++ PRIVATE
        /W4
        /permissive-
    )
else()
    target_compile_options(StableFluidsC++ PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

target_link_libraries(StableFluidsC++
    PRIVATE
    SFML::Graphics
    SFML::Window
    SFML::System
)
target_include_directories(
    StableFluidsC++
    PRIVATE
    include
    assets
)