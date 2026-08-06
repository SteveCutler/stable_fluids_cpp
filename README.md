# Stable Fluids C++

A realtime, CPU based 2D fluid simulation written in C++20 and rendered with
SFML. The project explores numerical fluid simulation, data oriented design,
multithreading, profiling, and numerical validation as a foundation for
future GPU implementation in Metal.

The current CPU version runs an interactive `1024 × 1024` simulation with coloured
density transport, configurable emitters, and live controls for the major
simulation parameters.

<img width="1019" height="1017" alt="Screenshot 2026-07-25 at 8 19 43 AM" src="https://github.com/user-attachments/assets/d2bd0658-17db-4171-81f7-a921710ac069" />

## Current features

-   Semi-Lagrangian advection for velocity and RGB density fields
-   Velocity diffusion and a Jacobi pressure projection
-   Buoyancy and procedural noise-driven motion
-   Three independent density channels for colour mixing
-   Multiple configurable density emitters
-   Closed-wall velocity and scalar boundary handling
-   Structure-of-arrays storage for simulation fields
-   Serial and row-partitioned multithreaded execution paths
-   Live FPS and per-stage timing instrumentation
-   Interactive parameter controls and velocity-field visualization
-   CMake/CTest numerical validation
-   Optional ThreadSanitizer build configuration

## CPU v1 accomplishments

-   Built the complete simulation pipeline from scalar and velocity fields rather
    than relying on a fluid-simulation library.
-   Refactored the grid into contiguous structure-of-arrays buffers for density,
    velocity, pressure, divergence, noise, and scratch data.
-   Implemented a reusable row-partitioning system for parallel CPU kernels.
-   Multithreaded the major simulation stages, including noise generation,
    velocity updates, advection, pressure projection, diffusion, and pixel
    generation.
-   Progressed from an early lower-resolution prototype to an interactive
    `1024 × 1024` CPU simulation.
-   Added per-kernel profiling to identify pressure projection and other
    computational bottlenecks.
-   Added a deterministic serial-versus-threaded parity test covering 100
    simulation steps.
-   Added divergence statistics and a projection sanity test to measure the
    numerical effect of the pressure solve.
-   Identified and documented the slow convergence of fixed-budget Jacobi
    projection on smooth, low-frequency divergence fields.

## Simulation pipeline

Each update performs the following stages:

1. Generate a time-varying OpenSimplex noise field.
2. Convert the noise gradient into curl-like velocity forcing.
3. Advect and diffuse the velocity field.
4. Measure divergence, solve for pressure, and subtract its gradient.
5. Inject RGB density from the emitters.
6. Diffuse and advect each density channel.
7. Convert the density fields into RGBA pixels for SFML.

## Controls

| Input | Action                                            |
| ----- | ------------------------------------------------- |
| `P`   | Pause or resume the simulation                    |
| `R`   | Reset density, velocity, pressure, and divergence |
| `V`   | Toggle velocity-vector visualization              |
| `M`   | Toggle serial and multithreaded execution         |
| Mouse | Adjust the on-screen parameter sliders            |

The UI exposes controls for buoyancy, density diffusion, decay, noise strength,
and noise frequency.

## Requirements

-   A C++20 compiler
-   CMake 3.16 or newer
-   SFML

On macOS with Homebrew:

```bash
brew install cmake sfml
```

## Build and run

From the repository root:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
./build/release/StableFluidsC++
```

Run the executable from the repository root so it can locate the font in the
`assets` directory.

For a debug build:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug
./build/debug/StableFluidsC++
```

## Validation

The project separates the reusable simulation code into the `fluid_core`
library and builds two CTest executables.

### CPU parity

`cpu_parity` runs deterministic serial and multithreaded simulations using the
same grid, emitter, seed, timestep, and step count. It compares the density and
velocity fields using maximum absolute error and RMS error, with a current
maximum-error tolerance of `1e-6`.

### Projection sanity

`projection_test` creates a known divergent velocity field, measures RMS and
maximum absolute divergence, performs projection, and measures the field
again. It also checks that projection leaves an initially zero field unchanged.
Both cases run through the serial and multithreaded paths.

Build and run all registered tests:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug
ctest --test-dir build/debug --output-on-failure
```

Run one test with full output:

```bash
ctest --test-dir build/debug -R '^cpu_parity$' -V
ctest --test-dir build/debug -R '^projection_test$' -V
```

### ThreadSanitizer

For supported Clang, AppleClang, or GCC configurations:

```bash
cmake -S . -B build/tsan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_TSAN=ON
cmake --build build/tsan
ctest --test-dir build/tsan --output-on-failure
```

## Known limitations

-   Pressure is solved with a fixed budget of 20 Jacobi iterations. This is
    inexpensive per iteration and straightforward to parallelize, but it
    converges slowly on smooth, low frequency divergence. I'm planning a faster pressure solver for a later version
-   The solver is currently two-dimensional and CPU-only.
-   CPU workers are currently created and joined for each parallel stage rather
    than being managed by a persistent thread pool.
-   The current validation suite checks CPU parity and projection behavior but is
    not yet a comprehensive numerical-convergence suite.
-   Performance measurements are displayed live, but a reproducible benchmark
    table across grid sizes and hardware is still yet to be added.

## Roadmap

-   [x] CPU simulation pipeline
-   [x] RGB density transport and interactive controls
-   [x] Structure-of-arrays field storage
-   [x] Serial and multithreaded kernels
-   [x] Per-stage profiling
-   [x] CPU numerical parity testing
-   [x] Projection sanity testing
-   [ ] Record a reproducible CPU-v1 performance baseline
-   [ ] Port simulation fields and an initial kernel to Metal
-   [ ] Validate CPU/GPU numerical agreement
-   [ ] Move the complete simulation pipeline to GPU compute
-   [ ] Investigate faster pressure solvers such as conjugate gradient or
        multigrid

## Project structure

```text
.
├── assets/          Runtime fonts and assets
├── include/         Grid, renderer, emitter, slider, and noise headers
├── src/             Simulation, application, and rendering implementation
├── Tests/           CPU parity and projection tests
└── CMakeLists.txt   Build, test, warning, and sanitizer configuration
```

## Project status

CPU v1 is feature-complete enough to serve as the numerical and performance
baseline for the next phase: a Metal compute implementation. The current focus
is preserving the same simulation behavior while moving field storage and
individual kernels onto the GPU.

## Grabbed the simplex noise implementartion for metal from here:

https://github.com/JoshuaSullivan/SimplexNoiseFilter/blob/091607969586d33f491fe29fca91802ef9ea50db/Sources/SimplexNoiseCompute/kernel/SimplexNoiseCompute.metal
