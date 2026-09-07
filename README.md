# Stable Fluids C++

An interactive **2D fluid simulation in C++20**, built from first principles and rendered with SFML 3. The completed CPU implementation combines **structure-of-arrays storage, serial and multithreaded execution, per-stage instrumentation, and numerical regression tests**.

A later **experimental Metal implementation** extends the project to GPU compute; its execution and numerical behavior have not been validated against the CPU baseline.

<img width="640" alt="RGB density mixing in the CPU fluid simulation; historical screenshot" src="https://github.com/user-attachments/assets/d2bd0658-17db-4171-81f7-a921710ac069" />

## Simulation and data layout

- Semi-Lagrangian advection with bilinear sampling for velocity and RGB density.
- Explicit velocity diffusion, iterative density diffusion, and Jacobi pressure projection.
- Buoyancy, configurable emitters, and OpenSimplex2-driven forcing.
- Contiguous field and scratch buffers, row-wise kernels, and buffer swaps between stages.

Each update applies noise-driven forcing, advects and diffuses velocity, projects pressure, injects density, diffuses and advects RGB density, then generates RGBA pixels. `Grid` owns the simulation fields; `Renderer` uploads pixels to an SFML texture and draws the controls, timings, and optional velocity vectors.

## CPU execution and instrumentation

Serial and threaded modes use the same numerical kernels. Threaded mode partitions rows across four workers and joins before boundary updates and buffer swaps.

The overlay reports stage times in milliseconds. Projection includes divergence, the pressure solve, gradient subtraction, and boundaries; its separately displayed substeps overlap that total. “Render prep” covers overlay/arrow preparation and texture upload, excluding draw calls and presentation.

## Numerical regression tests

| Test | Coverage |
| --- | --- |
| `cpu_parity` | 100 deterministic steps on a 64×64 grid; all RGB density and U/V velocity fields must be finite and agree within `1e-6` maximum absolute error. |
| `projection_test` | Reduced divergence for a smooth test field and unchanged zero-field divergence, in both execution modes. |
| `scalar_boundary_test` | Adjacent-cell handling at the bottom-right scalar boundary. |

CMake/CTest and optional ThreadSanitizer support the CPU validation workflow. These checks establish regression coverage, not physical-accuracy or convergence guarantees.

## Build and run

Requires a C++20 compiler, CMake 3.16+, and **SFML 3** (Graphics, Window, System). Verified on macOS with AppleClang 15 and SFML 3.0.2.

```bash
# macOS dependencies, if needed
brew install cmake sfml

cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j 4
ctest --test-dir build/debug --output-on-failure
./build/debug/StableFluidsC++

cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j 4
ctest --test-dir build/release --output-on-failure
```

The bundled font is located through a CMake-configured path, so build-tree executables can launch from another directory. Reconfigure after moving the checkout. Use `-DBUILD_TESTING=OFF` to omit tests.

For a supported ThreadSanitizer toolchain:

```bash
cmake -S . -B build/tsan -DCMAKE_BUILD_TYPE=Debug -DENABLE_TSAN=ON
cmake --build build/tsan -j 4
ctest --test-dir build/tsan --output-on-failure
```

## CPU controls

The default application uses a 1024×1024 grid.

| Input | Action |
| --- | --- |
| `P` | Pause/resume |
| `R` | Reset density, velocity, pressure, and divergence |
| `V` | Toggle velocity vectors |
| `M` | Toggle serial/threaded execution |
| Mouse | Adjust buoyancy, diffusion, decay, noise strength/frequency |

## Known limitations

- Pressure projection uses a fixed budget of 20 Jacobi iterations.
- A fixed `1/15` simulation timestep is taken per rendered update, so wall-clock simulation speed depends on throughput.
- Workers are created and joined for each parallel stage/iteration.
- Metal remains experimental, with unresolved boundary corner races and no CPU/GPU parity validation.


## Experimental Metal

```bash
cmake -S . -B build/metal -DCMAKE_BUILD_TYPE=Debug -DBUILD_METAL=ON
cmake --build build/metal -j 4
ctest --test-dir build/metal --output-on-failure
./build/metal/StableFluidsMetal
```

`BUILD_METAL` defaults to `OFF`. It requires Apple frameworks and Xcode Metal command-line tools, and adds a separate executable while retaining the CPU application. CMake builds shaders from the checked-in sources and supplies the metallib path; reconfigure/rebuild after moving the build tree.

Metal includes shared buffers, simulation kernels, SFML texture upload, and GPU/wait instrumentation. Its noise implementation and parameter defaults differ from CPU. An optional ownership test checks initialization and teardown without simulation dispatch; it skips device-dependent checks when no Metal device is available. CPU TSan does not validate GPU kernels.

## Project structure

```text
assets/fonts/             UI font, license, and provenance
include/, src/            CPU solver, rendering, emitters, and UI
include/metal/, src/metal/ Experimental Metal implementation
external/metal-cpp-main/  Vendored Apple interface and license
tests/                    CPU regression and Metal ownership tests
docs/                     Screenshots and third-party notices
CMakeLists.txt            Build, shaders, warnings, tests, and TSan
```

## Acknowledgements / Third-party code

- **FastNoiseLite — Jordan Peck and contributors:** CPU OpenSimplex2 noise; [embedded MIT notice](include/FastNoiseLite.h).
- **SFML:** rendering, window/input, and timing support; [upstream project](https://www.sfml-dev.org/), used as an external dependency.
- **Apple metal-cpp:** C++ interface to Metal; [retained Apache-2.0 license](external/metal-cpp-main/LICENSE.txt).
- **Joshua Sullivan's SimplexNoiseFilter:** source of the adapted Metal simplex code; [pinned upstream source](https://github.com/JoshuaSullivan/SimplexNoiseFilter/blob/091607969586d33f491fe29fca91802ef9ea50db/Sources/SimplexNoiseCompute/kernel/SimplexNoiseCompute.metal) and [retained MIT notice](docs/third-party/SimplexNoiseFilter-LICENSE.txt).
- **Roboto Mono Project Authors:** bundled UI font; [SIL Open Font License 1.1](assets/fonts/OFL-RobotoMono.txt) and [pinned provenance](assets/fonts/README.md).

See [third-party provenance](docs/third-party/README.md) for details. These notices cover their respective components; no license has been selected for the original project code.
