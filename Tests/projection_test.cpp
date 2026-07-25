#include "Grid.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string_view>

namespace{

//test dimesnions
constexpr std::size_t width = 64;
constexpr std::size_t height = 64;
constexpr std::size_t seed = 42;

//remaining divergence threshold
constexpr double maximumRemainingRatio = 0.25;


std::vector<Emitter*> emitters;

//test function
bool testDivergentField(bool thread){
    Grid grid(width, height, emitters, seed, thread);

    for (std::size_t y = 1; y < height - 1; ++y) {
        for (std::size_t x = 1; x < width - 1; ++x) {
            const float normalizedX =
                static_cast<float>(x) /
                static_cast<float>(width - 1);

            const float normalizedY =
                static_cast<float>(y) /
                static_cast<float>(height - 1);

            const float u = std::sin(
                2.0f *
                std::numbers::pi_v<float> *
                normalizedX
            );

            const float v = std::sin(
                2.0f *
                std::numbers::pi_v<float> *
                normalizedY
            );

            grid.setVelocityAt(x, y, u, v);
        }
    }

    const DivergenceStats before =
        grid.measureDivergence();

    grid.projectStep();

    const DivergenceStats after =
        grid.measureDivergence();
        
    const double remainingRatio =
    //divide by zero check
        before.rms > 0.0
        //calculate ratio between before after divergence values
            ? after.rms / before.rms
            : 1.0;

    //check if tests pass
     const bool passed =
        before.allFinite &&
        after.allFinite &&
        before.rms > 1.0e-8 &&
        remainingRatio <= maximumRemainingRatio;

    std::cout
        << "\nProjection sanity — "
        << (grid.m_mult_threaded ? "threaded" : "serial")
        << '\n'
        << std::scientific
        << "RMS before:      " << before.rms << '\n'
        << "RMS after:       " << after.rms << '\n'
        << "Max before:      "
        << before.maximumAbsolute << '\n'
        << "Max after:       "
        << after.maximumAbsolute << '\n'
        << "Remaining ratio: " << remainingRatio << '\n'
        << "Result:          "
        << (passed ? "PASS" : "FAIL") << '\n';

    return passed;

    };

    bool testZeroField(bool threaded)
{
    Grid grid(width, height, emitters, seed, true);

    const DivergenceStats before =
        grid.measureDivergence();

    grid.projectStep();

    const DivergenceStats after =
        grid.measureDivergence();

    constexpr double zeroTolerance = 1.0e-8;

    const bool passed =
        before.allFinite &&
        after.allFinite &&
        before.rms <= zeroTolerance &&
        after.rms <= zeroTolerance &&
        after.maximumAbsolute <= zeroTolerance;

    std::cout
        << "\nZero-field control — "
        << (threaded ? "threaded" : "serial")
        << '\n'
        << std::scientific
        << "RMS before: " << before.rms << '\n'
        << "RMS after:  " << after.rms << '\n'
        << "Result:     "
        << (passed ? "PASS" : "FAIL") << '\n';

    return passed;
    }
} // end of namespace

int main()
{
    bool allPassed = true;

    allPassed &= testDivergentField(false);
    allPassed &= testDivergentField(true);
    allPassed &= testZeroField(false);
    allPassed &= testZeroField(true);

    return allPassed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}



    
