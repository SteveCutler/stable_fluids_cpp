#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include "Grid.hpp"
#include "Emitter.hpp"
#include <iostream>

struct ErrorStats {
    //using double to avoid rounding errors
    double maximumAbsoluteError{0.0};
    double rmsError{0.0};
    std::size_t maximumErrorIndex{0};
    bool allFinite{true};
};

//compareFields helper function
ErrorStats compareFields(
    const std::vector<float>& reference,
    const std::vector<float>& candidate
)
{
    //first check: simple size check to see if the fields have the same number of elements
    if (reference.size() != candidate.size()) {
        throw std::invalid_argument(
            "Fields have different sizes"
        );
    }

    //initializing error variables
    ErrorStats stats;
    double squaredErrorSum = 0.0;

    //begin loop to compare values in the grids
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float a = reference[i];
        const float b = candidate[i];

        //check that every value is a usable number
        if (!std::isfinite(a) || !std::isfinite(b)) {
            stats.allFinite = false;
            continue;
        }

        //calculate error using doubles as it has more accuracy
        const double error =
            std::abs(static_cast<double>(a) -
                     static_cast<double>(b));

        //square it and add to sum
        squaredErrorSum += error * error;

        //compare it against max abs error and store if greater
        if (error > stats.maximumAbsoluteError) {
            stats.maximumAbsoluteError = error;
            //record index of greatest error
            stats.maximumErrorIndex = i;
        }
    }

    //check size to avoid dividing by 0
    if (!reference.empty()) {
        //sqrt to get rms error
        stats.rmsError = std::sqrt(
            squaredErrorSum /
            static_cast<double>(reference.size())
        );
    }

    return stats;
}


//main test function
int main()
{

    //test constexpr variables
    constexpr unsigned int size = 64;
    constexpr std::size_t stepCount = 100;
    constexpr float dt = 1.0f / 15.0f;
    constexpr int seed = 42;
    constexpr bool serial_thread = false;
    constexpr bool multi_thread = true;
    
    //error tolerance for comparison
    constexpr double tolerance = 1.0e-6;

    //test emitters
    //emitter size relative to size of grid
    Emitter serialEmitter{sf::Vector2f(size*.5f,size*.5f), size*.1f, sf::Vector3f(1.f,1.f,.1f)};
    Emitter threadedEmitter{sf::Vector2f(size*.5f,size*.5f), size*.1f, sf::Vector3f(1.f,1.f,.1f)};

    std::vector<Emitter*> serialEmitters{
        &serialEmitter
    };
  
    std::vector<Emitter*> threadedEmitters{
        &threadedEmitter
    };


    //test grids intialization
    Grid serialGrid(size, size, serialEmitters, seed, serial_thread);
    Grid threadedGrid(size, size, threadedEmitters, seed, multi_thread);


    //run for set step count
    for (std::size_t step = 0; step < stepCount; ++step) {
        serialGrid.update(dt);
        threadedGrid.update(dt);
    }

    //take results and compare fields
    const ErrorStats densityRed = compareFields(
        serialGrid.density_r(),
        threadedGrid.density_r());

    const ErrorStats velocityU = compareFields(
        serialGrid.u_velocity(),
        threadedGrid.u_velocity());

    const ErrorStats velocityV = compareFields(
        serialGrid.v_velocity(),
        threadedGrid.v_velocity());

    
    //check if tests pass
    const bool passed =
        densityRed.allFinite &&
        velocityU.allFinite &&
        velocityV.allFinite &&
        densityRed.maximumAbsoluteError <= tolerance &&
        velocityU.maximumAbsoluteError <= tolerance &&
        velocityV.maximumAbsoluteError <= tolerance;

    //output results
    std::cout
        << "Density R: max=" << densityRed.maximumAbsoluteError
        << ", rms=" << densityRed.rmsError << '\n';

    std::cout
        << "Velocity U: max=" << velocityU.maximumAbsoluteError
        << ", rms=" << velocityU.rmsError << '\n';

    std::cout
        << "Velocity V: max=" << velocityV.maximumAbsoluteError
        << ", rms=" << velocityV.rmsError << '\n';

    if (!passed) {
        std::cerr << "CPU parity test failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "CPU parity test passed\n";
    return EXIT_SUCCESS;
}