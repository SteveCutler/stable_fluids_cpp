#include "Grid.hpp"
#include <iostream>

// Test access keeps boundary handling out of the public simulation API.
struct ScalarBoundaryTestAccess {
    static void apply(Grid& grid, std::vector<float>& field) {
        grid.scalarBoundaries(field);
    }
};

int main() {
    Grid grid(5, 5, {}, 42, false);
    std::vector<float> field(25, 0.f);
    field[3 * 5 + 1] = 2.f;
    field[3 * 5 + 3] = 10.f;
    ScalarBoundaryTestAccess::apply(grid, field);
    // Bottom/right edges both inherit the adjacent interior value (10).
    // The old index used the opposite bottom edge and produced 6 instead.
    if (field[24] != 10.f || field[23] != 10.f || field[19] != 10.f) {
        std::cerr << "Bottom-right scalar boundary used a nonadjacent cell\n";
        return 1;
    }
    return 0;
}
