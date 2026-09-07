#include "MetalContext.hpp"
#include "MetalGrid.hpp"
#include <type_traits>
#include <iostream>

static_assert(!std::is_copy_constructible_v<MetalContext>);
static_assert(!std::is_copy_assignable_v<MetalContext>);
static_assert(!std::is_copy_constructible_v<MetalGrid>);
static_assert(!std::is_copy_assignable_v<MetalGrid>);

int main(int argc, char** argv) {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    // Missing library (or unavailable device) must leave safe, invalid objects.
    MetalContext context("/nonexistent/portfolio-missing-library.metallib");
    if (context.isValid()) return 1;
    MetalGrid grid(4, 4, 16, 16 * sizeof(float), {}, 42, context);
    if (grid.isValid()) return 1;
    grid.update(1.f / 15.f);
    grid.ClearBuffers();
    if (!grid.get_pixels().empty()) return 1;
    if (!context.get_device()) {
        std::cout << "No-device cleanup passed; device-dependent ownership checks skipped\n";
        return 77;
    }
    if (argc != 2) return 1;
    MetalContext validContext(argv[1]);
    if (!validContext.isValid()) return 1;
    if (validContext.CreatePipelineState("portfolio_missing_kernel")) return 1;
    // Allocate every pipeline and buffer, then destroy them without dispatching.
    // Draining the pool after these scopes also checks autoreleased string ownership.
    for (int iteration = 0; iteration < 2; ++iteration) {
        MetalGrid initializedGrid(4, 4, 16, 16 * sizeof(float), {}, 42, validContext);
        if (!initializedGrid.isValid() || initializedGrid.get_pixels().size() != 64) return 1;
    }
    return 0;
}
