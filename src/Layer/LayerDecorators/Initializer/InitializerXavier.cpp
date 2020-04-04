#include "../../../../include/Layer/Simple/LayerWeightsDecorators/Initializer/InitializerXavier.h"
#include "../../../../include/Core/Random.h"

void InitializerXavier::Initialize(Matrix2D &weights) {
    RandomGaussian randomGaussian = RandomGaussian(
            0.f,
            sqrtf(2.f / inputs));
    Random *rng = &randomGaussian;

    for (size_t i = 0; i < weights.getRows(); ++i)
        for (size_t j = 0; j < weights.getCols(); ++j)
            weights.setCell(i, j, rng->Next());
}

InitializerXavier::InitializerXavier(f32 inputs) : inputs(inputs) {}
