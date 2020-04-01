#include "../../../../../include/Layer/Simple/LayerWeightsDecorators/Initializer/InitializerUniform.h"
#include "../../../../../include/Core/Random.h"

void InitializerUniform::Initialize(Matrix2D &weights) {
    RandomUniform randomUniform(a, b);
    Random *rng = &randomUniform;
    for (size_t i = 0; i < weights.getRows(); ++i)
        for (size_t j = 0; j < weights.getCols(); ++j)
            weights.setCell(i, j, rng->Next());
}

InitializerUniform::InitializerUniform(f32 a, f32 b) : a(a),
                                                       b(b) {}
