#include "../../../../include/Layer/Simple/LayerWeightsDecorators/Initializer/InitializerZero.h"

void InitializerZero::Initialize(Matrix2D &weights) {
    weights.Clean();
}
