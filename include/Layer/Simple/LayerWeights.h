#pragma once


#include <memory>
#include "../../Core/Random.h"
#include "../Abstract/LayerDynamic.h"
#include "../../../src/Layer/Simple/LayerWeightsDecorators/DecoratorGradientDescent.h"
#include "../../../src/Layer/Simple/LayerWeightsDecorators/DecoratorInitializer.h"

class LayerWeights : public LayerDynamic {
    void backProp() override;
    void followProp() override;
    void assignData(const Matrix2D *d) override;
    void subGrad(f32 step) override;
    void clearGrad() override;
    Matrix2D gradLength;
    std::shared_ptr<DecoratorGradientDescent> gradientDescent;
public:
    LayerWeights(size_t rows, size_t cols, DecoratorInitializer* decoratorInitializer, DecoratorGradientDescent* decoratorGradientDescent);
};



