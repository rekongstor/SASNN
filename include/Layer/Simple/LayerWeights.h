#pragma once


#include <memory>
#include "../../Core/Random.h"
#include "../Abstract/LayerDynamic.h"
#include "../LayerDecorators/DecoratorGradientDescent.h"
#include "../LayerDecorators/DecoratorInitializer.h"

class LayerWeights : public LayerDynamic {
    void backProp() override;
    void followProp() override;
    void assignData(const Matrix2D *d) override;
    void subGrad(f32 step) override;
    void clearGrad() override;
    std::shared_ptr<DecoratorGradientDescent> gradientDescent;
    std::shared_ptr<DecoratorInitializer> initializer;
public:
    LayerWeights(size_t rows, size_t cols, DecoratorInitializer* decoratorInitializer, DecoratorGradientDescent* decoratorGradientDescent);
};



