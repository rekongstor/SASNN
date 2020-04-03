#pragma once


#include <map>
#include "../../Core/Matrix2D.h"

class Layer {
protected:
     Matrix2D *self;
    explicit Layer(Matrix2D &data);
    virtual void transposeData() = 0;
public:
    virtual void followProp() = 0;
    virtual void backProp() = 0;
    virtual void clearGrad() = 0;
    virtual void assignData(Matrix2D *d) = 0;
    [[nodiscard]]  Matrix2D &getData() ;
    virtual Matrix2D *getGrad();
    virtual void subGrad(f32 step);
    virtual ~Layer() = default;
};

