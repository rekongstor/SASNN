#pragma once


#include "../Core/Matrix2D.h"

class Layer {
protected:
    virtual const Matrix2D& getData() = 0;
public:
    virtual void followProp() = 0;
    virtual void backProp(Matrix2D &grad) = 0;
    virtual void clearGrad() = 0;
    virtual ~Layer()  = default;
};



