#pragma once


#include "../../Core/Matrix2D.h"

class Layer {
protected:
    const Matrix2D &self;
    explicit Layer(const Matrix2D &data);
public:
    virtual void followProp() = 0;
    virtual void backProp() = 0;
    virtual void clearGrad() = 0;
    [[nodiscard]] const Matrix2D &getData() const;
    virtual Matrix2D *getGrad();
    virtual ~Layer() = default;
};

