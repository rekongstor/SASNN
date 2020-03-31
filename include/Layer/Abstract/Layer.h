#pragma once


#include "../../Core/Matrix2D.h"

class Layer {
protected:
    const Matrix2D *self;
    explicit Layer(const Matrix2D &data);
    virtual void transposeData() = 0;
public:
    virtual void followProp() = 0;
    virtual void backProp() = 0;
    virtual void clearGrad() = 0;
    virtual void assignData(const Matrix2D *) = 0;
    [[nodiscard]] const Matrix2D &getData() const;
    virtual Matrix2D *getGrad();
    virtual void subGrad();
    virtual ~Layer() = default;
};

