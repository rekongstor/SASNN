#pragma once


#include "../../../../include/Core/Matrix2D.h"

class DecoratorInitializer {
public:
    virtual void Initialize(Matrix2D &weights) = 0;
};



