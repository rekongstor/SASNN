#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerExp : public LayerDynamic {
    Layer &left;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerExp(Layer &left);
};



