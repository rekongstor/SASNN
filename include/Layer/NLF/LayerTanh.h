#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerTanh : public LayerDynamic {
    Layer &left;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerTanh(Layer &left);
};



