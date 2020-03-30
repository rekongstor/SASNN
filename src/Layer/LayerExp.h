#pragma once


#include "../../include/Layer/Abstract/LayerDynamic.h"

class LayerExp : LayerDynamic {
    Layer &left;
public:
    explicit LayerExp(Layer &left);
private:
    void followProp() override;
    void backProp() override;

};



