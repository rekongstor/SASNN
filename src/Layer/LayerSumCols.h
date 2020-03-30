#pragma once


#include "../../include/Layer/Abstract/LayerDynamic.h"

class LayerSumCols : public LayerDynamic {
public:
    explicit LayerSumCols(Layer &left);
private:
    void followProp() override;
    void backProp() override;
    Layer &left;
};



