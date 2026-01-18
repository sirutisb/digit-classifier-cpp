#pragma once
#include <cstdint>

using Image = float[28][28];

struct LabeledImage {
    Image image;
    uint8_t label;
};
