#pragma once
#include <cstdint>

static constexpr std::size_t IMAGE_SIZE = 28*28;
using Image = float[IMAGE_SIZE];
struct LabeledImage {
    Image image;
    uint8_t label;
};