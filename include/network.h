#pragma once
#include "image.h"
#include <vector>

void train_model(
    const std::vector<LabeledImage>& train, int epochs, int batch_size, float learning_rate,
    std::vector<float>& w1, std::vector<float>& w2, std::vector<float>& w3,
    std::vector<float>& b1, std::vector<float>& b2, std::vector<float>& b3
);