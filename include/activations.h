#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

static float sigmoid(float x) { return 1 / (1 + expf(-x)); }
static void sigmoid_all(std::vector<float>& vec) {
    for (auto& v : vec) v = sigmoid(v);
}

static void softmax_all(std::vector<float>& vec) {
    float max = *std::max_element(vec.begin(), vec.end());
    float total = 0.0f;
    for (const auto& v : vec) total += expf(v - max);
    for (auto& v : vec) v = expf(v - max) / total;
}