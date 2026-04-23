#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

static float sigmoid(float z) { return 1 / (1 + expf(-z)); }
static void sigmoid_all(std::vector<float>& vec) {
    for (auto& v : vec) v = sigmoid(v);
}

inline float relu(float z) { return z > 0.0f ? z : 0.0f; }
inline float relu_derivative(float a) { return a > 0.0f ? 1.0f : 0.0f; }

static void softmax_all(std::vector<float>& vec) {
    float max = *std::max_element(vec.begin(), vec.end());
    float total = 0.0f;
    for (const auto& v : vec) total += expf(v - max);
    for (auto& v : vec) v = expf(v - max) / total;
}