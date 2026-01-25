#include "image.h"
#include "loader.h"
#include "network.h"
#include "progress.h"
#include "activations.h"
#include <cmath>
#include <print>
#include <random>
#include <vector>

std::vector<float> w1, w2, w3;
std::vector<float> b1, b2, b3;

std::vector<float> forwardPass(const Image& image) {
    std::vector<float> activations(image, image + IMAGE_SIZE);
    std::vector<float> next;

    // 1st hidden layer
    next = std::vector<float>(16, 0);
    for (int neuron = 0; neuron < next.size(); ++neuron) {
        float z = b1[neuron];
        for (int i = 0; i < activations.size(); ++i) {
            z += w1[neuron * activations.size() + i] * activations[i];
        }
        next[neuron] = z;
    }
    activations = std::move(next);
    sigmoid_all(activations);

    // 2nd hidden layer
    next = std::vector<float>(16, 0);
    for (int neuron = 0; neuron < next.size(); ++neuron) {
        float z = b2[neuron];
        for (int i = 0; i < activations.size(); ++i) {
            z += w2[neuron * activations.size() + i] * activations[i];
        }
        next[neuron] = z;
    }
    activations = std::move(next);
    sigmoid_all(activations);

    // output layer
    next = std::vector<float>(10, 0);
    for (int neuron = 0; neuron < next.size(); ++neuron) {
        float z = b3[neuron];
        for (int i = 0; i < activations.size(); ++i) {
            z += w3[neuron * activations.size() + i] * activations[i];
        }
        next[neuron] = z;
    }
    activations = std::move(next);
    sigmoid_all(activations);

    return activations;
}

void draw_mnist_digit(const Image& image) {
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            float num = image[x + y * 28];
            uint32_t color = 232 + (uint32_t)(num * 24);
            printf("\x1b[48;5;%dm  ", color);
        }
        printf("\n");
    }
    printf("\x1b[0m");
}

void test_model(const std::vector<LabeledImage>& test) {
    std::println("Testing model with {} images...", test.size());
    std::vector<std::vector<int>> cm(10, std::vector<int>(10, 0));

    int correct = 0, tested = 0;
    for (const auto& sample : test) {
        auto out = forwardPass(sample.image);
        int argmax = 0;
        float valmax = out[0];
        for (int i = 1; i < 10; i++) {
            if (out[i] > valmax) {
                valmax = out[i];
                argmax = i;
            }
        }
        uint8_t label = sample.label;
        uint8_t pred = argmax;

        cm[pred][label]++;
        if (pred == label) correct++;
        tested++;

        uint32_t chunks = test.size() < 100 ? 1 : test.size() / 100;
        if (tested % chunks == 0) {
            // draw_mnist_digit(sample.image);
            // std::println("Predicted: {} | Label: {}", pred, label);
            updateProgress((float)tested / test.size());
            // std::this_thread::sleep_for(std::chrono::milliseconds(500));
            // std::println("Label: {} | Predicted: {} | Accuracy: {}/{} ({}%)", label, pred, correct, tested, (float)(correct*100)/tested);
        }
    }

    std::println("Confusion Matrix:");
    for (int r = 0; r < 10; ++r) {
        for (int c = 0; c < 10; ++c) {
            std::print("{} \t", cm[r][c]);
        }
        std::println("\n\n");
    }
    std::println("Accuracy: {}/{} ({}%)", correct, tested, (float)(correct*100)/tested);
}

void init_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());

    // // Setup layers
    w1.resize(784 * 16);
    w2.resize(16 * 16);
    w3.resize(16 * 10);

    b1.assign(16, 0.0f);
    b2.assign(16, 0.0f);
    b3.assign(10, 0.0f);

    auto xavier_init = [&](float input_size) {
        return std::normal_distribution<float>(0.0, sqrtf(1.0f / input_size)); // Standard Xavier for Sigmoid
    };

    auto dist1 = xavier_init(784.0f);
    auto dist2 = xavier_init(16.0f);
    auto dist3 = xavier_init(16.0f);
    for (auto& w : w1) w = dist1(gen);
    for (auto& w : w2) w = dist2(gen);
    for (auto& w : w3) w = dist3(gen);
}

int main() {
    auto [train, test] = load_train_test(60000, 10000);
    init_weights();

    train_model(train, 8, 0.01f, w1, w2, w3, b1, b2, b3);

    test_model(test);
    return 0;
}
