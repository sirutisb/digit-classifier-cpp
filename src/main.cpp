#include "image.h"
#include "loader.h"
#include "progress.h"
#include "activations.h"
#include <cstdlib>
#include <print>
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

void train_model(const std::vector<LabeledImage>& train) {
    const LabeledImage& sample = train[0];
    const Image& image = sample.image;

    std::vector<float> a0(image, image + IMAGE_SIZE);
    std::vector<float> z1(16), a1(16);
    std::vector<float> z2(16), a2(16);
    std::vector<float> z3(10), a3(10);

    // 1st hidden layer (16)
    for (int neuron = 0; neuron < a1.size(); ++neuron) {
        float z = b1[neuron];
        for (int i = 0; i < a0.size(); ++i) {
            z += w1[neuron * a0.size() + i] * a0[i];
        }
        float a = sigmoid(z);

        z1[neuron] = z;
        a1[neuron] = a;
    }

    // 2nd hidden layer (16)
    for (int neuron = 0; neuron < a2.size(); ++neuron) {
        float z = b2[neuron];
        for (int i = 0; i < a1.size(); ++i) {
            z += w2[neuron * a1.size() + i] * a1[i];
        }
        float a = sigmoid(z);
        z2[neuron] = z;
        a2[neuron] = a;
    }

    // output layer (10)
    for (int neuron = 0; neuron < a3.size(); ++neuron) {
        float z = b3[neuron];
        for (int i = 0; i < a2.size(); ++i) {
            z += w3[neuron * a2.size() + i] * a2[i];
        }
        float a = sigmoid(z);
        z3[neuron] = z;
        a3[neuron] = a;
    }

    // a3 is our output

    // compute deltas
    // error is dL/da, a - t
    // delta = dL/da * da/dz        This is shared by many neurons for the layers
    // dL/dw = dL/da * da/dz * dz/dw
    std::vector<float> d3(10, 0);
    for (int i = 0; i < 10; ++i) {
        float dL_da = a3[i] - (i == sample.label ? 1.0f : 0.0f);
        float da_dz = a3[i] * (1.0f - a3[i]); // sigmoid derivative
        d3[i] = dL_da * da_dz; // dL/dz
    }

    const float learning_rate = 0.01f;

    for (int neuron = 0; neuron < 10; ++neuron) {
        for (int i = 0; i < 16; i++) {
            float grad_w = d3[neuron] * a2[i];
            w3[neuron * 16 + i] -= grad_w * learning_rate;
        }
        b3[neuron] -= d3[neuron] * learning_rate;
    }


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

        if (tested % (test.size() / 100) == 0) {
            updateProgress((float)tested / test.size());
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



void setup_weights() {
    // // Setup layers
    w1.resize(784 * 16);
    w2.resize(16 * 16);
    w3.resize(16 * 10);
    for (auto& w : w1) w = static_cast<float>(rand()) / RAND_MAX;
    for (auto& w : w2) w = static_cast<float>(rand()) / RAND_MAX;
    for (auto& w : w3) w = static_cast<float>(rand()) / RAND_MAX;

    b1.resize(16, 0);
    b2.resize(16, 0);
    b3.resize(10, 0);

    // load trained
    w1 = load_floats("../test/0_weight.bin", w1.size());
    w2 = load_floats("../test/2_weight.bin", w2.size());
    w3 = load_floats("../test/4_weight.bin", w3.size());
    b1 = load_floats("../test/0_bias.bin", b1.size());
    b2 = load_floats("../test/2_bias.bin", b2.size());
    b3 = load_floats("../test/4_bias.bin", b3.size());
}

int main() {
    auto [train, test] = load_train_test(0);
    setup_weights();
    test_model(test);
    return 0;
}
