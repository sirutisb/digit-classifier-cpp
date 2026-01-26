#include "image.h"
#include "loader.h"
#include "matrix.h"
#include "model.h"
#include "network.h"
#include "progress.h"
#include <print>
#include <vector>

std::vector<float> w1, w2, w3;
std::vector<float> b1, b2, b3;

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
        auto out = forwardPass(sample.image, w1, w2, w3, b1, b2, b3);
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

int main() {

    std::println("Making matrixes");
    Matrix m1(3, 3);
    Matrix m2(3, 3);

    std::println("Making set");
    for (int i = 1; i <= 3; i++) {
        for (int j = 1; j <= 3; ++j) {
            m1.get(i-1, j-1) = i * 3 + j;
            m2.get(i-1, j-1) = i * 3 + j;
        }
    }

    std::println("Mult");
    Matrix m3 = m1 * m2;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; ++j) {
            auto v = m3.get(i, j);
            std::print("{} ", v);
        }
        std::println();
    }

    return 1;
    auto [train, test] = load_train_test(6000, 2000);

    Model model{
        {784, Activation::None},
        {16, Activation::Sigmoid},
        {16, Activation::Sigmoid},
        {10, Activation::Sigmoid},
    };

    model.fit(train, 12, 64, 0.01f);
    model.evaluate(test);


    // train_model(train, 12, 64, 0.01f, w1, w2, w3, b1, b2, b3);

    // test_model(test);
    return 0;
}
