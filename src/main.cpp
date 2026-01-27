#include "loader.h"
#include "model.h"
#include <print>

int main() {
    std::println("Creating model..");
    Model model{
        {784, Activation::None},
        {16, Activation::Sigmoid},
        {16, Activation::Sigmoid},
        {10, Activation::Sigmoid},
    };

    std::println("Loading dataset...");
    auto [train, test] = load_train_test(60000, 10000);

    std::println("Testing before training...");
    model.evaluate(test);

    std::println("Training Model...");
    model.fit(train, 8, 32, 1.0f);

    std::println("Testing after training...");
    model.evaluate(test);
    return 0;
}
