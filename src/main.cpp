#include "image.h"
#include "loader.h"
#include "model.h"
#include <print>

int main() {
    std::println("Loading dataset...");
    std::vector<LabeledImage> train = loadImages(
        "../dataset/train-images.idx3-ubyte",
        "../dataset/train-labels.idx1-ubyte",
        5 // temporary, to not fetch all training samples
    );

    Model model{
        {
            {784, Activation::None},    // Input Layer
            {16, Activation::ReLu},     // Hidden 1
            {16, Activation::ReLu},     // Hidden 2
            {10, Activation::Softmax}   // Output Layer
        },
        0.001f // Learning Rate
    };

    model.fit(train);

    Image& testImage = train[1].image; // temporary
    model.predict(testImage);
    return 0;
}
