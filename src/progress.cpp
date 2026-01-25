#include "progress.h"
#include <iostream>

// The progess must be called with 100% at the end or else the newline will not be added.
void updateProgress(float progress) {
    const int barWidth = 100;
    int pos = barWidth * progress;
    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
    if (progress >= 1.0) std::cout << std::endl;
}