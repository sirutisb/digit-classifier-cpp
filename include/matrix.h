#pragma once
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>

// template <typename T>
// requires std::is_arithmetic_v<T>
using T = int;
class Matrix {
public:
    Matrix(size_t r, size_t c) : r_(r), c_(c), data_(std::make_unique<T[]>(r * c)) {
    }

    Matrix(const Matrix& other) : r_(other.r_), c_(other.c_), data_(std::make_unique<T[]>(other.r_ * other.c_)) {
        size_t n = r_ * c_;
        for (size_t i = 0; i < n; ++i) data_[i] = other.data_[i];
    }

    Matrix(Matrix&& other) : r_(other.r_), c_(other.c_), data_(std::move(other.data_)) {
        other.r_ = other.c_ = 0;
    }

    // Matrix& operator=(const Matrix& other) noexcept {

    // }

    Matrix& operator=(Matrix&& other) noexcept {
        r_ = other.r_;
        c_ = other.c_;
        data_ = std::move(other.data_);
        other.r_ = other.c_ = 0;
        return *this;
    }

    Matrix operator+(const Matrix& other) {
        if (r_ != other.r_ || c_ != other.c_) throw std::logic_error("Cannot add different sized matrices");
        Matrix res(r_, c_);
        size_t n = r_ * c_;
        for (size_t i = 0; i < n; ++i) res.data_[i] = data_[i] + other.data_[i];
        return res;
    }

    Matrix operator*(const Matrix& other) {
        if (c_ != other.r_) throw std::logic_error("Input matrices are wrong shape for multiplication");
        Matrix res(r_, other.c_);

        for (size_t i = 0; i < r_; ++i) {
            for (size_t j = 0; j < other.c_; ++j) {
                for (size_t k = 0; k < c_; ++k) {
                    res.get(i, j) += get(i, k) * other.get(k, j);
                }
            }
        }

        return res;
    }

    T& get(size_t r, size_t c) const { return data_[r * c_ + c]; }

    size_t r_, c_;
private:
    std::unique_ptr<T[]> data_; // row major
};