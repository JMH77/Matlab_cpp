//
// Created by lenovo on 2025/3/10.
//
#include "main03.h"

int main() {
    // 示例 1：向量傅里叶变换
    Eigen::VectorXd x(8);
    x << 0, 1, 2, 3, 4, 5, 6, 7;
    FFTBase* fft1 = new VectorFFT();
    Eigen::VectorXcd y = fft1->transform(x, 10, 1);  // 10 点 FFT
    std::cout << "FFT of vector x (n=10):\n" << y << "\n\n";
    delete fft1;

    // 示例 2：矩阵傅里叶变换（沿列）
    Eigen::MatrixXd X(8, 2);
    X << 0, 8,
            1, 7,
            2, 6,
            3, 5,
            4, 4,
            5, 3,
            6, 2,
            7, 1;
    FFTBase* fft2 = new MatrixFFT();
    Eigen::MatrixXcd Y1 = fft2->transform(X, 6, 1);  // 沿列 6 点 FFT
    std::cout << "FFT of matrix X (n=6, dim=1):\n" << Y1 << "\n\n";

    // 示例 3：矩阵傅里叶变换（沿行）
    Eigen::MatrixXcd Y2 = fft2->transform(X, 6, 2);  // 沿行 6 点 FFT
    std::cout << "FFT of matrix X (n=6, dim=2):\n" << Y2 << "\n\n";

    delete fft2;
    return 0;
}
