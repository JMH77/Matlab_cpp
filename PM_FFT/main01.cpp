#include "main01.h"

int main() {
    // 示例 1：向量傅里叶变换
    Eigen::VectorXd x(8);
    x << 0, 1, 2, 3, 4, 5, 6, 7;
    FFTBase* fft1 = new VectorFFT();
    Eigen::VectorXcd y = fft1->transform(x);
    std::cout << "FFT of vector x:\n" << y << "\n\n";
    delete fft1;

    // 示例 2：矩阵傅里叶变换
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
    Eigen::MatrixXcd Y = fft2->transform(X);
    std::cout << "FFT of matrix X:\n" << Y << "\n\n";
    delete fft2;

    return 0;
}

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.