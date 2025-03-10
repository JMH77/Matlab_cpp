#ifndef MAIN01_H
#define MAIN01_H

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

// 基类：定义统一的接口
class FFTBase {
public:
    virtual ~FFTBase() = default;

    // 重载 transform 方法，分别支持向量和矩阵
    virtual Eigen::VectorXcd transform(const Eigen::VectorXd& X) = 0;
    virtual Eigen::MatrixXcd transform(const Eigen::MatrixXd& X) = 0;
};

// 派生类：处理向量的 FFT
class VectorFFT : public FFTBase {
public:
    Eigen::VectorXcd transform(const Eigen::VectorXd& X) override {
        Eigen::FFT<double> fft;
        Eigen::VectorXcd y(X.size());
        fft.fwd(y, X);  // 执行 FFT
        return y;
    }

    // 对矩阵输入抛出异常
    Eigen::MatrixXcd transform(const Eigen::MatrixXd& X) override {
        throw std::invalid_argument("VectorFFT can only process vector inputs, not matrices.");
    }
};

// 派生类：处理矩阵的 FFT（对每列进行 FFT）
class MatrixFFT : public FFTBase {
public:
    Eigen::VectorXcd transform(const Eigen::VectorXd& X) override {
        throw std::invalid_argument("MatrixFFT can only process matrix inputs, not vectors.");
    }

    Eigen::MatrixXcd transform(const Eigen::MatrixXd& X) override {
        Eigen::FFT<double> fft;
        Eigen::MatrixXcd Y(X.rows(), X.cols());
        for (int col = 0; col < X.cols(); ++col) {
            Eigen::VectorXd x_col = X.col(col);  // 提取列
            Eigen::VectorXcd y_col;
            fft.fwd(y_col, x_col);  // 执行 FFT
            Y.col(col) = y_col;
        }
        return Y;
    }
};

#endif // MAIN01_H

