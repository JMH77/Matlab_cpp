//
// Created by lenovo on 2025/3/10.
//

#ifndef PM_FFT_MAIN02_H
#define PM_FFT_MAIN02_H

#endif //PM_FFT_MAIN02_H
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>


class FFTBase {
public:
    virtual ~FFTBase() = default;

    // 重载 transform 方法，支持 n 点 DFT
    virtual Eigen::VectorXcd transform(const Eigen::VectorXd& X, int n) = 0;
    virtual Eigen::MatrixXcd transform(const Eigen::MatrixXd& X, int n) = 0;
};

//派生类VectorFFT
class VectorFFT : public FFTBase {
public:
    Eigen::VectorXcd transform(const Eigen::VectorXd& X, int n) override {
        Eigen::FFT<double> fft;
        Eigen::VectorXd x_processed = processInput(X, n);  // 处理输入数据
        Eigen::VectorXcd y(n);
        fft.fwd(y, x_processed);  // 执行 n 点 FFT
        return y;
    }

    // 对矩阵输入抛出异常
    Eigen::MatrixXcd transform(const Eigen::MatrixXd& X, int n) override {
        throw std::invalid_argument("VectorFFT can only process vector inputs, not matrices.");
    }

private:
    // 处理输入数据：截断或补零
    Eigen::VectorXd processInput(const Eigen::VectorXd& X, int n) {
        if (X.size() < n) {
            // 补零
            Eigen::VectorXd padded = Eigen::VectorXd::Zero(n);
            padded.head(X.size()) = X;
            return padded;
        } else if (X.size() > n) {
            // 截断
            return X.head(n);
        } else {
            // 无需处理
            return X;
        }
    }
};


//派生类MatrixFFT
class MatrixFFT : public FFTBase {
public:
    Eigen::VectorXcd transform(const Eigen::VectorXd& X, int n) override {
        throw std::invalid_argument("MatrixFFT can only process matrix inputs, not vectors.");
    }

    Eigen::MatrixXcd transform(const Eigen::MatrixXd& X, int n) override {
        Eigen::FFT<double> fft;
        Eigen::MatrixXcd Y(n, X.cols());
        for (int col = 0; col < X.cols(); ++col) {
            Eigen::VectorXd x_col = X.col(col);  // 提取列
            Eigen::VectorXd x_processed = processInput(x_col, n);  // 处理列数据
            Eigen::VectorXcd y_col(n);
            fft.fwd(y_col, x_processed);  // 执行 n 点 FFT
            Y.col(col) = y_col;
        }
        return Y;
    }

private:
    // 处理输入数据：截断或补零
    Eigen::VectorXd processInput(const Eigen::VectorXd& X, int n) {
        if (X.size() < n) {
            // 补零
            Eigen::VectorXd padded = Eigen::VectorXd::Zero(n);
            padded.head(X.size()) = X;
            return padded;
        } else if (X.size() > n) {
            // 截断
            return X.head(n);
        } else {
            // 无需处理
            return X;
        }
    }
};

