//
// Created by lenovo on 2025/3/10.
//

#ifndef PM_FFT_MAIN03_H
#define PM_FFT_MAIN03_H

#endif //PM_FFT_MAIN03_H
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

class FFTBase {
public:
    virtual ~FFTBase() = default;

    // 重载 transform 方法，支持 n 点 DFT 和指定维度
    virtual Eigen::VectorXcd transform(const Eigen::VectorXd& X, int n, int dim) = 0;
    virtual Eigen::MatrixXcd transform(const Eigen::MatrixXd& X, int n, int dim) = 0;
};


//派生类VectorFFT
class VectorFFT : public FFTBase {
public:
    Eigen::VectorXcd transform(const Eigen::VectorXd& X, int n, int dim) override {
        if (dim != 1) {
            throw std::invalid_argument("VectorFFT does not support dim parameter.");
        }
        Eigen::FFT<double> fft;
        Eigen::VectorXd x_processed = processInput(X, n);  // 处理输入数据
        Eigen::VectorXcd y(n);
        fft.fwd(y, x_processed);  // 执行 n 点 FFT
        return y;
    }

    // 对矩阵输入抛出异常
    Eigen::MatrixXcd transform(const Eigen::MatrixXd& X, int n, int dim) override {
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
    Eigen::VectorXcd transform(const Eigen::VectorXd& X, int n, int dim) override {
        throw std::invalid_argument("MatrixFFT can only process matrix inputs, not vectors.");
    }

    Eigen::MatrixXcd transform(const Eigen::MatrixXd& X, int n, int dim) override {
        if (dim < 1 || dim > 2) {
            throw std::invalid_argument("MatrixFFT only supports dim = 1 (columns) or dim = 2 (rows).");
        }

        Eigen::FFT<double> fft;
        Eigen::MatrixXcd Y;

        if (dim == 1) {
            // 沿列变换（默认行为）
            Y.resize(n, X.cols());
            for (int col = 0; col < X.cols(); ++col) {
                Eigen::VectorXd x_col = X.col(col);  // 提取列
                Eigen::VectorXd x_processed = processInput(x_col, n);  // 处理列数据
                Eigen::VectorXcd y_col(n);
                fft.fwd(y_col, x_processed);  // 执行 n 点 FFT
                Y.col(col) = y_col;
            }
        } else if (dim == 2) {
            // 沿行变换
            Y.resize(X.rows(), n);
            for (int row = 0; row < X.rows(); ++row) {
                Eigen::VectorXd x_row = X.row(row);  // 提取行
                Eigen::VectorXd x_processed = processInput(x_row, n);  // 处理行数据
                Eigen::VectorXcd y_row(n);
                fft.fwd(y_row, x_processed);  // 执行 n 点 FFT
                Y.row(row) = y_row;
            }
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

