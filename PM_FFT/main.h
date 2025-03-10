//
// Created by lenovo on 2025/3/10.
//

#ifndef PM_FFT_H
#define PM_FFT_H

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <stdexcept>

// 基类：定义统一的接口
class FFTBase {
public:
    virtual ~FFTBase() = default;

    // 默认行为：不支持的操作抛出异常
    virtual Eigen::VectorXcd transform(const Eigen::VectorXd& X) {
        throw std::invalid_argument("This FFT implementation does not support this operation.");
    }

    virtual Eigen::MatrixXcd transform(const Eigen::MatrixXd& X) {
        throw std::invalid_argument("This FFT implementation does not support this operation.");
    }

    virtual Eigen::VectorXcd transform(const Eigen::VectorXd& X, int n) {
        throw std::invalid_argument("This FFT implementation does not support this operation.");
    }

    virtual Eigen::MatrixXcd transform(const Eigen::MatrixXd& X, int n) {
        throw std::invalid_argument("This FFT implementation does not support this operation.");
    }

    virtual Eigen::VectorXcd transform(const Eigen::VectorXd& X, int n, int dim) {
        throw std::invalid_argument("This FFT implementation does not support this operation.");
    }

    virtual Eigen::MatrixXcd transform(const Eigen::MatrixXd& X, int n, int dim) {
        throw std::invalid_argument("This FFT implementation does not support this operation.");
    }

protected:
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

// 派生类：处理向量的 FFT
class VectorFFT : public FFTBase {
public:
    // 向量 FFT（默认 n = X.size()）
    Eigen::VectorXcd transform(const Eigen::VectorXd& X) override {
        Eigen::FFT<double> fft;
        Eigen::VectorXcd y(X.size());
        fft.fwd(y, X);  // 执行 FFT
        return y;
    }

    // 向量 FFT（指定 n）
    Eigen::VectorXcd transform(const Eigen::VectorXd& X, int n) override {
        Eigen::FFT<double> fft;
        Eigen::VectorXd x_processed = processInput(X, n);  // 处理输入数据
        Eigen::VectorXcd y(n);
        fft.fwd(y, x_processed);  // 执行 n 点 FFT
        return y;
    }

    // 向量 FFT（指定 n 和 dim，dim 参数无效）
    Eigen::VectorXcd transform(const Eigen::VectorXd& X, int n, int dim) override {
        if (dim != 1) {
            throw std::invalid_argument("VectorFFT does not support dim parameter.");
        }
        return transform(X, n);  // 调用重载方法
    }
};

// 派生类：处理矩阵的 FFT
class MatrixFFT : public FFTBase {
public:
    // 矩阵 FFT（默认对每列进行 FFT，n = X.rows()）
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

    // 矩阵 FFT（指定 n）
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

    // 矩阵 FFT（指定 n 和 dim）
    Eigen::MatrixXcd transform(const Eigen::MatrixXd& X, int n, int dim) override {
        if (dim < 1 || dim > 2) {
            throw std::invalid_argument("MatrixFFT only supports dim = 1 (columns) or dim = 2 (rows).");
        }

        Eigen::FFT<double> fft;
        Eigen::MatrixXcd Y;

        if (dim == 1) {
            // 沿列变换
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
};

#endif // PM_FFT_H


/*
代码结构与优化说明:
1. 基类 FFTBase 的设计
统一接口：提供了所有可能的 transform 方法，默认抛出异常，防止派生类遗漏实现。
辅助函数：processInput 函数用于处理输入数据的补零和截断操作，避免代码重复。
2. 派生类 VectorFFT 和 MatrixFFT 的设计
方法重载：
VectorFFT 支持向量 FFT，忽略 dim 参数。
MatrixFFT 支持矩阵 FFT，支持 dim 参数（1 为列，2 为行）。
异常处理：
对不支持的操作抛出异常，提供清晰的错误提示。
3. 代码复用与扩展性
复用：processInput 函数被多次复用，减少了代码重复。
扩展性：可以通过添加新的派生类（如 TensorFFT）支持多维数组的 FFT 计算。
*/