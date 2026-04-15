#include "kalman_filter.h"

KalmanFilter::KalmanFilter() {
    // 상태 전이 행렬 (8x8)
    F_ = Eigen::MatrixXd::Identity(8, 8);
    F_(0, 4) = 1; F_(1, 5) = 1; F_(2, 6) = 1; F_(3, 7) = 1;

    // 관측 행렬 (4x8)
    H_ = Eigen::MatrixXd::Zero(4, 8);
    H_(0, 0) = 1; H_(1, 1) = 1; H_(2, 2) = 1; H_(3, 3) = 1;

    // 프로세스 노이즈
    Q_ = Eigen::MatrixXd::Identity(8, 8);
    Q_(4, 4) = 0.01; Q_(5, 5) = 0.01;
    Q_(6, 6) = 0.01; Q_(7, 7) = 0.01;

    // 관측 노이즈
    R_ = Eigen::MatrixXd::Identity(4, 4);
    R_(2, 2) = 10; R_(3, 3) = 10;

    // 오차 공분산
    P_ = Eigen::MatrixXd::Identity(8, 8);
    P_(4, 4) = 100; P_(5, 5) = 100;
    P_(6, 6) = 100; P_(7, 7) = 100;

    x_ = Eigen::VectorXd::Zero(8);
}

void KalmanFilter::init(const Eigen::VectorXd& bbox) {
    x_.head(4) = bbox;
    x_.tail(4) = Eigen::VectorXd::Zero(4);
}

void KalmanFilter::predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

Eigen::VectorXd KalmanFilter::update(const Eigen::VectorXd& bbox) {
    Eigen::VectorXd y = bbox - H_ * x_;
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
    x_ = x_ + K * y;
    P_ = (Eigen::MatrixXd::Identity(8, 8) - K * H_) * P_;
    return x_.head(4);
}

Eigen::VectorXd KalmanFilter::getState() const {
    return x_.head(4);
}