#pragma once
#include <Eigen/Dense>

class KalmanFilter {
public:
    KalmanFilter();
    void init(const Eigen::VectorXd& bbox);
    void predict();
    Eigen::VectorXd update(const Eigen::VectorXd& bbox);
    Eigen::VectorXd getState() const;

private:
    Eigen::MatrixXd F_;  // 상태 전이 행렬
    Eigen::MatrixXd H_;  // 관측 행렬
    Eigen::MatrixXd Q_;  // 프로세스 노이즈
    Eigen::MatrixXd R_;  // 관측 노이즈
    Eigen::MatrixXd P_;  // 오차 공분산
    Eigen::VectorXd x_;  // 상태 벡터 (cx, cy, w, h, vx, vy, vw, vh)
};