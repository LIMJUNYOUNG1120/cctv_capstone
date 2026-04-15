#pragma once
#include "kalman_filter.h"
#include <opencv2/opencv.hpp>
#include <vector>

enum TrackState {
    Tentative = 1,
    Confirmed = 2,
    Deleted = 3
};

class Track {
public:
    Track(const Eigen::VectorXd& bbox, int id,
        const std::vector<float>& feature);

    void predict();
    void update(const Eigen::VectorXd& bbox,
        const std::vector<float>& feature);
    void markMissed();
    bool isConfirmed() const;
    bool isDeleted() const;
    cv::Rect getRect() const;
    int getId() const;
    const std::vector<float>& getFeature() const;

private:
    KalmanFilter kf_;
    int id_;
    TrackState state_;
    int hits_;
    int misses_;
    int maxMisses_;
    int minHits_;
    std::vector<float> feature_;
};