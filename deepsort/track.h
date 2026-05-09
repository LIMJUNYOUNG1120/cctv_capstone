#pragma once
#include "kalman_filter.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

enum TrackState {
    Tentative = 1,
    Confirmed = 2,
    Deleted = 3
};

struct PersonFeatures {
    std::vector<float> osnetFeature;  // OSNet 외형 벡터
    std::string gender;               // 성별
    std::string upperType;            // 상의 종류
    std::string lowerType;            // 하의 종류
    std::string upperColor;           // 상의 색상
    std::string lowerColor;           // 하의 색상
    float heightRatio;                // 키 비율
    float bodyRatio;                  // 체형 비율
};

class Track {
public:
    Track(const Eigen::VectorXd& bbox, int id,
        const PersonFeatures& features);

    void predict();
    void update(const Eigen::VectorXd& bbox,
        const PersonFeatures& features);
    void markMissed();
    bool isConfirmed() const;
    bool isDeleted() const;
    cv::Rect getRect() const;
    int getId() const;
    const PersonFeatures& getFeatures() const;

private:
    KalmanFilter kf_;
    int id_;
    TrackState state_;
    int hits_;
    int misses_;
    int maxMisses_;
    int frameCount_;
    int featureUpdateInterval_;
    int minHits_;
    PersonFeatures features_;
};