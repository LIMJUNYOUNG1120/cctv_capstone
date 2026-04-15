#include "track.h"

Track::Track(const Eigen::VectorXd& bbox, int id,
    const std::vector<float>& feature)
    : id_(id), state_(Tentative), hits_(1), misses_(0),
    maxMisses_(5), minHits_(3), feature_(feature) {
    kf_.init(bbox);
}

void Track::predict() {
    kf_.predict();
}

void Track::update(const Eigen::VectorXd& bbox,
    const std::vector<float>& feature) {
    kf_.update(bbox);
    hits_++;
    misses_ = 0;
    // Æ¯Â¡ º¤ÅÍ ¾÷µ¥ÀÌÆ® (ÀÌµ¿ Æò±Õ)
    for (int i = 0; i < (int)feature_.size(); i++)
        feature_[i] = 0.9f * feature_[i] + 0.1f * feature[i];
    if (state_ == Tentative && hits_ >= minHits_)
        state_ = Confirmed;
}

void Track::markMissed() {
    misses_++;
    if (misses_ > maxMisses_)
        state_ = Deleted;
}

bool Track::isConfirmed() const {
    return state_ == Confirmed;
}

bool Track::isDeleted() const {
    return state_ == Deleted;
}

cv::Rect Track::getRect() const {
    Eigen::VectorXd state = kf_.getState();
    int x1 = (int)(state(0) - state(2) / 2);
    int y1 = (int)(state(1) - state(3) / 2);
    int w = (int)state(2);
    int h = (int)state(3);
    return cv::Rect(x1, y1, w, h);
}

int Track::getId() const {
    return id_;
}

const std::vector<float>& Track::getFeature() const {
    return feature_;
}