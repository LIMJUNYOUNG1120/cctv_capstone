#include "track.h"

Track::Track(const Eigen::VectorXd& bbox, int id,
    const PersonFeatures& features)
    : id_(id), state_(Tentative), hits_(1), misses_(0),
    maxMisses_(5), minHits_(3), features_(features) {
    kf_.init(bbox);
}

void Track::predict() {
    kf_.predict();
}

void Track::update(const Eigen::VectorXd& bbox,
    const PersonFeatures& features) {
    kf_.update(bbox);
    hits_++;
    misses_ = 0;

    if (!features_.osnetFeature.empty() &&
        !features.osnetFeature.empty()) {
        for (int i = 0; i < (int)features_.osnetFeature.size(); i++)
            features_.osnetFeature[i] =
            0.9f * features_.osnetFeature[i] +
            0.1f * features.osnetFeature[i];
    }

    if (!features.gender.empty())
        features_.gender = features.gender;
    if (!features.upperType.empty())
        features_.upperType = features.upperType;
    if (!features.lowerType.empty())
        features_.lowerType = features.lowerType;
    if (!features.upperColor.empty())
        features_.upperColor = features.upperColor;
    if (!features.lowerColor.empty())
        features_.lowerColor = features.lowerColor;
    if (features.heightRatio > 0)
        features_.heightRatio = features.heightRatio;
    if (features.bodyRatio > 0)
        features_.bodyRatio = features.bodyRatio;

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

const PersonFeatures& Track::getFeatures() const {
    return features_;
}