#include "tracker.h"
#include <numeric>
#include <cmath>

Tracker::Tracker(const std::string& reidModelPath)
    : nextId_(1), iouThreshold_(0.3f), reidThreshold_(0.5f),
    env_(ORT_LOGGING_LEVEL_WARNING, "ReID") {

    std::wstring wpath(reidModelPath.begin(), reidModelPath.end());
    reidSession_ = new Ort::Session(env_, wpath.c_str(), sessionOptions_);
}

std::vector<float> Tracker::extractFeature(
    const cv::Rect& box, const cv::Mat& frame) {

    // 박스 범위 체크
    cv::Rect safeBox = box;
    safeBox.x = std::max(0, safeBox.x);
    safeBox.y = std::max(0, safeBox.y);
    safeBox.width = std::min(safeBox.width, frame.cols - safeBox.x);
    safeBox.height = std::min(safeBox.height, frame.rows - safeBox.y);
    if (safeBox.width <= 0 || safeBox.height <= 0)
        return std::vector<float>(512, 0.0f);

    // 크롭 및 전처리
    cv::Mat crop = frame(safeBox);
    cv::Mat resized, rgb;
    cv::resize(crop, resized, cv::Size(128, 256));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // 정규화
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    rgb -= mean;
    rgb /= std;

    // HWC → CHW
    std::vector<float> inputData(3 * 256 * 128);
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < 256; h++)
            for (int w = 0; w < 128; w++)
                inputData[c * 256 * 128 + h * 128 + w] =
                rgb.at<cv::Vec3f>(h, w)[c];

    // 추론
    std::vector<int64_t> inputShape = { 1, 3, 256, 128 };
    Ort::MemoryInfo memInfo =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        inputData.data(), inputData.size(),
        inputShape.data(), inputShape.size());

    const char* inputName = "input";
    const char* outputName = "output";
    auto outputTensors = reidSession_->Run(
        Ort::RunOptions{ nullptr },
        &inputName, &inputTensor, 1,
        &outputName, 1);

    float* output = outputTensors[0]
        .GetTensorMutableData<float>();
    auto shape = outputTensors[0]
        .GetTensorTypeAndShapeInfo().GetShape();
    int featureSize = (int)shape[1];

    std::vector<float> feature(output, output + featureSize);

    // L2 정규화
    float norm = 0.0f;
    for (float v : feature) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0)
        for (float& v : feature) v /= norm;

    return feature;
}

float Tracker::cosineSimilarity(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    float dot = 0.0f;
    for (int i = 0; i < (int)a.size(); i++)
        dot += a[i] * b[i];
    return dot;
}

float Tracker::iou(const cv::Rect& a, const cv::Rect& b) {
    int interX1 = std::max(a.x, b.x);
    int interY1 = std::max(a.y, b.y);
    int interX2 = std::min(a.x + a.width, b.x + b.width);
    int interY2 = std::min(a.y + a.height, b.y + b.height);

    int interW = std::max(0, interX2 - interX1);
    int interH = std::max(0, interY2 - interY1);
    float interArea = (float)(interW * interH);

    float aArea = (float)(a.width * a.height);
    float bArea = (float)(b.width * b.height);
    float unionArea = aArea + bArea - interArea;

    return unionArea > 0 ? interArea / unionArea : 0.0f;
}

std::vector<std::vector<double>> Tracker::costMatrix(
    const std::vector<cv::Rect>& detections,
    const std::vector<std::vector<float>>& features) {

    std::vector<std::vector<double>> matrix(
        tracks_.size(),
        std::vector<double>(detections.size(), 1.0));

    for (int t = 0; t < (int)tracks_.size(); t++) {
        for (int d = 0; d < (int)detections.size(); d++) {
            float iouVal = iou(tracks_[t].getRect(), detections[d]);
            float reidVal = cosineSimilarity(
                tracks_[t].getFeature(), features[d]);
            // IOU 40% + Re-ID 60% 가중치
            matrix[t][d] = 1.0 - (0.6f * iouVal + 0.4f * reidVal);
        }
    }
    return matrix;
}

void Tracker::update(
    const std::vector<cv::Rect>& detections,
    const cv::Mat& frame) {

    // 모든 트랙 예측
    for (auto& track : tracks_)
        track.predict();

    // 탐지된 사람 Re-ID 특징 추출
    std::vector<std::vector<float>> features;
    for (auto& det : detections)
        features.push_back(extractFeature(det, frame));

    if (tracks_.empty()) {
        for (int d = 0; d < (int)detections.size(); d++) {
            Eigen::VectorXd bbox(4);
            bbox << detections[d].x + detections[d].width / 2.0,
                detections[d].y + detections[d].height / 2.0,
                detections[d].width,
                detections[d].height;
            tracks_.emplace_back(bbox, nextId_++, features[d]);
        }
        return;
    }

    // 헝가리안 알고리즘으로 매칭
    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    auto matrix = costMatrix(detections, features);
    hungarian.solve(matrix, assignment);

    std::vector<bool> detMatched(detections.size(), false);

    for (int t = 0; t < (int)tracks_.size(); t++) {
        int d = assignment[t];
        if (d >= 0 && d < (int)detections.size() &&
            matrix[t][d] < (1.0 - 0.5)) {
            Eigen::VectorXd bbox(4);
            bbox << detections[d].x + detections[d].width / 2.0,
                detections[d].y + detections[d].height / 2.0,
                detections[d].width,
                detections[d].height;
            tracks_[t].update(bbox, features[d]);
            detMatched[d] = true;
        }
        else {
            tracks_[t].markMissed();
        }
    }

    // 매칭 안 된 탐지 → 새 트랙 생성
    for (int d = 0; d < (int)detections.size(); d++) {
        if (!detMatched[d]) {
            Eigen::VectorXd bbox(4);
            bbox << detections[d].x + detections[d].width / 2.0,
                detections[d].y + detections[d].height / 2.0,
                detections[d].width,
                detections[d].height;
            tracks_.emplace_back(bbox, nextId_++, features[d]);
        }
    }

    // 삭제된 트랙 제거
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
            [](const Track& t) { return t.isDeleted(); }),
        tracks_.end());
}

std::vector<Track*> Tracker::getConfirmedTracks() {
    std::vector<Track*> confirmed;
    for (auto& track : tracks_)
        if (track.isConfirmed())
            confirmed.push_back(&track);
    return confirmed;
}