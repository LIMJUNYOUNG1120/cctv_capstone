#include "tracker.h"
#include <numeric>
#include <cmath>

Tracker::Tracker(const std::string& osnetPath,
    const std::string& clothingPath)
    : nextId_(1), matchThreshold_(0.5f),
    env_(ORT_LOGGING_LEVEL_WARNING, "Tracker") {

    std::wstring wOsnet(osnetPath.begin(), osnetPath.end());
    std::wstring wClothing(clothingPath.begin(), clothingPath.end());
    osnetSession_ = new Ort::Session(env_, wOsnet.c_str(), sessionOptions_);
    clothingSession_ = new Ort::Session(env_, wClothing.c_str(), sessionOptions_);
}

Tracker::~Tracker() {
    delete osnetSession_;
    delete clothingSession_;
}

// ── OSNet 특징 추출 ──────────────────────────
std::vector<float> Tracker::extractOsnet(
    const cv::Rect& box, const cv::Mat& frame) {

    cv::Rect safe = box;
    safe.x = std::max(0, safe.x);
    safe.y = std::max(0, safe.y);
    safe.width = std::min(safe.width, frame.cols - safe.x);
    safe.height = std::min(safe.height, frame.rows - safe.y);
    if (safe.width <= 0 || safe.height <= 0)
        return std::vector<float>(512, 0.0f);

    cv::Mat crop, rgb;
    cv::resize(frame(safe), crop, cv::Size(128, 256));
    cv::cvtColor(crop, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    rgb -= cv::Scalar(0.485, 0.456, 0.406);
    rgb /= cv::Scalar(0.229, 0.224, 0.225);

    std::vector<float> input(3 * 256 * 128);
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < 256; h++)
            for (int w = 0; w < 128; w++)
                input[c * 256 * 128 + h * 128 + w] =
                rgb.at<cv::Vec3f>(h, w)[c];

    std::vector<int64_t> shape = { 1, 3, 256, 128 };
    Ort::MemoryInfo memInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memInfo, input.data(), input.size(),
        shape.data(), shape.size());

    const char* inName = "input";
    const char* outName = "output";
    auto out = osnetSession_->Run(
        Ort::RunOptions{ nullptr },
        &inName, &tensor, 1, &outName, 1);

    float* data = out[0].GetTensorMutableData<float>();
    auto   outShape = out[0].GetTensorTypeAndShapeInfo().GetShape();
    int    size = (int)outShape[1];

    std::vector<float> feat(data, data + size);
    float norm = 0.0f;
    for (float v : feat) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0) for (float& v : feat) v /= norm;
    return feat;
}

// ── 색상 분류 ────────────────────────────────
std::string Tracker::classifyColor(const cv::Mat& region) {
    if (region.empty()) return "unknown";
    cv::Mat hsv;
    cv::cvtColor(region, hsv, cv::COLOR_BGR2HSV);
    cv::Scalar mean = cv::mean(hsv);
    float sat = mean[1], val = mean[2];
    if (sat < 60 && val > 160) return "white";
    if (val < 80)              return "black";
    if (sat < 60)              return "gray";

    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(0, 60, 80),
        cv::Scalar(180, 255, 255), mask);
    if (cv::countNonZero(mask) < 10) return "gray";

    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);
    int histSize = 180;
    float range[] = { 0, 180 };
    const float* r = { range };
    cv::Mat hist;
    cv::calcHist(&ch[0], 1, 0, mask, hist, 1, &histSize, &r);
    cv::GaussianBlur(hist, hist, cv::Size(5, 1), 0);
    cv::Point maxLoc;
    cv::minMaxLoc(hist, nullptr, nullptr, nullptr, &maxLoc);
    int h = maxLoc.y;

    if (h < 15 || h >= 165) return "red";
    if (h < 25)  return "orange";
    if (h < 35)  return "yellow";
    if (h < 85)  return "green";
    if (h < 130) return "blue";
    if (h < 165) return "purple";
    return "unknown";
}

// ── 전체 특징 추출 ───────────────────────────
PersonFeatures Tracker::extractFeatures(
    const cv::Rect& box, const cv::Mat& frame) {

    PersonFeatures f;

    cv::Rect safe = box;
    safe.x = std::max(0, safe.x);
    safe.y = std::max(0, safe.y);
    safe.width = std::min(safe.width, frame.cols - safe.x);
    safe.height = std::min(safe.height, frame.rows - safe.y);
    if (safe.width <= 0 || safe.height <= 0) return f;

    // ── OSNet ────────────────────────────────
    f.osnetFeature = extractOsnet(safe, frame);

    // ── 키 비율 ──────────────────────────────
    f.heightRatio = (float)safe.height / frame.rows;

    // ── 체형 비율 (어깨너비 / 키) ────────────
    f.bodyRatio = (float)safe.width / safe.height;

    // ── 색상 분류 ────────────────────────────
    cv::Rect upper(safe.x,
        safe.y + (int)(safe.height * 0.15f),
        safe.width,
        (int)(safe.height * 0.35f));
    cv::Rect lower(safe.x,
        safe.y + (int)(safe.height * 0.50f),
        safe.width,
        (int)(safe.height * 0.40f));

    upper.x = std::max(0, upper.x);
    upper.y = std::max(0, upper.y);
    upper.width = std::min(upper.width, frame.cols - upper.x);
    upper.height = std::min(upper.height, frame.rows - upper.y);
    lower.x = std::max(0, lower.x);
    lower.y = std::max(0, lower.y);
    lower.width = std::min(lower.width, frame.cols - lower.x);
    lower.height = std::min(lower.height, frame.rows - lower.y);

    if (upper.width > 0 && upper.height > 0)
        f.upperColor = classifyColor(frame(upper));
    if (lower.width > 0 && lower.height > 0)
        f.lowerColor = classifyColor(frame(lower));

    // ── MobileNetV2 (성별·옷 종류) ───────────
    cv::Mat crop, rgb;
    cv::resize(frame(safe), crop, cv::Size(224, 224));
    cv::cvtColor(crop, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    rgb -= cv::Scalar(0.485, 0.456, 0.406);
    rgb /= cv::Scalar(0.229, 0.224, 0.225);

    std::vector<float> input(3 * 224 * 224);
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < 224; h++)
            for (int w = 0; w < 224; w++)
                input[c * 224 * 224 + h * 224 + w] =
                rgb.at<cv::Vec3f>(h, w)[c];

    std::vector<int64_t> shape = { 1, 3, 224, 224 };
    Ort::MemoryInfo memInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memInfo, input.data(), input.size(),
        shape.data(), shape.size());

    const char* inName = "input";
    const char* outNames[] = { "gender", "upper", "lower" };
    auto out = clothingSession_->Run(
        Ort::RunOptions{ nullptr },
        &inName, &tensor, 1, outNames, 3);

    // 성별
    float* genderData = out[0].GetTensorMutableData<float>();
    f.gender = (genderData[0] > genderData[1]) ? "male" : "female";

    // 상의 종류
    float* upperData = out[1].GetTensorMutableData<float>();
    std::vector<std::string> upperTypes = { "tshirt","hoodie","jacket","shirt" };
    int upperIdx = std::max_element(upperData, upperData + 4) - upperData;
    f.upperType = upperTypes[upperIdx];

    // 하의 종류
    float* lowerData = out[2].GetTensorMutableData<float>();
    std::vector<std::string> lowerTypes = { "pants","skirt","shorts" };
    int lowerIdx = std::max_element(lowerData, lowerData + 3) - lowerData;
    f.lowerType = lowerTypes[lowerIdx];

    return f;
}

// ── 유사도 계산 ──────────────────────────────
float Tracker::cosineSimilarity(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    if (a.empty() || b.empty()) return 0.0f;
    float dot = 0.0f;
    for (int i = 0; i < (int)a.size(); i++)
        dot += a[i] * b[i];
    return dot;
}

float Tracker::computeSimilarity(
    const PersonFeatures& a,
    const PersonFeatures& b) {

    float score = 0.0f;

    // OSNet 외형 벡터 (20%)
    score += 0.20f * cosineSimilarity(
        a.osnetFeature, b.osnetFeature);

    // 성별 (15%)
    if (!a.gender.empty() && !b.gender.empty())
        score += 0.15f * (a.gender == b.gender ? 1.0f : 0.0f);

    // 상의 종류 (10%)
    if (!a.upperType.empty() && !b.upperType.empty())
        score += 0.10f * (a.upperType == b.upperType ? 1.0f : 0.0f);

    // 하의 종류 (10%)
    if (!a.lowerType.empty() && !b.lowerType.empty())
        score += 0.10f * (a.lowerType == b.lowerType ? 1.0f : 0.0f);

    // 체형 비율 (10%)
    float bodyDiff = std::abs(a.bodyRatio - b.bodyRatio);
    score += 0.10f * std::max(0.0f, 1.0f - bodyDiff * 2.0f);

    // 키 비율 (10%)
    float heightDiff = std::abs(a.heightRatio - b.heightRatio);
    score += 0.10f * std::max(0.0f, 1.0f - heightDiff * 5.0f);

    // 상의 색상 (5%)
    if (!a.upperColor.empty() && !b.upperColor.empty())
        score += 0.05f * (a.upperColor == b.upperColor ? 1.0f : 0.0f);

    // 하의 색상 (5%)
    if (!a.lowerColor.empty() && !b.lowerColor.empty())
        score += 0.05f * (a.lowerColor == b.lowerColor ? 1.0f : 0.0f);

    return score;
}

// ── 비용 행렬 ────────────────────────────────
std::vector<std::vector<double>> Tracker::costMatrix(
    const std::vector<cv::Rect>& detections,
    const std::vector<PersonFeatures>& features) {

    std::vector<std::vector<double>> matrix(
        tracks_.size(),
        std::vector<double>(detections.size(), 1.0));

    for (int t = 0; t < (int)tracks_.size(); t++)
        for (int d = 0; d < (int)detections.size(); d++)
            matrix[t][d] = 1.0 - computeSimilarity(
                tracks_[t].getFeatures(), features[d]);

    return matrix;
}

// ── 트래커 업데이트 ──────────────────────────
void Tracker::update(
    const std::vector<cv::Rect>& detections,
    const cv::Mat& frame) {

    for (auto& track : tracks_)
        track.predict();

    std::vector<PersonFeatures> features;
    for (auto& det : detections)
        features.push_back(extractFeatures(det, frame));

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

    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    auto matrix = costMatrix(detections, features);
    hungarian.solve(matrix, assignment);

    std::vector<bool> detMatched(detections.size(), false);

    for (int t = 0; t < (int)tracks_.size(); t++) {
        int d = assignment[t];
        if (d >= 0 && d < (int)detections.size() &&
            matrix[t][d] < (1.0 - matchThreshold_)) {
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

    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
            [](const Track& t) { return t.isDeleted(); }),
        tracks_.end());
}

// ── Confirmed 트랙 반환 ──────────────────────
std::vector<Track*> Tracker::getConfirmedTracks() {
    std::vector<Track*> confirmed;
    for (auto& track : tracks_)
        if (track.isConfirmed())
            confirmed.push_back(&track);
    return confirmed;
}