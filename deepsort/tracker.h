#pragma once
#include "track.h"
#include "hungarian.h"
#include <vector>
#include <onnxruntime_cxx_api.h>

class Tracker {
public:
    Tracker(const std::string& reidModelPath);
    void update(const std::vector<cv::Rect>& detections,
        const cv::Mat& frame);
    std::vector<Track*> getConfirmedTracks();

private:
    std::vector<Track> tracks_;
    int nextId_;
    float iouThreshold_;
    float reidThreshold_;

    // Re-ID ¸ðµ¨
    Ort::Env env_;
    Ort::Session* reidSession_;
    Ort::SessionOptions sessionOptions_;

    float iou(const cv::Rect& a, const cv::Rect& b);
    std::vector<float> extractFeature(
        const cv::Rect& box, const cv::Mat& frame);
    float cosineSimilarity(
        const std::vector<float>& a,
        const std::vector<float>& b);
    std::vector<std::vector<double>> costMatrix(
        const std::vector<cv::Rect>& detections,
        const std::vector<std::vector<float>>& features);
};