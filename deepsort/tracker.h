#pragma once
#include "track.h"
#include "hungarian.h"
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>

class Tracker {
public:
    Tracker(const std::string& osnetPath,
        const std::string& clothingPath);
    ~Tracker();

    void update(const std::vector<cv::Rect>& detections,
        const cv::Mat& frame);
    std::vector<Track*> getConfirmedTracks();

private:
    std::vector<Track> tracks_;
    int nextId_;
    float matchThreshold_;
    int frameCount_;
    int featureUpdateInterval_;

    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    Ort::Session* osnetSession_;
    Ort::Session* clothingSession_;

    std::vector<float> extractOsnet(
        const cv::Rect& box, const cv::Mat& frame);
    PersonFeatures extractFeatures(
        const cv::Rect& box, const cv::Mat& frame);
    std::string classifyColor(const cv::Mat& region);
    float computeSimilarity(
        const PersonFeatures& a, const PersonFeatures& b);
    float cosineSimilarity(
        const std::vector<float>& a, const std::vector<float>& b);
    std::vector<std::vector<double>> costMatrix(
        const std::vector<cv::Rect>& detections,
        const std::vector<PersonFeatures>& features);
};