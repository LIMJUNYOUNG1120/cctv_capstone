#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "deepsort/tracker.h"
#include <sstream>
#include <iomanip>
#include "json.hpp"
#include <fstream>

using json = nlohmann::json;

// 복장 색상 분류 함수
std::pair<std::string, std::string> classifyColor(
    const cv::Mat& frame, const cv::Rect& rect) {

    auto getColor = [&](cv::Rect region) -> std::string {
        region.x = std::max(0, region.x);
        region.y = std::max(0, region.y);
        region.width  = std::min(region.width,  frame.cols - region.x);
        region.height = std::min(region.height, frame.rows - region.y);
        if (region.width <= 0 || region.height <= 0)
            return "unknown";

        cv::Mat crop = frame(region);
        cv::Mat hsv;
        cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);

        cv::Scalar meanVal = cv::mean(hsv);
        float saturation = meanVal[1];
        float value      = meanVal[2];

        if (saturation < 60 && value > 160) return "white";
        if (value < 80)                     return "black";
        if (saturation < 60)                return "gray";

        cv::Mat mask;
        cv::inRange(hsv,
            cv::Scalar(0,   60,  80),
            cv::Scalar(180, 255, 255),
            mask);

        if (cv::countNonZero(mask) < 10)
            return "gray";

        std::vector<cv::Mat> channels;
        cv::split(hsv, channels);
        cv::Mat hue = channels[0];

        int histSize = 180;
        float range[] = {0, 180};
        const float* histRange = {range};
        cv::Mat hist;
        cv::calcHist(&hue, 1, 0, mask, hist,
            1, &histSize, &histRange);

        cv::GaussianBlur(hist, hist, cv::Size(5, 1), 0);

        cv::Point maxLoc;
        cv::minMaxLoc(hist, nullptr, nullptr, nullptr, &maxLoc);
        int dominantHue = maxLoc.y;

        if (dominantHue < 15  || dominantHue >= 165) return "red";
        if (dominantHue < 25)  return "orange";
        if (dominantHue < 35)  return "yellow";
        if (dominantHue < 85)  return "green";
        if (dominantHue < 130) return "blue";
        if (dominantHue < 165) return "purple";

        return "unknown";
    };

    // 상의: 박스 상위 15% ~ 50%
    cv::Rect upper(
        rect.x,
        rect.y + (int)(rect.height * 0.15f),
        rect.width,
        (int)(rect.height * 0.35f));

    // 하의: 박스 하위 50% ~ 90%
    cv::Rect lower(
        rect.x,
        rect.y + (int)(rect.height * 0.50f),
        rect.width,
        (int)(rect.height * 0.40f));

    return {getColor(upper), getColor(lower)};
}

int main() {
    // ── 모델 로드 ──────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    Ort::SessionOptions sessionOptions;

    Ort::Session session(env,
        L"C:/cctvcapstone/project/cctv_capstone/yolov8n.onnx",
        sessionOptions);
    std::cout << "Model load success!" << std::endl;

    // ── Homography 로드 ────────────────────────
    cv::Mat H;
    cv::FileStorage fs(
        "C:/cctvcapstone/homography.yml",
        cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cout << "Homography file not found!" << std::endl;
        system("pause");
        return -1;
    }
    fs["H"] >> H;
    fs.release();
    std::cout << "Homography loaded!" << std::endl;

    // ── 트래커 초기화 ──────────────────────────
    Tracker tracker("C:/cctvcapstone/osnet.onnx");
    std::cout << "Tracker initialized!" << std::endl;

    // ── 영상 로드 ──────────────────────────────
    cv::VideoCapture cap("C:\\cctvcapstone\\frames\\frame_%04d.jpg");
    if (!cap.isOpened()) {
        std::cout << "Video open failed" << std::endl;
        system("pause");
        return -1;
    }
    std::cout << "Video open success!" << std::endl;

    // ── 메인 루프 ──────────────────────────────
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Video ended" << std::endl;
            break;
        }

        // ── 전처리 ─────────────────────────────
        cv::Mat resized, rgb;
        cv::resize(frame, resized, cv::Size(640, 640));
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

        // HWC → CHW 변환
        std::vector<float> inputData(3 * 640 * 640);
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < 640; h++)
                for (int w = 0; w < 640; w++)
                    inputData[c * 640 * 640 + h * 640 + w] =
                        rgb.at<cv::Vec3f>(h, w)[c];

        // ── 입력 텐서 준비 ─────────────────────
        std::vector<int64_t> inputShape = {1, 3, 640, 640};
        Ort::MemoryInfo memInfo =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            inputData.data(), inputData.size(),
            inputShape.data(), inputShape.size());

        // ── 추론 실행 ──────────────────────────
        const char* inputName  = "images";
        const char* outputName = "output0";
        auto start = std::chrono::high_resolution_clock::now();
        auto outputTensors = session.Run(
            Ort::RunOptions{nullptr},
            &inputName, &inputTensor, 1,
            &outputName, 1);
        auto end = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>
            (end - start).count();

        // ── 출력 파싱 ──────────────────────────
        float* output = outputTensors[0]
            .GetTensorMutableData<float>();

        std::vector<cv::Rect> boxes;
        std::vector<float>    scores;

        for (int i = 0; i < 8400; i++) {
            float personScore = output[4 * 8400 + i];
            if (personScore < 0.3f) continue;

            float cx = output[0 * 8400 + i];
            float cy = output[1 * 8400 + i];
            float w  = output[2 * 8400 + i];
            float h  = output[3 * 8400 + i];

            int x1 = (int)((cx - w / 2) * frame.cols / 640);
            int y1 = (int)((cy - h / 2) * frame.rows / 640);
            int bw = (int)(w * frame.cols / 640);
            int bh = (int)(h * frame.rows / 640);

            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            bw = std::min(bw, frame.cols - x1);
            bh = std::min(bh, frame.rows - y1);

            boxes.push_back(cv::Rect(x1, y1, bw, bh));
            scores.push_back(personScore);
        }

        // ── NMS 적용 ───────────────────────────
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, 0.3f, 0.2f, indices);

        std::vector<cv::Rect> finalBoxes;
        for (int idx : indices)
            finalBoxes.push_back(boxes[idx]);

        // ── DeepSORT 업데이트 ──────────────────
        tracker.update(finalBoxes, frame);
        auto confirmedTracks = tracker.getConfirmedTracks();

        // ── 결과 표시 ──────────────────────────
        for (auto* track : confirmedTracks) {
            cv::Rect rect = track->getRect();

            rect.x = std::max(0, rect.x);
            rect.y = std::max(0, rect.y);
            rect.width  = std::min(rect.width,  frame.cols - rect.x);
            rect.height = std::min(rect.height, frame.rows - rect.y);
            if (rect.width <= 0 || rect.height <= 0) continue;

            // ── Homography 좌표 변환 ───────────
            cv::Point2f footPoint(
                rect.x + rect.width  / 2.0f,
                rect.y + rect.height);

            std::vector<cv::Point2f> src = {footPoint};
            std::vector<cv::Point2f> dst;
            cv::perspectiveTransform(src, dst, H);

            // 좌표 소수점 2자리로 표시
            std::ostringstream posStream;
            posStream << std::fixed << std::setprecision(2)
                      << "(" << dst[0].x << "m, "
                      << dst[0].y << "m)";

            // 박스 그리기
            cv::rectangle(frame, rect,
                cv::Scalar(0, 255, 0), 2);

            // ID 표시
            cv::putText(frame,
                "ID:" + std::to_string(track->getId()),
                cv::Point(rect.x, rect.y - 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);

                // 색상 분류
auto [upperColor, lowerColor] = classifyColor(frame, rect);

cv::putText(frame,
    "Top:" + upperColor,
    cv::Point(rect.x, rect.y - 45),
    cv::FONT_HERSHEY_SIMPLEX, 0.5,
    cv::Scalar(0, 200, 255), 1);

cv::putText(frame,
    "Bot:" + lowerColor,
    cv::Point(rect.x, rect.y - 30),
    cv::FONT_HERSHEY_SIMPLEX, 0.5,
    cv::Scalar(255, 200, 0), 1);

// ID 표시
cv::putText(frame,
    "ID:" + std::to_string(track->getId()),
    cv::Point(rect.x, rect.y - 25),
    cv::FONT_HERSHEY_SIMPLEX, 0.6,
    cv::Scalar(0, 255, 0), 2);

            // 좌표 표시
            cv::putText(frame,
                posStream.str(),
                cv::Point(rect.x, rect.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 0), 1);
        }

        std::cout << "Inference: " << ms << "ms"
                  << " | Tracked: " << confirmedTracks.size()
                  << " person(s)" << std::endl;
                  
         // ── JSON 저장 ──────────────────────────
json result = json::array();
for (auto* track : confirmedTracks) {
    cv::Rect rect = track->getRect();
    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width  = std::min(rect.width,  frame.cols - rect.x);
    rect.height = std::min(rect.height, frame.rows - rect.y);
    if (rect.width <= 0 || rect.height <= 0) continue;

    cv::Point2f footPoint(
        rect.x + rect.width  / 2.0f,
        rect.y + rect.height);
    std::vector<cv::Point2f> src = {footPoint};
    std::vector<cv::Point2f> dst;
    cv::perspectiveTransform(src, dst, H);

    auto [upperColor, lowerColor] = classifyColor(frame, rect);
    result.push_back({
        {"id",         track->getId()},
        {"x",          dst[0].x},
        {"y",          dst[0].y},
        {"upper",      upperColor},
        {"lower",      lowerColor}
    });
}

std::ofstream file("C:/cctvcapstone/positions.json");
file << result.dump(2);
file.close();         

        cv::imshow("CCTV Detection + Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}