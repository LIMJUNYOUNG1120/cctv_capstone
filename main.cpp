#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "deepsort/tracker.h"

int main() {
    // ── 모델 로드 ──────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    Ort::SessionOptions sessionOptions;

    Ort::Session session(env,
        L"C:/cctvcapstone/project/cctv_capstone/yolov8n.onnx",
        sessionOptions);
    std::cout << "Model load success!" << std::endl;

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
        std::vector<int64_t> inputShape = { 1, 3, 640, 640 };
        Ort::MemoryInfo memInfo =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            inputData.data(), inputData.size(),
            inputShape.data(), inputShape.size());

        // ── 추론 실행 ──────────────────────────
        const char* inputName = "images";
        const char* outputName = "output0";
        auto start = std::chrono::high_resolution_clock::now();
        auto outputTensors = session.Run(
            Ort::RunOptions{ nullptr },
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
            float w = output[2 * 8400 + i];
            float h = output[3 * 8400 + i];

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
        cv::dnn::NMSBoxes(boxes, scores, 0.3f, 0.4f, indices);

        std::vector<cv::Rect> finalBoxes;
        for (int idx : indices)
            finalBoxes.push_back(boxes[idx]);

        // ── DeepSORT 업데이트 ──────────────────
        tracker.update(finalBoxes, frame);
        auto confirmedTracks = tracker.getConfirmedTracks();

        // ── 결과 표시 ──────────────────────────
        for (auto* track : confirmedTracks) {
            cv::Rect rect = track->getRect();

            // 범위 체크
            rect.x = std::max(0, rect.x);
            rect.y = std::max(0, rect.y);
            rect.width = std::min(rect.width, frame.cols - rect.x);
            rect.height = std::min(rect.height, frame.rows - rect.y);
            if (rect.width <= 0 || rect.height <= 0) continue;

            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
            std::string label = "ID:" + std::to_string(track->getId());
            cv::putText(frame, label,
                cv::Point(rect.x, rect.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);
        }

        std::cout << "Inference: " << ms << "ms"
            << " | Tracked: " << confirmedTracks.size()
            << " person(s)" << std::endl;

        cv::imshow("CCTV Detection + Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}