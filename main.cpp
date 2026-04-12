#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

int main() {
    // ── 모델 로드 ──────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    Ort::SessionOptions sessionOptions;

    Ort::Session session(env,
        L"C:/cctvcapstone/project/cctv_capstone/yolov8n.onnx",
        sessionOptions);
    std::cout << "Model load success!" << std::endl;

    // ── 이미지 로드 ────────────────────────────
    cv::Mat frame = cv::imread("C:/cctvcapstone/test.jpg");
    if (frame.empty()) {
        std::cout << "Image load failed" << std::endl;
        system("pause");
        return -1;
    }
    std::cout << "Image load success!" << std::endl;

    // ── 전처리 ─────────────────────────────────
    cv::Mat resized, rgb;
    cv::resize(frame, resized, cv::Size(640, 640));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // HWC → CHW 변환
    std::vector<float> inputData(3 * 640 * 640);
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 640; h++) {
            for (int w = 0; w < 640; w++) {
                inputData[c * 640 * 640 + h * 640 + w] =
                    rgb.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    std::vector<int64_t> inputShape = { 1, 3, 640, 640 };

    Ort::MemoryInfo memInfo =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        inputData.data(), inputData.size(),
        inputShape.data(), inputShape.size());

    // ── 추론 실행 + 속도 측정 ──────────────────
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
    std::cout << "Inference: " << ms << "ms" << std::endl;

    // ── 출력 파싱 ──────────────────────────────
    float* output = outputTensors[0].GetTensorMutableData<float>();

    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;

    for (int i = 0; i < 8400; i++) {
        // person 클래스 score (row 4)
        float personScore = output[4 * 8400 + i];
        if (personScore < 0.3f) continue;

        float cx = output[0 * 8400 + i];
        float cy = output[1 * 8400 + i];
        float w = output[2 * 8400 + i];
        float h = output[3 * 8400 + i];

        // 640 기준 좌표 → 원본 이미지 좌표로 변환
        int x1 = (int)((cx - w / 2) * frame.cols / 640);
        int y1 = (int)((cy - h / 2) * frame.rows / 640);
        int bw = (int)(w * frame.cols / 640);
        int bh = (int)(h * frame.rows / 640);

        // 범위 벗어나는 박스 제거
        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        bw = std::min(bw, frame.cols - x1);
        bh = std::min(bh, frame.rows - y1);

        boxes.push_back(cv::Rect(x1, y1, bw, bh));
        scores.push_back(personScore);
    }

    // ── NMS 적용 ───────────────────────────────
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, 0.3f, 0.4f, indices);
    std::cout << "Detected: " << indices.size() << " person(s)" << std::endl;

    // ── 박스 그리기 ────────────────────────────
    for (int idx : indices) {
        cv::rectangle(frame, boxes[idx],
            cv::Scalar(0, 255, 0), 2);
        std::string label = "person "
            + std::to_string((int)(scores[idx] * 100)) + "%";
        cv::putText(frame, label,
            cv::Point(boxes[idx].x, boxes[idx].y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(0, 255, 0), 1);
    }

    // ── 결과 출력 ──────────────────────────────
    cv::imshow("Detection Result", frame);
    cv::imwrite("C:/cctvcapstone/result.jpg", frame);
    std::cout << "Result saved: C:/cctvcapstone/result.jpg" << std::endl;
    cv::waitKey(0);
    return 0;
}