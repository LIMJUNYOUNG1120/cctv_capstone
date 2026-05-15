#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <map>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "deepsort/tracker.h"
#include "json.hpp"

using json = nlohmann::json;

struct CameraInfo {
    int id;
    int mjpegPort;
    float refHeight;
    float scale;
    cv::Mat H;
    cv::Point2f camProjectedPos;
    float refDistance = 0.0f;
    float refBboxHeight = 0.0f;
    bool  distCalibDone = false;
    cv::Mat frame;
    std::mutex frameMutex;
    std::vector<uchar> jpegBuf;
    std::mutex jpegMutex;
    std::atomic<bool> hasFrame{ false };
    Tracker* tracker = nullptr;
    cv::VideoCapture* cap = nullptr;
};

std::map<int, CameraInfo*> activeCameras;
std::mutex cameraMapMutex;
int camCounter = 0;
const int BASE_MJPEG_PORT = 8091;
cv::Mat globalH;

void calcCameraProjectedPos(CameraInfo* cam) {
    if (cam->H.empty()) return;
    cv::Point2f imgCenter(320.0f, 240.0f);
    std::vector<cv::Point2f> src = { imgCenter };
    std::vector<cv::Point2f> dst;
    cv::perspectiveTransform(src, dst, cam->H);
    cam->camProjectedPos = dst[0];
    std::cout << "CAM" << cam->id
        << " 투영점: ("
        << cam->camProjectedPos.x << ", "
        << cam->camProjectedPos.y << ")" << std::endl;
}

std::string estimateHeight(
    float bboxHeight,
    float frameHeight,
    CameraInfo* cam,
    cv::Point2f footMapPos) {

    float estimatedCm = 0.0f;

    if (cam->distCalibDone && !cam->H.empty()
        && cam->refBboxHeight > 0 && cam->refDistance > 0) {
        // 거리 보정 방식
        float dx = footMapPos.x - cam->camProjectedPos.x;
        float dy = footMapPos.y - cam->camProjectedPos.y;
        float curDistance = std::sqrt(dx * dx + dy * dy);

        if (curDistance > 0) {
            float distRatio = curDistance / cam->refDistance;
            float correctedBbox = bboxHeight * distRatio;
            estimatedCm = cam->refHeight
                * (correctedBbox / cam->refBboxHeight);
        }
    }
    else if (cam->scale > 0.0f) {
        // 기본 보정 계수 방식
        estimatedCm = bboxHeight * cam->scale;
    }
    else {
        // 보정 없음
        float ratio = bboxHeight / frameHeight;
        estimatedCm = ratio * 170.0f * 2.0f;
    }

    if (estimatedCm < 140.0f) return "~140cm";
    else if (estimatedCm < 150.0f) return "140~150cm";
    else if (estimatedCm < 160.0f) return "150~160cm";
    else if (estimatedCm < 170.0f) return "160~170cm";
    else                           return "170cm~";
}

void runMjpegServer(CameraInfo* cam) {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    SOCKET serverSock = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(serverSock, SOL_SOCKET, SO_REUSEADDR,
        (char*)&opt, sizeof(opt));

    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(cam->mjpegPort);

    bind(serverSock, (sockaddr*)&addr, sizeof(addr));
    listen(serverSock, 10);

    std::cout << "MJPEG CAM" << cam->id
        << ": http://localhost:"
        << cam->mjpegPort << "/stream" << std::endl;

    while (true) {
        SOCKET clientSock = accept(serverSock, nullptr, nullptr);
        if (clientSock == INVALID_SOCKET) continue;

        std::thread([clientSock, cam]() {
            char buf[4096] = {};
            recv(clientSock, buf, sizeof(buf), 0);

            std::string header =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                "Cache-Control: no-cache\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: keep-alive\r\n\r\n";
            send(clientSock, header.c_str(), (int)header.size(), 0);

            while (true) {
                std::vector<uchar> jpegBuf;
                {
                    std::lock_guard<std::mutex> lock(cam->jpegMutex);
                    jpegBuf = cam->jpegBuf;
                }
                if (!jpegBuf.empty()) {
                    std::string frameHeader =
                        "--frame\r\n"
                        "Content-Type: image/jpeg\r\n"
                        "Content-Length: " +
                        std::to_string(jpegBuf.size()) + "\r\n\r\n";
                    int r1 = send(clientSock,
                        frameHeader.c_str(),
                        (int)frameHeader.size(), 0);
                    int r2 = send(clientSock,
                        (char*)jpegBuf.data(),
                        (int)jpegBuf.size(), 0);
                    int r3 = send(clientSock, "\r\n", 2, 0);
                    if (r1 < 0 || r2 < 0 || r3 < 0) break;
                }
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(33));
            }
            closesocket(clientSock);
            }).detach();
    }
    closesocket(serverSock);
}

void runApiServer() {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    SOCKET serverSock = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(serverSock, SOL_SOCKET, SO_REUSEADDR,
        (char*)&opt, sizeof(opt));

    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(8098);

    bind(serverSock, (sockaddr*)&addr, sizeof(addr));
    listen(serverSock, 10);
    std::cout << "API server: http://localhost:8098" << std::endl;

    while (true) {
        SOCKET clientSock = accept(serverSock, nullptr, nullptr);
        if (clientSock == INVALID_SOCKET) continue;

        std::thread([clientSock]() {
            std::string req;
            char buf[8192] = {};
            int received = recv(clientSock, buf, sizeof(buf) - 1, 0);
            if (received > 0) req = std::string(buf, received);

            std::string body;

            auto getParam = [&](const std::string& key,
                const std::string& def) {
                    std::string search = key + "=";
                    auto pos = req.find(search);
                    if (pos == std::string::npos) return def;
                    pos += search.size();
                    auto end = req.find_first_of(" &\r\n", pos);
                    return req.substr(pos, end - pos);
                };

            if (req.find("GET /add_camera") != std::string::npos) {
                std::lock_guard<std::mutex> lock(cameraMapMutex);
                camCounter++;
                int camId = camCounter;
                int mjpegPort = BASE_MJPEG_PORT + camId - 1;

                CameraInfo* cam = new CameraInfo();
                cam->id = camId;
                cam->mjpegPort = mjpegPort;
                cam->refHeight = std::stof(
                    getParam("ref_height", "170"));
                cam->scale = 0.0f;
                cam->distCalibDone = false;
                cam->tracker = new Tracker(
                    "C:/cctvcapstone/osnet.onnx",
                    "C:/cctvcapstone/clothing.onnx");

                if (!globalH.empty()) {
                    cam->H = globalH.clone();
                    calcCameraProjectedPos(cam);
                }

                cam->cap = new cv::VideoCapture();
                for (int idx = camId - 1; idx < 10; idx++) {
                    cam->cap->open(idx);
                    if (cam->cap->isOpened()) {
                        std::cout << "CAM" << camId
                            << " opened at index "
                            << idx << std::endl;
                        break;
                    }
                }

                activeCameras[camId] = cam;

                std::thread(runMjpegServer, cam).detach();

                std::thread([cam]() {
                    while (true) {
                        if (!cam->cap->isOpened()) {
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(100));
                            continue;
                        }
                        cv::Mat frame;
                        *cam->cap >> frame;
                        if (frame.empty()) continue;
                        std::lock_guard<std::mutex> lock(
                            cam->frameMutex);
                        cam->frame = frame.clone();
                        cam->hasFrame = true;
                    }
                    }).detach();

                body = "{\"cam_id\":" + std::to_string(camId)
                    + ",\"mjpeg_port\":" + std::to_string(mjpegPort)
                    + ",\"stream\":\"http://localhost:"
                    + std::to_string(mjpegPort) + "/stream\"}";

                std::cout << "CAM" << camId << " added!" << std::endl;

            }
            else if (req.find("GET /cameras") != std::string::npos) {
                std::lock_guard<std::mutex> lock(cameraMapMutex);
                json arr = json::array();
                for (auto& kv : activeCameras) {
                    arr.push_back({
                        {"id",             kv.first},
                        {"mjpeg",          kv.second->mjpegPort},
                        {"stream",         "http://localhost:"
                            + std::to_string(kv.second->mjpegPort)
                            + "/stream"},
                        {"active",         kv.second->hasFrame.load()},
                        {"has_homography", !kv.second->H.empty()},
                        {"dist_calib",     kv.second->distCalibDone}
                        });
                }
                body = arr.dump();

            }
            else if (req.find("GET /set_homography")
                != std::string::npos) {
                int camId = std::stoi(getParam("cam_id", "1"));
                std::string path = getParam("path", "");

                // URL 디코딩
                std::string decodedPath;
                for (size_t i = 0; i < path.size(); i++) {
                    if (path[i] == '+') {
                        decodedPath += ' ';
                    }
                    else if (path[i] == '%'
                        && i + 2 < path.size()) {
                        int val;
                        std::istringstream iss(
                            path.substr(i + 1, 2));
                        iss >> std::hex >> val;
                        decodedPath += (char)val;
                        i += 2;
                    }
                    else {
                        decodedPath += path[i];
                    }
                }

                std::lock_guard<std::mutex> lock(cameraMapMutex);
                if (activeCameras.count(camId)
                    && !decodedPath.empty()) {
                    cv::FileStorage fs(decodedPath,
                        cv::FileStorage::READ);
                    if (fs.isOpened()) {
                        fs["H"] >> activeCameras[camId]->H;
                        fs.release();
                        calcCameraProjectedPos(activeCameras[camId]);
                        std::cout << "CAM" << camId
                            << " Homography 로드 완료"
                            << std::endl;
                        body = "{\"ok\":true}";
                    }
                    else {
                        body = "{\"ok\":false,"
                            "\"error\":\"file not found\"}";
                    }
                }
                else {
                    body = "{\"ok\":false,"
                        "\"error\":\"cam not found\"}";
                }

            }
            else if (req.find("GET /set_scale")
                != std::string::npos) {
                int camId = std::stoi(
                    getParam("cam_id", "1"));
                float scale = std::stof(
                    getParam("scale", "1.0"));
                float refBbox = std::stof(
                    getParam("ref_bbox", "0"));
                float refHeight = std::stof(
                    getParam("ref_height", "170"));

                std::lock_guard<std::mutex> lock(cameraMapMutex);
                if (activeCameras.count(camId)) {
                    auto* cam = activeCameras[camId];
                    cam->scale = scale;
                    cam->refHeight = refHeight;

                    if (refBbox > 0 && !cam->H.empty()) {
                        cam->refBboxHeight = refBbox;

                        // 보정 당시 발 위치 (화면 중앙 하단)
                        cv::Point2f footPixel(320.0f, 480.0f);
                        std::vector<cv::Point2f> src = { footPixel };
                        std::vector<cv::Point2f> dst;
                        cv::perspectiveTransform(src, dst, cam->H);

                        float dx = dst[0].x
                            - cam->camProjectedPos.x;
                        float dy = dst[0].y
                            - cam->camProjectedPos.y;
                        cam->refDistance = std::sqrt(
                            dx * dx + dy * dy);
                        cam->distCalibDone = true;

                        std::cout << "CAM" << camId
                            << " 거리 보정 완료: refDist="
                            << cam->refDistance << std::endl;
                    }
                }
                body = "{\"ok\":true}";
            }

            std::string response =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Content-Length: "
                + std::to_string(body.size()) + "\r\n\r\n"
                + body;
            send(clientSock, response.c_str(),
                (int)response.size(), 0);
            closesocket(clientSock);
            }).detach();
    }
}

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    Ort::SessionOptions sessionOptions;

    Ort::Session session(env,
        L"C:/cctvcapstone/project/cctv_capstone/yolov8n.onnx",
        sessionOptions);
    std::cout << "YOLOv8 loaded!" << std::endl;

    cv::FileStorage fs(
        "C:/cctvcapstone/homography.yml",
        cv::FileStorage::READ);
    if (fs.isOpened()) {
        fs["H"] >> globalH;
        fs.release();
        std::cout << "Homography loaded!" << std::endl;
    }
    else {
        std::cout << "Homography not found, "
            "will use per-camera mapping" << std::endl;
    }

    std::thread(runApiServer).detach();

    std::cout << "System ready!" << std::endl;
    std::cout << "Add camera: http://localhost:8098/add_camera"
        << std::endl;
    std::cout << "Camera list: http://localhost:8098/cameras"
        << std::endl;

    while (true) {
        json allResult = json::array();

        std::map<int, CameraInfo*> camerasSnapshot;
        {
            std::lock_guard<std::mutex> lock(cameraMapMutex);
            camerasSnapshot = activeCameras;
        }

        if (camerasSnapshot.empty()) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(100));
            continue;
        }

        for (auto& kv : camerasSnapshot) {
            CameraInfo* cam = kv.second;
            if (!cam->hasFrame) continue;

            cv::Mat frame;
            {
                std::lock_guard<std::mutex> lock(cam->frameMutex);
                frame = cam->frame.clone();
            }
            if (frame.empty()) continue;

            cv::Mat resized, rgb;
            cv::resize(frame, resized, cv::Size(640, 640));
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
            rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

            std::vector<float> inputData(3 * 640 * 640);
            std::vector<cv::Mat> channels(3);
            cv::split(rgb, channels);
            for (int c = 0; c < 3; c++) {
                std::memcpy(
                    inputData.data() + c * 640 * 640,
                    channels[c].data,
                    640 * 640 * sizeof(float));
            }

            std::vector<int64_t> inputShape = { 1, 3, 640, 640 };
            Ort::MemoryInfo memInfo =
                Ort::MemoryInfo::CreateCpu(
                    OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor =
                Ort::Value::CreateTensor<float>(
                    memInfo,
                    inputData.data(), inputData.size(),
                    inputShape.data(), inputShape.size());

            const char* inputName = "images";
            const char* outputName = "output0";
            auto outputTensors = session.Run(
                Ort::RunOptions{ nullptr },
                &inputName, &inputTensor, 1,
                &outputName, 1);

            float* output =
                outputTensors[0].GetTensorMutableData<float>();

            std::vector<cv::Rect> boxes;
            std::vector<float>    scores;

            for (int i = 0; i < 8400; i++) {
                float personScore = output[4 * 8400 + i];
                if (personScore < 0.25f) continue;

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

            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, scores, 0.25f, 0.35f, indices);

            std::vector<cv::Rect> finalBoxes;
            for (int idx : indices)
                finalBoxes.push_back(boxes[idx]);

            cam->tracker->update(finalBoxes, frame);
            auto confirmedTracks =
                cam->tracker->getConfirmedTracks();

            for (auto* track : confirmedTracks) {
                cv::Rect rect = track->getRect();
                rect.x = std::max(0, rect.x);
                rect.y = std::max(0, rect.y);
                rect.width = std::min(rect.width,
                    frame.cols - rect.x);
                rect.height = std::min(rect.height,
                    frame.rows - rect.y);
                if (rect.width <= 0 || rect.height <= 0) continue;

                const PersonFeatures& feat = track->getFeatures();

                cv::Mat& H = cam->H.empty() ? globalH : cam->H;

                float mapX = 0, mapY = 0;
                cv::Point2f footMapPos(0, 0);
                if (!H.empty()) {
                    cv::Point2f footPoint(
                        rect.x + rect.width / 2.0f,
                        rect.y + rect.height);
                    std::vector<cv::Point2f> src = { footPoint };
                    std::vector<cv::Point2f> dst;
                    cv::perspectiveTransform(src, dst, H);
                    mapX = dst[0].x;
                    mapY = dst[0].y;
                    footMapPos = dst[0];
                }

                std::string heightLabel = estimateHeight(
                    (float)rect.height,
                    (float)frame.rows,
                    cam,
                    footMapPos);

                cv::rectangle(frame, rect,
                    cv::Scalar(0, 255, 0), 2);

                int tx = rect.x;
                int ty = rect.y - 5;
                auto putLabel = [&](const std::string& text,
                    int& y, cv::Scalar color) {
                        cv::putText(frame, text,
                            cv::Point(tx, y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45,
                            color, 1);
                        y -= 16;
                    };

                std::ostringstream pos;
                pos << std::fixed << std::setprecision(1)
                    << "(" << mapX << "," << mapY << ")";

                putLabel(pos.str(),
                    ty, cv::Scalar(255, 255, 0));
                putLabel("H:" + heightLabel,
                    ty, cv::Scalar(0, 255, 255));
                putLabel("Bot:" + feat.lowerType
                    + "/" + feat.lowerColor,
                    ty, cv::Scalar(255, 200, 0));
                putLabel("Top:" + feat.upperType
                    + "/" + feat.upperColor,
                    ty, cv::Scalar(0, 200, 255));
                putLabel(feat.gender,
                    ty, cv::Scalar(200, 255, 0));
                putLabel("ID:" + std::to_string(track->getId()),
                    ty, cv::Scalar(0, 255, 0));

                cv::putText(frame,
                    "CAM" + std::to_string(cam->id),
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 255, 255), 2);

                allResult.push_back({
                    {"id",        track->getId()},
                    {"cam",       cam->id},
                    {"x",         mapX},
                    {"y",         mapY},
                    {"gender",    feat.gender},
                    {"upper",     feat.upperColor},
                    {"lower",     feat.lowerColor},
                    {"upperType", feat.upperType},
                    {"lowerType", feat.lowerType},
                    {"height",    heightLabel}
                    });
            }

            std::vector<uchar> jpegBuf;
            cv::imencode(".jpg", frame, jpegBuf,
                { cv::IMWRITE_JPEG_QUALITY, 80 });
            {
                std::lock_guard<std::mutex> lock(cam->jpegMutex);
                cam->jpegBuf = jpegBuf;
            }
        }

        std::ofstream file("C:/cctvcapstone/positions.json");
        file << allResult.dump(2);
        file.close();

        std::this_thread::sleep_for(
            std::chrono::milliseconds(33));
    }

    return 0;
}