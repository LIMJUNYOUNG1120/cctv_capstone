#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> pixelPoints;
cv::Mat image;

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    if (pixelPoints.size() >= 4) return;

    pixelPoints.push_back(cv::Point2f(x, y));
    std::cout << "Marker " << pixelPoints.size()
              << " pixel: (" << x << ", " << y << ")" << std::endl;

    cv::circle(image, cv::Point(x, y), 8,
        cv::Scalar(0, 0, 255), -1);
    cv::putText(image,
        std::to_string(pixelPoints.size()),
        cv::Point(x + 10, y - 10),
        cv::FONT_HERSHEY_SIMPLEX, 1.0,
        cv::Scalar(0, 0, 255), 2);
    cv::imshow("Calibration", image);

    if (pixelPoints.size() == 4) {
        std::cout << "\n4 markers collected!" << std::endl;

        float width, height;
        std::cout << "Horizontal distance (m): ";
        std::cin >> width;
        std::cout << "Vertical distance (m): ";
        std::cin >> height;

        std::vector<cv::Point2f> realPoints = {
            {0.0f,   0.0f  },
            {width,  0.0f  },
            {0.0f,   height},
            {width,  height}
        };

        cv::Mat H = cv::findHomography(pixelPoints, realPoints);

        std::cout << "\nHomography Matrix:" << std::endl;
        std::cout << H << std::endl;

        std::cout << "\nTransform Test:" << std::endl;
        for (int i = 0; i < 4; i++) {
            std::vector<cv::Point2f> src = {pixelPoints[i]};
            std::vector<cv::Point2f> dst;
            cv::perspectiveTransform(src, dst, H);
            std::cout << "Marker " << i + 1
                      << " pixel(" << pixelPoints[i].x
                      << ", " << pixelPoints[i].y << ")"
                      << " -> real(" << dst[0].x
                      << "m, " << dst[0].y << "m)" << std::endl;
        }

        cv::FileStorage fs(
            "C:/cctvcapstone/homography.yml",
            cv::FileStorage::WRITE);
        fs << "H" << H;
        fs.release();
        std::cout << "\nSaved: C:/cctvcapstone/homography.yml" << std::endl;

        system("pause");
    }
}

int main() {
    image = cv::imread("C:/cctvcapstone/frames/frame_0001.jpg");
    if (image.empty()) {
        std::cout << "Image load failed" << std::endl;
        system("pause");
        return -1;
    }

    std::cout << "Click 4 markers in order:" << std::endl;
    std::cout << "Top-Left -> Top-Right -> Bottom-Left -> Bottom-Right" << std::endl;

    cv::imshow("Calibration", image);
    cv::setMouseCallback("Calibration", onMouse);
    cv::waitKey(0);

    return 0;
}