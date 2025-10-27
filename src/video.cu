#include "video.cuh"
#include "image.cuh"

#include <opencv2/opencv.hpp>

#include <string>
#include <stdexcept>

void decodeVideo(std::string filepath, void (*callback)(uchar3*, int2)){
    cv::VideoCapture cap(filepath);
    if (!cap.isOpened()) {
        throw std::runtime_error("Cannot open video");
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        cv::Mat rgbFrame;
        cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);

        int2 size = { rgbFrame.cols, rgbFrame.rows };
        uchar3* data = reinterpret_cast<uchar3*>(rgbFrame.data);

        callback(data, size);
    }

    cap.release();
}

void decodeFrambuffer(std::string filepath, void (*callback)(Framebuffer)){
    cv::VideoCapture renderCap(filepath);
    cv::VideoCapture normalCap(filepath);
    cv::VideoCapture albedoCap(filepath);

    if (!renderCap.isOpened() || normalCap.isOpened() || !albedoCap.isOpened()) {
        throw std::runtime_error("Cannot open videos");
    }

    cv::Mat renderFrame, normalFrame, albedoFrame;
    while (true) {
        renderCap.read(renderFrame);
        normalCap.read(normalFrame);
        albedoCap.read(albedoFrame);

        if (renderFrame.empty() || normalFrame.empty() || albedoFrame.empty())
            break;

        callback({
            {renderFrame.cols, renderFrame.rows},
            (Pixel*) renderFrame.data,
            (Pixel*) albedoFrame.data,
            (Pixel*) normalFrame.data
        });
    }


    renderCap.release();
    normalCap.release();
    albedoCap.release();
}