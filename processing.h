#ifndef PROCESSING_H
#define PROCESSING_H

#include <opencv2/core.hpp>


cv::Mat procesarFrame(cv::Mat frame);
double calcularNitidez(const cv::Mat& frame);

#endif
