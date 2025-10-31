#include "processing.h" 
#include <opencv2/imgproc.hpp> 

// Función para procesar frame
cv::Mat procesarFrame(cv::Mat frame) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Reducción de ruido
    cv::Mat denoised;
    cv::bilateralFilter(gray, denoised, 9, 75, 75); 
    
    // Mejorar contraste
    cv::Mat contrasted;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->setTilesGridSize(cv::Size(8,8));
    clahe->apply(denoised, contrasted); 

    // Mejorar nitidez
    cv::Mat sharpened;
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 0,-1,0, -1,5,-1, 0,-1,0);
    cv::filter2D(contrasted, sharpened, -1, kernel);

    return sharpened; 
}

// Calculamos si el nivel de nitidez es lo suficientemente alto para que se pueda usar el frame
double calcularNitidez(const cv::Mat& frame) {
    cv::Mat laplacian;
    cv::Laplacian(frame, laplacian, CV_64F); 
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    return stddev.val[0] * stddev.val[0];
}
