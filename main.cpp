#include<iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "processing.h"

using namespace std;


int main(int argc, char *argv[]) {

   if (argc < 2) {
        cerr << "Error: Debes proporcionar una fuente de video." << endl;
        cerr << "Uso: " << argv[0] << " <fuente_de_video>" << endl;
        return 1; // Salir con código de error
    }
   cv::VideoCapture vid;
   vid.open(argv[1]); 
   if(!vid.isOpened()) {
      cerr << "Error opening input." << endl;
      return 1;
   }
   cv::namedWindow("Video izquierdo procesado", cv::WINDOW_NORMAL);
   cv::namedWindow("Video derecho procesado", cv::WINDOW_NORMAL);
   
   const double UMBRAL_NITIDEZ = 200;
   cv::Mat img;
   
   while(true) {
      vid >> img;
      if (img.empty()) {
            std::cout << "Fin del video." << std::endl;
            break;
        }
      
      cv::Rect roi_izq(0, 0, img.cols/2, img.rows);
      cv::Mat img_izq = img(roi_izq);

      cv::Rect roi_der(img.cols/2, 0, img.cols/2, img.rows);
      cv::Mat img_der = img(roi_der);

      cv::Mat proc_izq = procesarFrame(img_izq);
      double nitidez_izq = calcularNitidez(proc_izq);

      cv::Mat proc_der = procesarFrame(img_der);
      double nitidez_der = calcularNitidez(proc_der);

      
      if (nitidez_izq < UMBRAL_NITIDEZ || nitidez_der < UMBRAL_NITIDEZ){
         continue;
      }
      std::cout <<"Nitidez izq: " << nitidez_izq << std::endl;
      std::cout <<"Nitidez der: " << nitidez_der << std::endl;
     
      cv::imshow("Video izquierdo procesado", proc_izq);
      cv::imshow("Video derecho procesado", proc_der);


      if(cv::waitKey(10) != -1)
         break;
      }
      vid.release();
      cv::destroyAllWindows();
      return 0;
}
