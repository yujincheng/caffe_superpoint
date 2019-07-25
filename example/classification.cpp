#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <superpoint.h>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;


int main(){
  shared_ptr<Net<float> > net_;
  net_.reset(new Net<float>("../demo2/model2.prototxt", TEST));
  net_->CopyTrainedLayersFrom("../demo2/model2.caffemodel");
  Caffe::set_mode(Caffe::GPU);
  LOG(INFO)  << "123412341234";
  int Height = 480;
  int Width = 640;
  float* tmpfloat = new float [1*1*Height*Width];
  std::ifstream inpfile("../demo2/inp.qwe", std::ios::binary);
  inpfile.read((char*)tmpfloat, 1*1*Height*Width*sizeof(float));
  inpfile.close();
  
  // int width = input_layer->width();
  // int height = input_layer->height();
  std::vector<cv::KeyPoint> kpts;
  std::vector<std::vector<float> > dspts;
  ExactSP(net_, tmpfloat, kpts, dspts, Height, Width );
    
    // delete[] input_img;
    // delete[] result_semi;
    // delete[] result_desc;



  std::cout << std::endl;
  for (int j = 0; j < dspts.size() ; j++ ){
    std::cout <<  dspts[0][j] << " ";
    if (j % 4 == 3){
      std::cout << std::endl;
    }
  }

  cv::Mat inputimg(Height, Width, CV_32FC1, tmpfloat);
  for(int i = 0; i < kpts.size(); i++)
  { 
    cv::Point p(kpts[i].pt.x, kpts[i].pt.y);
    cv::circle(inputimg, p , 1, (0, 255, 0), -1);
  }
  cv::imshow("src", inputimg);
  cv::waitKey();
  return 0;
}

