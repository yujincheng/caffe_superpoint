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
#include <SuperPoint.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace std;


int main(){


  SuperPoint spnet = SuperPoint("../demo2/model2.prototxt", "../demo2/model2.caffemodel", 200);
//   shared_ptr<Net<float> > net_;
//   net_.reset(new Net<float>("../demo2/model2.prototxt", TEST));
//   net_->CopyTrainedLayersFrom("../demo2/model2.caffemodel");
//   Caffe::set_mode(Caffe::GPU);
//   LOG(INFO)  << "123412341234";
  int Height = 480;
  int Width = 640;
  float* tmpfloat = new float [1*1*Height*Width];
  std::ifstream inpfile("../demo2/data/14.qwe.1", std::ios::binary);
  inpfile.read((char*)tmpfloat, 1*1*Height*Width*sizeof(float));
  inpfile.close();
  int num_channels_ = 1;
  auto input_geometry_ = cv::Size(640,480);
//  cv::Mat inputimg = cv::imread("../demo2/data/14.png", 0);
  cv::Mat inputimg(Height, Width, CV_32FC1, tmpfloat);
  cv::Mat img = cv::imread("../demo2/data/14.png", 0);
  cv::Mat sample;
  for (int i : {0,2,3,4,5,6,7} ){
  cout << "sample_resized : " << (int)img.ptr(0)[i] << endl;
  cout << "inputimg : " << inputimg.ptr<float>(0)[i] << endl;
  }

  if (img.channels() == 3 && num_channels_ == 1){
      cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
      cout << "1234" << endl;
      }
  else if (img.channels() == 4 && num_channels_ == 1)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
      cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
      sample = img;
  cv::Mat sample_resized;
  if (sample.size() != input_geometry_){
      cv::resize(sample, sample_resized, input_geometry_);
      cout << "2234" << endl;
      }
  else
      sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
  else {
      sample_resized.convertTo(sample_float, CV_32FC1);
      cout << "3234" << endl;
  }
  
  for (int i : {0,2,3,4,5,6,7} ){
  cout << "sample_float : " << sample_float.ptr<float>(0)[i] << endl;
  cout << "inputimg : " << inputimg.ptr<float>(0)[i] << endl;
  }

  //cv::imshow("src", inputimg);
  //cv::waitKey();
//   cv::Mat inputimg(Height, Width, CV_32FC1, tmpfloat);

  
// return 0;

  // int width = input_layer->width();
  // int height = input_layer->height();
  std::vector<cv::KeyPoint> kpts;
  std::vector<std::vector<float> > dspts;
//   ExactSP(net_, tmpfloat, kpts, dspts, Height, Width );
  cv::Mat imgshow = sample_float;
  spnet.ExactSP(imgshow,  kpts, dspts );
    
    // delete[] input_img;
    // delete[] result_semi;
    // delete[] result_desc;



  std::cout << std::endl;
  for (int j = 0; j < 256 ; j++ ){
    std::cout <<  dspts[195][j] << " ";
    if (j % 4 == 3){
      std::cout << std::endl;
    }
  }

//   cv::Mat inputimg(Height, Width, CV_32FC1, tmpfloat);
  for(int i = 0; i < kpts.size(); i++)
  { 
    cv::Point p(kpts[i].pt.x, kpts[i].pt.y);
    cv::circle(inputimg, p , 1, (0, 255, 0), -1);
  }
  cv::imshow("src", imgshow);
  cv::waitKey();
  return 0;
}
