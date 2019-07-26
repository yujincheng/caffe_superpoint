#ifndef _SUPERPOINT_H_
#define _SUPERPOINT_H_

#include <vector>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>

// using namespace caffe;
// using namespace std;

// void top_k(std::vector<point>& input_arr, int32_t n, int32_t k);
int ExactSP(caffe::shared_ptr< caffe::Net<float> > net_, const float* grey, std::vector<cv::KeyPoint>& kpts, std::vector<std::vector<float> >& dspts, int H, int W);
#endif
