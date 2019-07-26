#include <superpoint.h>
#include <stdlib.h>
#include <chrono>
using namespace std;
// using namespace caffe;

class point
{
    public:
        int W;   
        int H;  
        float semi;   
        point(int a, int b, float c) {H=a;W=b;semi=c;}
        point() {}
};

class max_heap_t {
    public:
    
    vector<point> arr;
    int32_t  n;

    max_heap_t (vector<point> input_arr, int32_t arr_size){
        arr.assign(input_arr.begin(),input_arr.begin()+arr_size);
        n = arr_size;
    }

    ~max_heap_t () {
    }

    /* time complexity => O(nlogn) */
    void    build_heap_from_top_to_bottom() {
      
        for (int32_t i = 1; i < n; i++) {
           heap_ajust_from_bottom_to_top(i);
        }
    }

    /* O(logn) */
    void    heap_ajust_from_bottom_to_top(int32_t bottom_index) {
        point tmp = arr[bottom_index];
        while (bottom_index > 0) {
            int32_t parent_index = (bottom_index - 1) / 2;
            if (arr[parent_index].semi < tmp.semi ) {
                arr[bottom_index] = arr[parent_index];
                bottom_index = parent_index;
            }
            else {
                break;
            }
        }
        arr[bottom_index] = tmp;
    }

     /* O(n) */
    void    build_heap_from_bottom_to_top() {
        int32_t max_index = n - 1;
        for (int32_t i = (max_index - 1) / 2; i >= 0; i--) {
            heap_adjust_from_top_to_bottom(i, max_index);
        }
    }

    /* O(logn) */
    void    heap_adjust_from_top_to_bottom(int32_t top_index, int32_t bottom_index) {
        point tmp = arr[top_index];
        while (top_index <= (bottom_index - 1) / 2) {
            point max_one = tmp;
            int32_t child_idx = 0;
            int32_t left_child_idx = top_index * 2 + 1;
            int32_t right_child_idx = top_index * 2 + 2;
            
            if (left_child_idx <= bottom_index && max_one.semi < arr[left_child_idx].semi ) {
                max_one = arr[left_child_idx];
                child_idx = left_child_idx;
            }
            if (right_child_idx <= bottom_index && max_one.semi < arr[right_child_idx].semi ) {
                max_one = arr[right_child_idx];
                child_idx = right_child_idx;
            }
          
            if (max_one.semi != tmp.semi) {
                arr[top_index] = max_one;
                top_index = child_idx;
            }
            else {
                break;
            }
        }
        arr[top_index] = tmp;
    }

    void    sort() {
        // build  heap first
        build_heap_from_bottom_to_top();

        // sort
        point tmp;
        for (int32_t i = n - 1; i > 0;) {
            // move heap top to end
            tmp = arr[0];
            arr[0] = arr[i];
            arr[i] = tmp;

            // adjust the heap
            heap_adjust_from_top_to_bottom(0, --i);
        }
    }

};

class min_heap_t {
    public:
    
    vector<point> arr;
    int32_t  n;

    min_heap_t (vector<point> input_arr, int32_t arr_size){
        arr.assign(input_arr.begin(),input_arr.begin()+arr_size);
        n = arr_size;
    }

    ~min_heap_t () {
    }

    /* time complexity => O(nlogn) */
    void    build_heap_from_top_to_bottom() {
      
        for (int32_t i = 1; i < n; i++) {
           heap_ajust_from_bottom_to_top(i);
        }
    }

    /* O(logn) */
    void    heap_ajust_from_bottom_to_top(int32_t bottom_index) {
        point tmp = arr[bottom_index];
        while (bottom_index > 0) {
            int32_t parent_index = (bottom_index - 1) / 2;
            if (arr[parent_index].semi > tmp.semi ) {
                arr[bottom_index] = arr[parent_index];
                bottom_index = parent_index;
            }
            else {
                break;
            }
        }
        arr[bottom_index] = tmp;
    }

     /* O(n) */
    void    build_heap_from_bottom_to_top() {
        int32_t max_index = n - 1;
        for (int32_t i = (max_index - 1) / 2; i >= 0; i--) {
            heap_adjust_from_top_to_bottom(i, max_index);
        }
    }

    /* O(logn) */
    void    heap_adjust_from_top_to_bottom(int32_t top_index, int32_t bottom_index) {
        point tmp = arr[top_index];
        while (top_index <= (bottom_index - 1) / 2) {
            point max_one = tmp;
            int32_t child_idx = 0;
            int32_t left_child_idx = top_index * 2 + 1;
            int32_t right_child_idx = top_index * 2 + 2;
            
            if (left_child_idx <= bottom_index && max_one.semi > arr[left_child_idx].semi ) {
                max_one = arr[left_child_idx];
                child_idx = left_child_idx;
            }
            if (right_child_idx <= bottom_index && max_one.semi > arr[right_child_idx].semi ) {
                max_one = arr[right_child_idx];
                child_idx = right_child_idx;
            }
          
            if (max_one.semi != tmp.semi) {
                arr[top_index] = max_one;
                top_index = child_idx;
            }
            else {
                break;
            }
        }
        arr[top_index] = tmp;
    }

    void    sort() {
        // build  heap first
        build_heap_from_bottom_to_top();

        // sort
        point tmp;
        for (int32_t i = n - 1; i > 0;) {
            // move heap top to end
            tmp = arr[0];
            arr[0] = arr[i];
            arr[i] = tmp;

            // adjust the heap
            heap_adjust_from_top_to_bottom(0, --i);
        }
    }

};

void top_k(vector<point>& input_arr, int32_t n, int32_t k) {
    // O(k)
    // we suppose the k element of the min heap if the default top k element
    min_heap_t min_heap(input_arr, k);
    min_heap.build_heap_from_bottom_to_top();
    
    for (int32_t i = k; i < n; ++i) {
        // compare each element with the min element of the min heap
        // if the element > the min element of the min heap
        // we think may be the element is one of what we wanna to find in the top k
        if (input_arr[i].semi > min_heap.arr[0].semi){
            // swap
            min_heap.arr[0] = input_arr[i];
            
            // heap adjust
            min_heap.heap_adjust_from_top_to_bottom(0, k - 1);
        }
    }
    
    input_arr.assign(min_heap.arr.begin(),min_heap.arr.end());
}


int ExactSP(caffe::shared_ptr< caffe::Net<float> > net_, const float* grey, std::vector<cv::KeyPoint>& kpts, std::vector<std::vector<float> >& dspts, int H, int W){
  int Height = 480;
  int Width = 640;
  int Cell = 8;
  int D = 256;
  int Feature_Length = 65;
  int NMS_Threshold = 4;
  int KEEP_K_POINTS = 200;
  // int width = Heigh;
  // int height = Width;
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  float* input_data = input_layer->mutable_cpu_data();
//   LOG(INFO) << Width << "  "<< Height;
                            std::chrono::steady_clock::time_point timepoint[10];
                            timepoint[0] =std::chrono::steady_clock::now();
  memcpy(input_data, grey, Width*Height*sizeof(float) );
                            timepoint[1] =std::chrono::steady_clock::now();
  net_->Forward();
                            timepoint[2] =std::chrono::steady_clock::now();
  std::vector< caffe::Blob<float>* > output_layers = net_->output_blobs();
//   LOG(INFO) << output_layers.size();
  for(int i : {0,1,2,3}){
    LOG(INFO) << output_layers[0]->shape()[i] << "  " << output_layers[1]->shape()[i];
  }
    int num_semi = 1*Feature_Length*Height/Cell*Width/Cell;
    // float* result_semi = new float[num_semi];
    // float* result_desc = new float[1*D*Height/Cell*Width/Cell];
    float* result_semi = output_layers[0]->mutable_cpu_data();
    float* result_desc = output_layers[1]->mutable_cpu_data();

    // for(int i=0; i<num_semi; i++) {
	// 	    result_semi[i] = exp(result_semi[i]); //e^x
    //     // result_semi[i] = pow(2, result_semi[i]); //2^x
    //     // result_semi[i] = pow(4, result_semi[i]); //4^x
	//   }

    float semi[Height][Width];
    // point coarse_semi[Height/Cell][Width/Cell];
    // float coarse_desc[Height/Cell][Width/Cell][D];
                            timepoint[3] =std::chrono::steady_clock::now();
    
    // cout << "\nRun normalize ..." << endl;
    for(int i=0; i<Height/Cell; i++) {
        for(int j=0; j<Width/Cell; j++) {
            //semi softmax
            // float cell_sum = 0;
            // for(int k=0; k<Feature_Length; k++) {
            //     cell_sum = cell_sum + result_semi[k+j*Feature_Length+i*Feature_Length*Width/Cell];
            // }
            for(int kh=0; kh<Cell; kh++) {
                for(int kw=0; kw<Cell; kw++) {
                    // semi[kh+i*Cell][kw+j*Cell] = result_semi[kw+kh*Cell+j*Feature_Length+i*Feature_Length*Width/Cell]/cell_sum;
                    semi[kh+i*Cell][kw+j*Cell] = result_semi[kw+kh*Cell+j*Feature_Length+i*Feature_Length*Width/Cell];
                    // LOG(INFO) << kh+i*Cell << " " << kw+j*Cell << " " << semi[kh+i*Cell][kw+j*Cell];
                }
            }
            
            //max 1 point
            /* float max_semi=0;
            for(int kh=0; kh<Cell; kh++) {
                for(int kw=0; kw<Cell; kw++) {
                    if(semi[kh+i*Cell][kw+j*Cell] > max_semi) {
                        max_semi = semi[kh+i*Cell][kw+j*Cell];
                        coarse_semi[i][j].H = kh+i*Cell;
                        coarse_semi[i][j].W = kw+j*Cell;
                        coarse_semi[i][j].semi = max_semi;
                    }
                }
            } */
            
            //desc normalize
// float desc_sum_2 = 0;
// for(int k=0; k<D; k++) {
//     desc_sum_2 = desc_sum_2 + pow(result_desc[k+j*D+i*D*Width/Cell],2);
// }
// float desc_sum = sqrt(desc_sum_2);
// for(int k=0; k<D; k++) {
//     coarse_desc[i][j][k] = result_desc[k+j*D+i*D*Width/Cell]/desc_sum;
//     // coarse_desc[i][j][k] = (float)(int)(result_desc[k+j*D+i*D*Width/Cell]/desc_sum*512);
//     // coarse_desc[i][j][k] = coarse_desc[i][j][k]>127? 127:coarse_desc[i][j][k];
//     // coarse_desc[i][j][k] = coarse_desc[i][j][k]<-128? -128:coarse_desc[i][j][k];
// }
        }
    }
                            timepoint[4] =std::chrono::steady_clock::now();

    std::vector<point> tmp_point;
    
    //NMS
    for(int i=0; i<Height; i++) {
        for(int j=0; j<Width; j++) {
            if(semi[i][j] != 0) {
                float tmp_semi = semi[i][j];
                for(int kh=std::max(0,i-NMS_Threshold); kh<std::min(Height,i+NMS_Threshold+1); kh++)
                    for(int kw=std::max(0,j-NMS_Threshold); kw<std::min(Width,j+NMS_Threshold+1); kw++)
                        if(i!=kh||j!=kw) {
                            if(tmp_semi>=semi[kh][kw])
                                semi[kh][kw] = 0;
                            else
                                semi[i][j] = 0;
                        }
                if(semi[i][j]!=0)
                    // LOG(INFO) << "tmp_point : " << i << "  " << j;
                    tmp_point.push_back(point(i,j,semi[i][j]));
            }
        }
    }
    
                            timepoint[5] =std::chrono::steady_clock::now();
    top_k(tmp_point,tmp_point.size(),KEEP_K_POINTS);
                            timepoint[6] =std::chrono::steady_clock::now();
    // cv::Mat desc( int(tmp_point.size()), D, CV_32FC1);
    // std::vector<cv::KeyPoint> points;
    dspts.clear();
    kpts.clear();


        // for (int j = 0; j < D ; j++ ){
        // // std::cout <<  result_desc[0 + j] << " ";
        // std::cout <<  output_layers[1]->data_at(0,0,0,j) << " ";
        // if (j % 4 == 3){
        // std::cout << std::endl;
        // }
        // }
    // std::cout << int(tmp_point[0].W / Cell) << std::endl ;
    // std::cout << int(tmp_point[0].H / Cell) << std::endl ;

    for(int i=0; i<tmp_point.size(); i++) {
        kpts.push_back(cv::KeyPoint(tmp_point[i].W, tmp_point[i].H, 0, 0));
        std::vector<float> DataDesc(D);   //第i+1行的所有元素  
        // if(tmp_point[i].W == 149){
        //     std::cout << i << " " << tmp_point[i].W << " " <<  tmp_point[i].H<< " "  <<  tmp_point[i].semi << std::endl;
        // }
        int x1 = int(tmp_point[i].W / Cell);
        int x2 = int(tmp_point[i].H / Cell);
        for(int j = 0; j < D; j++){
            DataDesc[j] = result_desc[ x1*D + x2*(D*Width/Cell) +j];
        }
        dspts.push_back(DataDesc);
    }
    
    timepoint[5] =std::chrono::steady_clock::now();
                            for (int i = 0; i < 7; i ++ ){
                                    LOG(INFO) << " time step: " << i << " : " << std::chrono::duration<double,std::milli>(timepoint[i+1] - timepoint[i]).count() << std::endl;
                                }
    return 0;
                            

}