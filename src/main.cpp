#include "network.h"
#include "mtcnn.h"
#include<math.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgcodecs/imgcodecs.hpp> 
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include <iostream>
#include <string>
#include <fstream>
using namespace cv;
using namespace std;
#define DISP_WINNANE "camera"
#define QUIT_KEY     'q'
#define CAMID         0
using namespace tensorflow;
using tensorflow::Tensor;
using tensorflow::Status;
using namespace std;
const int height = 160;
const int width = 160;
const int depth = 3;
float * result = new float[128];
const string input_layer_1 = "input:0";
const string input_layer_2 = "phase_train:0";
const string output_layer = "embeddings:0";

typedef struct faceID {
  string picture;
  string name;
  float* embedding;
} faceID;

std::vector<faceID> faceIDs;
void getImageTensor(tensorflow::Tensor &input_tensor, Mat& Image){

    resize(Image, Image, Size(160, 160));
    cv::int64_t start=getTickCount();
 //   cv::Mat Image = cv::imread(path);
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();

    //mean and std
    //c * r * 3 => c * 3r * 1
    cv::Mat temp = Image.reshape(1, Image.rows * 3);

    cv::Mat mean3;
    cv::Mat stddev3;
    cv::meanStdDev(temp, mean3, stddev3);

    double mean_pxl = mean3.at<double>(0);
    double stddev_pxl = stddev3.at<double>(0);

    //prewhiten
    Image.convertTo(Image, CV_64FC1);
    //cv::Mat tempmean = cv::Vec3d(mean_pxl, mean_pxl, mean_pxl);
    //std::cout<<tempmean.size()<<std::endl; 
    //Image = Image - cv::Vec3d(mean_pxl, mean_pxl, mean_pxl);
    Image = Image - cv::Vec<double,1>(mean_pxl);
    Image = Image / stddev_pxl;
   
     
    // copying the data into the corresponding tensor
    for (int y = 0; y < height; ++y) {
        const double* source_row = Image.ptr<double>(y);
        for (int x = 0; x < width; ++x) {
            const double* source_pixel = source_row + (x * depth);
            for (int c = 0; c < depth; ++c) {
                const double* source_value = source_pixel + (2-c);//RGB->BGR
                input_tensor_mapped(0, y, x, c) = *source_value;
            }
        }
    }
     cout<<"The image preprocess cost "<<1000 *(double)(getTickCount()-start)/getTickFrequency()<<" ms"<<endl;
}
double Recogize(const std::unique_ptr<tensorflow::Session> &session, Tensor& image, float * res){

    Tensor phase_train(tensorflow::DT_BOOL, tensorflow::TensorShape());
    phase_train.scalar<bool>()() = false;

    std::vector<Tensor> outputs;
    cv::int64_t start=getTickCount();
    Status run_status = session->Run({{input_layer_1, image},
    							 {input_layer_2,phase_train }},
    							 {output_layer},
    							 {},
    							 &outputs);
    // cout<<"The network cost "<<1000 * (double)(getTickCount()-start)/getTickFrequency()<<" ms"<<endl;
    // cout<<outputs[0].DebugString()<<endl;

    if(!run_status.ok()){
        LOG(ERROR) << "Running model failed"<<run_status;
        return 0;
    }
    auto outMap = outputs[0].tensor<float, 2>();
    if(res != NULL){
        for(int i = 0; i < 128; i++)
            res[i] = outMap(i);
    }else{
        double sum = 0;
        for(int i = 0; i < 128; i++)
            sum += (outMap(i) - result[i]) * (outMap(i) - result[i]) ;
        return sqrt(sum);
    }
    return 0;
}

void getSession(string graph_path, std::unique_ptr<tensorflow::Session> &session){
    tensorflow::GraphDef graph_def;
    if (!ReadBinaryProto(tensorflow::Env::Default(), graph_path, &graph_def).ok()) {
        LOG(ERROR) << "Read proto";
        return ;
    }
    tensorflow::SessionOptions sess_opt;
    sess_opt.config.mutable_gpu_options()->set_allow_growth(true);

    (&session)->reset(tensorflow::NewSession(sess_opt));

    if (!session->Create(graph_def).ok()) {
        LOG(ERROR) << "Create graph";
        return ;
    }
}

double calculateL2(float * a, float * b){
        double sum = 0;
        for(int i = 0; i < 128; i++)
            sum += (a[i] - b[i]) * (a[i] - b[i]) ;
        return sqrt(sum);
}

int main(int argc, const char * argv[])
{
	if(argc < 3){
		cout<<"Please specify your photo and your name on common line";
		return -1;
	}
    size_t pos = 0;
    
    std::vector<string> names;
    std::vector<string> pictures;
    
    string graph_path = "./model/20170512-110547.pb";
    string testName = argv[2];
    //Mat chenlian = imread(argv[1]);
    Tensor chen(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1,height,width,depth }));
    Tensor img(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1,height,width,depth }));
    unique_ptr<tensorflow::Session> session;

    //facenet model initialize
    getSession(graph_path, session);
    //project the cv::Mat into tensorflow::Tensor
        
    ifstream file("./model/database.txt");
    if(file.is_open())
    {
      string line;
      while(file >> line){
        pos = line.find("|", 0);
        faceID temp;
        temp.name = (line.substr(pos+1, line.size()));
        temp.picture = (line.substr(0,pos));
        temp.embedding = new float[128];
        std::cout<<"names: "<<temp.name<<std::endl;
        std::cout<<"pictures: "<<temp.picture<<std::endl;
        Mat chenlian = imread(temp.picture);
        getImageTensor(chen, chenlian);
        //for model test
        Recogize(session, chen, temp.embedding);
        faceIDs.push_back(temp);
      }

      file.close();
    }

    
        //chenlian = imread(argv[1]);
        //project the cv::Mat into tensorflow::Tensor
        //getImageTensor(chen, chenlian);
        //for model test
        //Recogize(session, chen, result);


    Mat image;
    double ftick, etick;
    double ticksPerUs;

    cv::VideoCapture camera(CAMID);
    ticksPerUs = cv::getTickFrequency() / 1000000;
    if (!camera.isOpened()) {
        std::cerr << "failed to open camera" << std::endl;
        return -1;
    }
                //col row
    mtcnn find(800, 480);

    do {
      cv::int64_t start=getTickCount();
        camera >> image;
            
        resize(image, image, Size(800, 480));
        ftick = cv::getCPUTickCount();
        find.findFace(image);
        etick = cv::getCPUTickCount();
        // std::cout<<"faceDetect cost: "<<(etick - ftick)/ticksPerUs / 1000<<" ms"<<std::endl;
        int gap = 30;
        for(vector<struct Bbox>::iterator it=find.thirdBbox_.begin(); it!=find.thirdBbox_.end();it++)
            if((*it).exist){
                Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
                Mat face = image(temp).clone();
                getImageTensor(img, face);
                double r = Recogize(session, img, result);
                r = 100;
                for(auto faceid : faceIDs){
                  double tempr = calculateL2(faceid.embedding, result);
                  std::cout<<tempr<<"L2distance with"<<faceid.name<<std::endl;
                  if ( tempr < r){
                    testName = faceid.name;
                    r = tempr;
                  }
                }  
                if(r < 0.6)
                    putText(image,testName,Point(50,gap),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,8);
                else
                    putText(image,"others",Point(50,gap),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,8);
                gap += 30;
            }

        cv::imshow(DISP_WINNANE, image);
        find.thirdBbox_.clear();
        cout<<"The whole process cost "<<1000 * (double)(getTickCount()-start)/getTickFrequency()<<" ms"<<endl;
            

    } while (QUIT_KEY != cv::waitKey(1));
    return 0;
}
