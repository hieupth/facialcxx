#ifndef SPOOFING_H
#define SPOOFING_H

#include <ortcxx/pipeline.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

using namespace cv;
using namespace std;
using namespace ortcxx::pipeline;
using namespace ortcxx::model;


void cropFace(
    cv::Mat src, cv::Mat& dst, 
    cv::Rect rectPoint, 
    int out_height, int out_width
); 

class Spoofing
{
    protected:
        shared_ptr<Model> extractor;
        shared_ptr<Model> classifier;

    public:
        Spoofing(shared_ptr<Model> extractor, shared_ptr<Model> classifier);
        void preprocess(cv::Mat input, cv::Mat &output);
        std::vector<float> postprocess(std::vector<float> input);
        std::vector<float> inference(cv::Mat input);
        Ort::Value createOrtValueFromMat(const cv::Mat& mat);
    };

#endif // SPOOFING_H