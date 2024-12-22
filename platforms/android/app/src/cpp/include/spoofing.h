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

const float SRC_MAP[5][5][2]={
    // left
    {{51.642, 50.115}, {57.617, 49.990}, {35.740, 69.007}, {51.157, 89.050}, {57.025, 89.702}},
    // lef profile
    {{45.031, 50.118}, {65.568, 50.872}, {39.677, 68.111}, {45.177, 86.190}, {64.246, 86.758}},
    // frontal
    {{39.730, 51.138}, {72.270, 51.138}, {56.000, 68.493}, {42.463, 87.010}, {69.537, 87.010}},
    // right
    {{46.845, 50.872}, {67.382, 50.118}, {72.737, 68.111}, {48.167, 86.758}, {67.236, 86.190}},
    // right frontal
    {{54.796, 49.990}, {60.771, 50.115}, {76.673, 69.007}, {55.388, 89.702}, {61.257, 89.050}}
};

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
        Ort::Value createOrtValueFromMat(cv::Mat& mat);
    };

#endif // SPOOFING_H