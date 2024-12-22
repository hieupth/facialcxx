
#include <numeric>
#include <algorithm>
#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>
#include "ortcxx/model.h"

using namespace cv;
using namespace std;
using namespace ortcxx::model;

struct PredictResultHeadFace
{
    float score_obj = 0.0;
    float score_head = 0.0;
    float score_face = 0.0;
    float ymin = 0.0;
    float xmin = 0.0;
    float ymax = 0.0;
    float xmax = 0.0;
    float keypoints[10];
};

class YoloV7 {
    public:
        YoloV7(shared_ptr<Model> yolo);
        ~YoloV7();
        cv::Mat preprocess(const cv::Mat img);
        std::pair<std::vector<PredictResultHeadFace>, std::vector<PredictResultHeadFace>> infer(const cv::Mat &processed_img);
        void detect(const cv::Mat processed_img, std::vector<PredictResultHeadFace> *res);
        void postprocess(const std::vector<PredictResultHeadFace> &output, std::vector<PredictResultHeadFace> *res);
    protected:
        shared_ptr<Model> yolo;

    private:
        float dw;
        float dh;
        float ratio;
        cv::Size img_shape;

        const float SCORE_THRESHOLD_ = 0.3f;
        const float IOU_THRESHOLD_ = 0.52f;
        const cv::Size INPUT_SHAPE = cv::Size(640, 640);
        const cv::Scalar COLOR_ = cv::Scalar(114, 114, 114);
};


