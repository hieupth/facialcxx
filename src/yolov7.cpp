#include <numeric>
#include <algorithm>
#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>
#include "yolov7.h"

using namespace cv;
using namespace std;
using namespace ortcxx::model;
using namespace Ort;

std::pair<float, int> IoU(const PredictResultHeadFace &box1, const PredictResultHeadFace &box2)
{
    float x1 = std::max(box1.xmin, box2.xmin);
    float y1 = std::max(box1.ymin, box2.ymin);
    float x2 = std::min(box1.xmax, box2.xmax);
    float y2 = std::min(box1.ymax, box2.ymax);

    float width = std::max(0.0f, x2 - x1 + 1);
    float height = std::max(0.0f, y2 - y1 + 1);
    float interArea = width * height;

    float box1Area = (box1.xmax - box1.xmin + 1) * (box1.ymax - box1.ymin + 1);
    float box2Area = (box2.xmax - box2.xmin + 1) * (box2.ymax - box2.ymin + 1);

    if (box1Area > box2Area)
    {
        return {interArea / box2Area, 0};
    }

    return {interArea / box1Area, 1};
};

void check_overlap(std::vector<PredictResultHeadFace> &res, float overlap)
{
    for (size_t i = 0; i < res.size(); i++)
    {
        for (size_t j = i + 1; j < res.size(); j++)
        {
            auto [iou, idx] = IoU(res[i], res[j]);
            if (iou > overlap)
            {
                if (idx == 0)
                {
                    res[i].score_obj = 0.0;
                }
                else
                {
                    res[j].score_obj = 0.0;
                }
            }
        }
    }

    res.erase(std::remove_if(res.begin(), res.end(), [](const PredictResultHeadFace &face)
                             { return face.score_obj == 0.0; }),
              res.end());
};


YoloV7::YoloV7(shared_ptr<Model> yolo) {
    this->yolo = yolo;

};

YoloV7::~YoloV7() {
    // Destructor
    this->yolo = nullptr;
};


cv::Mat YoloV7::preprocess(
    const cv::Mat img
) {
    this->img_shape = img.size(); //BUGHERE
    
    //attribute
    this->ratio = std::min(
        static_cast<float>(this->INPUT_SHAPE.height) / img_shape.height,
        static_cast<float>(this->INPUT_SHAPE.width) / img_shape.width
    );
    this->ratio = std::min(this->ratio, 1.0f);

    int new_unpad_w = static_cast<int>(std::round(img_shape.width * this->ratio));
    int new_unpad_h = static_cast<int>(std::round(img_shape.height * this->ratio));
    //attributes
    dw = (this->INPUT_SHAPE.width - new_unpad_w) / 2;
    dh = (this->INPUT_SHAPE.height - new_unpad_h) / 2;

    cv::Mat resized_img;
    if (img_shape != cv::Size(new_unpad_w, new_unpad_h))
    {
        cv::resize(img, resized_img, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LANCZOS4);
    }
    else
    {
        resized_img = img;
    }

    int top = static_cast<int>(std::round(dh - 0.1));
    int bottom = static_cast<int>(std::round(dh + 0.1));
    int left = static_cast<int>(std::round(this->dw - 0.1));
    int right = static_cast<int>(std::round(this->dw + 0.1));

    cv::Mat processed_img;
    cv::copyMakeBorder(resized_img, processed_img, top, bottom, left, right, cv::BORDER_CONSTANT, COLOR_);

    return processed_img;
};

std::pair<std::vector<PredictResultHeadFace>, std::vector<PredictResultHeadFace>> YoloV7::infer(const cv::Mat &processed_img)
{
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    cv::Mat blob = cv::dnn::blobFromImage(processed_img, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), false, false);
    size_t input_tensor_size = blob.total();
    std::vector<int64_t> input_node_dims = {1, 3, 640, 640};
   
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        (float *)blob.data, 
        input_tensor_size, 
        input_node_dims.data(), 
        input_node_dims.size()
    );

    std::vector<Ort::Value> inputTensorValues;
    inputTensorValues.push_back(std::move(tensor));

    //remove
    // std::vector<const char *> input_node_names = {"images"};
    // std::vector<const char *> output_node_names = {"head", "face"};

    // Run the model
    auto outputTensor = this->yolo->run(
        inputTensorValues, 
        shared_ptr<const char*>(), 
        Ort::RunOptions{}
    );

    auto head_result = outputTensor->at(0).GetTensorMutableData<float>();
    auto face_result = outputTensor->at(1).GetTensorMutableData<float>(); //!BUG HERE

    // Extract the output tensors with Head shape (N, 6) and Face shape (N, 21)
    std::vector<PredictResultHeadFace> results_head;
    std::vector<PredictResultHeadFace> results_face;

    size_t num_head = outputTensor->at(0).GetTensorTypeAndShapeInfo().GetElementCount();
    size_t num_face = outputTensor->at(1).GetTensorTypeAndShapeInfo().GetElementCount();

    
    for (int i = 0; i < num_head; i += 7)
    {
        PredictResultHeadFace head;
        head.xmin = head_result[i + 1];
        head.ymin = head_result[i + 2];
        head.xmax = head_result[i + 3];
        head.ymax = head_result[i + 4];
        head.score_obj = head.score_head = head_result[i + 6];
        results_head.push_back(head);
    }

    for (int i = 0; i < num_face; i += 22)
    {
        PredictResultHeadFace face;
        face.xmin = face_result[i + 1];
        face.ymin = face_result[i + 2];
        face.xmax = face_result[i + 3];
        face.ymax = face_result[i + 4];
        face.score_obj = face.score_face = face_result[i + 6];
        int x = 0;
        for (size_t j = 0; j < 10; j += 2)
        {
            face.keypoints[j] = face_result[i + j + 7 + x];
            face.keypoints[j + 1] = face_result[i + j + 8 + x];
            x++;
        }
        results_face.push_back(face);
    }

    return {results_head, results_face};
};

void YoloV7::detect(const cv::Mat processed_img, std::vector<PredictResultHeadFace> *res)
{
    auto [head_output, face_output] = infer(processed_img);

    // Map the output to the result -> HARDCODE HERE
    // for (int tensorIndex = 0; tensorIndex < 2; ++tensorIndex)
    // {
    //     postprocess(tensorIndex == 0 ? head_output : face_output, res);
    // }
    //
    postprocess(face_output, res);
    check_overlap(*res, this->IOU_THRESHOLD_);
}

void YoloV7::postprocess(
    const std::vector<PredictResultHeadFace> &output,
    std::vector<PredictResultHeadFace> *res
) {
    for (PredictResultHeadFace row : output)
    {
        std::vector<float> det_bboxes = {row.xmin, row.ymin, row.xmax, row.ymax};
        float det_score = row.score_obj;

        if (det_score > this->SCORE_THRESHOLD_)
        {
            PredictResultHeadFace face;

            for (size_t i = 0; i < 4; i += 2)
            {
                det_bboxes[i] = (det_bboxes[i] - this->dw) / this->ratio;
                det_bboxes[i + 1] = (det_bboxes[i + 1] - this->dh) / this->ratio;
            }
            face.ymin = std::max(0.0f, det_bboxes[1]);
            face.xmin = std::max(0.0f, det_bboxes[0]);
            face.ymax = std::min(this->img_shape.height - 1.0f, det_bboxes[3]);
            face.xmax = std::min(this->img_shape.width - 1.0f, det_bboxes[2]);
            face.score_obj = det_score;
            face.score_head = row.score_head;
            face.score_face = row.score_face;
            

            if (row.score_face != 0.0)
            {
                for (size_t i = 0; i < 10; i += 2)
                {
                    face.keypoints[i] = (row.keypoints[i] - this->dw) / this->ratio;
                    face.keypoints[i + 1] = (row.keypoints[i + 1] - this->dh) / this->ratio;
                }
            }
            res->push_back(face);
        }
    }
};

