#include "spoofing.h"
#include <iostream>
#include <ortcxx/model.h>
#include <ortcxx/pipeline.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <PillowResize.hpp>

using namespace cv;
using namespace std;
using namespace ortcxx::pipeline;
using namespace ortcxx::model;

Spoofing::Spoofing(shared_ptr<Model> extractor, shared_ptr<Model> classifier)
{
    // Additional initialization if needed
    this->extractor = extractor;
    this->classifier = classifier;
}

void Spoofing::preprocess(cv::Mat input, cv::Mat &output) {
    // Implement preprocessing logic here
    std::cout << "Spoofing Preprocessing..." << std::endl;
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
    //resize
    cv::Mat resized_image;
//    cv::resize(input, resized_image, cv::Size(256, 256));
    resized_image = PillowResize::resize(input, cv::Size(256, 256), 2);
    // Normalize the image 0- 1
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);
    // NEED TO FIX 0.5 / 0.5 ?
    resized_image = (resized_image - 0.5) / 0.5;
    output = resized_image;
}

std::vector<float> Spoofing::postprocess(std::vector<float> input) {
    // Implement postprocessing logic here
    std::cout << "Spoofing Postprocessing..." << std::endl;

    // Apply softmax
    std::vector<float> softmax_output(input.size());
    // float max_val = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        // softmax_output[i] = std::exp(input[i] - max_val);
        softmax_output[i] = std::exp(input[i]);
        sum += softmax_output[i];
    }
    for (size_t i = 0; i < input.size(); ++i) {
        softmax_output[i] /= sum;
    }

    return softmax_output;
}

std::vector<float> Spoofing::inference(cv::Mat input) {
    // Preprocess the input
    cv::Mat preprocessed_input;
    preprocess(input, preprocessed_input);

    // Convert preprocessed input to Ort::Value
    Ort::Value input_tensor = this->createOrtValueFromMat(preprocessed_input);
    auto dims_input = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
    // print dims2
    for (int i = 0; i < dims_input.size(); i++) {
        std::cout << "dims_input: " << dims_input[i] << std::endl;
    }

    // Run the extractor model
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_tensor));
    
    
    auto extract_outputs = this->extractor->run(
        inputs, 
        shared_ptr<const char*>(),
        Ort::RunOptions()
    );

    // Get the second output from the extractor
    auto second_output = std::move(extract_outputs->at(1));

    // Run the classifier model with the second output
    std::vector<Ort::Value> classifier_inputs;
    classifier_inputs.push_back(std::move(second_output));
    auto classifier_outputs = this->classifier->run(
        classifier_inputs, 
        std::shared_ptr<const char*>(),
        Ort::RunOptions()
    );

    // Convert the final output to a vector of floats
    auto final_output = classifier_outputs->at(0).GetTensorMutableData<float>();
    std::vector<float> output_vector;
    for (int i = 0; i < 2; i++) {
        output_vector.push_back(final_output[i]);
    }; //MAKE SURE THIS IS CORRECT

    // Postprocess the output
    return postprocess(output_vector);
}

Ort::Value Spoofing::createOrtValueFromMat(cv::Mat& mat) {
    //* USE THIS INSTEAD CREATE FROM SCRATCH
    cv::Mat blob = cv::dnn::blobFromImage(mat, 1.0, cv::Size(256, 256), (0, 0, 0), false, false, CV_32F);
    size_t input_tensor_size = blob.total();
    // Get the dimensions of the input mat
    std::vector<int64_t> dims = {1, mat.channels(), mat.rows, mat.cols};
    // Create an Ort::MemoryInfo object
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(
        memory_info, 
        (float*)blob.data, 
        input_tensor_size, 
        dims.data(), 
        dims.size()
    );
}


void cropFace(cv::Mat src, cv::Mat& dst, cv::Rect rectPoint, int out_height, int out_width) {
    // warp
    if (rectPoint.x < 0) {
        rectPoint.x = 0;
    }
    if (rectPoint.y < 0) {
        rectPoint.y = 0;
    }
    if (rectPoint.width > src.cols) {
        rectPoint.width = src.cols;
    }
    if (rectPoint.height > src.rows) {
        rectPoint.height = src.rows;
    }
    // crop
    Mat ROI(src, rectPoint);
    dst = ROI.clone();
}
