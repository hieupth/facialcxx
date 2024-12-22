#include <string>
#include <math.h>
#include <ortcxx/model.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

// #include <blazeface.h>
#include <spoofing.h>
#include <yolov7.h>
#include <filesystem>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;
using namespace Ort;
using namespace ortcxx;

// BlazeFace* blazefacePipeline;
Spoofing* SpoofingPipeline;
YoloV7* yoloPipeline;


std::vector<float> checkSpoof(cv::Mat& input_mat) {
	cvtColor(input_mat, input_mat, COLOR_RGBA2BGR);

	cv::Mat original = input_mat.clone();
    cv::Mat processed_img = yoloPipeline->preprocess(input_mat);
    std::vector<PredictResultHeadFace> res_yolo;
    
	yoloPipeline->detect(processed_img, &res_yolo);
	// printf("res_yolo.size(): %ld\n", res_yolo.size());

	if (res_yolo.size() > 0) {
		// Crop the face based on the bounding box
		vector<float> outputs;

		for (int i = 0; i < res_yolo.size(); i++) {
			
			PredictResultHeadFace res = res_yolo[i];
			float score = (res.score_face == 0.0) ? res.score_head : res.score_face;

			cv::Rect bbox = cv::Rect(
				static_cast<int>(std::round(res.xmin)),
				static_cast<int>(std::round(res.ymin)),
				static_cast<int>(std::round(res.xmax - res.xmin)),
				static_cast<int>(std::round(res.ymax - res.ymin))
			);
			
			cv::Mat cropped_face;
			cropFace(original, cropped_face, bbox, 256, 256);
			cvtColor(cropped_face, cropped_face, COLOR_BGR2RGB);
			cv::imwrite("/home/tiennv/hungnq/facialcxx/check_cropped_face_" + std::to_string(i) + ".jpg", cropped_face);

			// Inference
			std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
			std::vector<float> output_spoof = SpoofingPipeline->inference(cropped_face);
			outputs.push_back(output_spoof[0]);
			outputs.push_back(output_spoof[1]);
			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration = end - start;
			std::cout << "Time taken to runSpoof: " << duration.count() << " seconds." << std::endl;
		}
		
		return outputs;
	} else {
		// Return an empty vector if no faces are detected
		return std::vector<float>();
	}
}


int main() {

	std::string extractorPath = "...";
	std::string embedderPath = "...";;
	std::string yoloPath = "...";;

	// Load the model
	auto providers = Ort::GetAvailableProviders();
	for (const std::string& p : providers) {
		std::cout << "- " << p << std::endl;
	}
	map<string, any> c;
	c["parallel"] = false;
	c["inter_ops_threads"] = 1;
	c["intra_ops_threads"] = 1;
	c["graph_optimization_level"] = 1;
	c["session.use_env_allocators"] = true;

	// Load the model with shared env and allocator
	std::shared_ptr<Ort::Env> ortEnv = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
	ortEnv->CreateAndRegisterAllocator(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault), {});

	// std::shared_ptr<Model> blazefaceModel = Model::create(blazeFacePath, ortEnv, c, providers, false);
	printf(	"Model loaded\n");
	std::shared_ptr<Model> extractorModel = make_shared<Model>(extractorPath, ortEnv, c, providers, false);
	std::shared_ptr<Model> embedderModel = make_shared<Model>(embedderPath, ortEnv, c, providers, false);
	std::shared_ptr<Model> yoloModel = make_shared<Model>(yoloPath, ortEnv, c, providers, false);

	// Assign pointers
	// blazefacePipeline = new BlazeFace(blazefaceModel);
	SpoofingPipeline = new Spoofing(extractorModel, embedderModel);
	yoloPipeline = new YoloV7(yoloModel);

	
	// Load the image
	std::string path = "...";
	for (const auto & entry : fs::directory_iterator(path)){
		std::cout << entry.path() << std::endl;
		// Create a 4D Mat with CV_32F (float) data type, initialized with zeros
		std::string image_path = entry.path();
		cv::Mat image;
		try
		{
			image = cv::imread(image_path, cv::IMREAD_COLOR);
		}
		catch(const cv::Exception& e)
		{
			std::cerr << e.what() << '\n';
		}
		cv::Mat original = image.clone();
		//check spoof
		auto start = std::chrono::high_resolution_clock::now();
		std::vector<float> outputs_check = checkSpoof(original);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = end - start;
		std::cout << "Time taken to checkSpoof: " << duration.count() << " seconds." << std::endl;

		printf("Check spoofing: ");
		for (int i = 0; i < outputs_check.size(); i++) {
			std::cout << outputs_check[i] << " ";
		}
		std::cout << std::endl;

	}
	
	return 0;
	
}

