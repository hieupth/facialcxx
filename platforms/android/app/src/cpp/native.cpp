#include <jni.h>
#include <string>
#include <math.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <ortcxx/model.h>
#include <opencv2/opencv.hpp>

// #include <blazeface.h>
#include <spoofing.h>
#include <yolov7.h>

using namespace cv;
using namespace std;
using namespace Ort;
using namespace ortcxx;


////////////////////////
#define TENSOR(METHOD_NAME) \
  Java_ai_spoofing_TensorUtils_##METHOD_NAME
////////////////////////
#ifdef __cplusplus
extern "C" {
#endif


// BlazeFace* blazefacePipeline;
YoloV7* yoloPipeline;
Spoofing* SpoofingPipeline;

std::map<int, int> EXIF_ORIENTATION_2_OPENCV_ROTATION = {
    {90, cv::ROTATE_90_CLOCKWISE},
    {180, cv::ROTATE_180},
    {270, cv::ROTATE_90_COUNTERCLOCKWISE}
};

// Convert the bitmap to OpenCV Mat
// Reference https://github.com/opencv/opencv/blob/17234f82d025e3bbfbf611089637e5aa2038e7b8/modules/java/generator/src/cpp/utils.cpp
void bitmapToMat(JNIEnv *env, jobject bitmap, Mat &dst, jboolean needUnPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;

    try {
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        dst.create(info.height, info.width, CV_8UC4);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (needUnPremultiplyAlpha) cvtColor(tmp, dst, COLOR_mRGBA2RGBA);
            else tmp.copyTo(dst);
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            cvtColor(tmp, dst, COLOR_BGR5652RGBA);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
        return;
    }
}

// Convert a mat to Bitmap
// Reference https://github.com/opencv/opencv/blob/17234f82d025e3bbfbf611089637e5aa2038e7b8/modules/java/generator/src/cpp/utils.cpp
void matToBitmap(JNIEnv *env, Mat src, jobject bitmap, jboolean needPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;

    try {
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(src.dims == 2);
        CV_Assert(info.height == (uint32_t) src.rows);
        CV_Assert(info.width == (uint32_t) src.cols);
        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (src.type() == CV_8UC1) {
                cvtColor(src, tmp, COLOR_GRAY2RGBA);
            } else if (src.type() == CV_8UC3) {
                cvtColor(src, tmp, COLOR_RGB2RGBA);
            } else if (src.type() == CV_8UC4) {
                if (needPremultiplyAlpha) cvtColor(src, tmp, COLOR_RGBA2mRGBA);
                else src.copyTo(tmp);
            }
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if (src.type() == CV_8UC1) {
                cvtColor(src, tmp, COLOR_GRAY2BGR565);
            } else if (src.type() == CV_8UC3) {
                cvtColor(src, tmp, COLOR_RGB2BGR565);
            } else if (src.type() == CV_8UC4) {
                cvtColor(src, tmp, COLOR_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return;
    }
}


extern "C"
JNIEXPORT void JNICALL
Java_ai_spoofing_TensorUtils_initModel(
    JNIEnv *env, jclass clazz, jobject assetManager,
    jstring yoloPath,
    jstring extractorPath,
    jstring embedderPath
) {

    // YoloPath
    const char *yolo = env->GetStringUTFChars(yoloPath, 0);
    // convert to string
    std::string yoloString(yolo);
    //1. declaring character array
    char fhtpCharArray[yoloString.length() + 1];
    //2. copying the contents of the string to char array
    strcpy(fhtpCharArray, yoloString.c_str());
    //3. read model
    char *bufferYoloV7 = nullptr;
    long sizeYoloV7 = 0;
    if (!(env->IsSameObject(assetManager, NULL))) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        AAsset *asset = AAssetManager_open(mgr, fhtpCharArray, AASSET_MODE_UNKNOWN);
        assert(asset != nullptr);

        sizeYoloV7 = AAsset_getLength(asset);
        bufferYoloV7 = (char *) malloc(sizeof(char) * sizeYoloV7);
        AAsset_read(asset, bufferYoloV7, sizeYoloV7);
        AAsset_close(asset);
    }

    // Extractor Spoofing
    const char *extractor = env->GetStringUTFChars(extractorPath, 0);
    // convert to string
    std::string extractorString(extractor);
    // 1. declaring character array
    char extractorCharArray[extractorString.length() + 1];
    //2. copying the contents of the string to char array
    strcpy(extractorCharArray, extractorString.c_str());
    //3. read model
    char *bufferExtractor = nullptr;
    long sizeExtractor = 0;
    if (!(env->IsSameObject(assetManager, NULL))) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        AAsset *asset = AAssetManager_open(mgr, extractorCharArray, AASSET_MODE_UNKNOWN);
        assert(asset != nullptr);

        sizeExtractor = AAsset_getLength(asset);
        bufferExtractor = (char *) malloc(sizeof(char) * sizeExtractor);
        AAsset_read(asset, bufferExtractor, sizeExtractor);
        AAsset_close(asset);
    }

    // Embedder Spoofing
    const char *embed = env->GetStringUTFChars(embedderPath, 0);
    // convert to string
    std::string embedString(embed);
    //1. declaring character array
    char embedderCharArray[embedString.length() + 1];
    //2. copying the contents of the string to char array
    strcpy(embedderCharArray, embedString.c_str());
    //3. read model
    char *bufferEmbedder = nullptr;
    long sizeEmbedder = 0;
    if (!(env->IsSameObject(assetManager, NULL))) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        AAsset *asset = AAssetManager_open(mgr, embedderCharArray, AASSET_MODE_UNKNOWN);
        assert(asset != nullptr);

        sizeEmbedder = AAsset_getLength(asset);
        bufferEmbedder = (char *) malloc(sizeof(char) * sizeEmbedder);
        AAsset_read(asset, bufferEmbedder, sizeEmbedder);
        AAsset_close(asset);
    }

    // Load the model
    auto providers = Ort::GetAvailableProviders();
    for (std::string p : providers) {
        std::cout << "- " << p << std::endl;
    };
    map<string, any> c;
    c["parallel"] = false;
    c["inter_ops_threads"] = 1;
    c["intra_ops_threads"] = 1;
    c["graph_optimization_level"] = 1;
    c["session.use_env_allocators"] = true;

    // Load the model with share env and allocator
    std::shared_ptr<Ort::Env> ortEnv = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
    ortEnv->CreateAndRegisterAllocator(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault), {});

    std::shared_ptr<Model> yoloModel = Model::create(
        bufferYoloV7, sizeYoloV7, ortEnv, c, providers, false
    );
    std::shared_ptr<Model> extractorModel = Model::create(
        bufferExtractor, sizeExtractor, ortEnv, c, providers, false
    );
    std::shared_ptr<Model> embedderModel = Model::create(
        bufferEmbedder, sizeEmbedder, ortEnv, c, providers, false
    );

    // Assign pointers
    // blazefacePipeline = new BlazeFace(blazefaceModel);
    yoloPipeline = new YoloV7(yoloModel);
    SpoofingPipeline = new Spoofing(extractorModel, embedderModel);

    //DONE
}

extern "C"
JNIEXPORT jobject JNICALL
Java_ai_spoofing_TensorUtils_checkspoof(JNIEnv *env, jclass clazz, jbyteArray imageData, jint rotationDegree) {
    // Get the length of the byte array
    jsize length = env->GetArrayLength(imageData);
    // Allocate a buffer to hold the bytes
    jbyte *bufferImage = env->GetByteArrayElements(imageData, 0);
    // Decode data into Mat
    cv::Mat input_mat;
    input_mat = cv::imdecode(cv::Mat(1, length, CV_8UC1, bufferImage), cv::IMREAD_UNCHANGED);
    if (rotationDegree != 0)
        cv::rotate(input_mat, input_mat, EXIF_ORIENTATION_2_OPENCV_ROTATION[rotationDegree]);

    // From Bitmap to OpenCV Mat
    // Mat input_mat;
    // bitmapToMat(env, imageData, input_mat, false);
    // cvtColor(input_mat, input_mat, COLOR_RGBA2BGR);

    cv::Mat original = input_mat.clone();
    cv::Mat processed_img = yoloPipeline->preprocess(input_mat);
    std::vector<PredictResultHeadFace> res_yolo;
    yoloPipeline->detect(processed_img, &res_yolo);

    if (res_yolo.size() > 0) {

        // Create a list to store DetectionResult objects
        jclass detectionResultClass = env->FindClass("ai/spoofing/DetectionResult");
        jmethodID detectionResultConstructor = env->GetMethodID(detectionResultClass, "<init>",
                                                                "(Landroid/graphics/Bitmap;[F)V");
        jobjectArray results = env->NewObjectArray(res_yolo.size(), detectionResultClass, nullptr);

        for (int i = 0; i < res_yolo.size(); i++) {

            PredictResultHeadFace res = res_yolo[i];

            float padding_k = 0.8;

            float width = res.xmax - res.xmin;
            float height = res.ymax - res.ymin;
            float pad_x = width * padding_k - width;
            float pad_y = height * padding_k - height;

            cv::Rect bbox = cv::Rect(
                static_cast<int>(std::round(res.xmin - pad_x / 2)),
                static_cast<int>(std::round(res.ymin - pad_y / 2)),
                static_cast<int>(std::round(width * padding_k)),
                static_cast<int>(std::round(height * padding_k))
            );

            cv::Mat cropped_face;
            cropFace(original, cropped_face, bbox, 256, 256);
            cvtColor(cropped_face, cropped_face, COLOR_BGR2RGB);

            // Inference
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<float> probs = SpoofingPipeline->inference(cropped_face);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            // Log inference time to Android logcat
            __android_log_print(ANDROID_LOG_DEBUG, "SpoofingInference", "Inference time: %lld ms", duration.count());

            // create jobject
            // Create a new Bitmap with the correct size and format
            jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
            jmethodID createBitmapMethod = env->GetStaticMethodID(bitmapClass, "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
            jobject config = env->GetStaticObjectField(env->FindClass("android/graphics/Bitmap$Config"), env->GetStaticFieldID(env->FindClass("android/graphics/Bitmap$Config"), "ARGB_8888", "Landroid/graphics/Bitmap$Config;"));

            //convert to RGB before create bitmap
            cvtColor(cropped_face, cropped_face, COLOR_BGR2RGB);
            jobject faceBitmap = env->CallStaticObjectMethod(bitmapClass, createBitmapMethod, cropped_face.cols, cropped_face.rows, config);
            matToBitmap(env, cropped_face, faceBitmap, false); // Function to create Bitmap from Mat

            //create crops
            jfloatArray probsArray = env->NewFloatArray(probs.size());
            env->SetFloatArrayRegion(probsArray, 0, probs.size(), probs.data());

            jobject detectionResult = env->NewObject(
                    detectionResultClass,
                    detectionResultConstructor,
                    faceBitmap,
                    probsArray
            );
            env->SetObjectArrayElement(results, i, detectionResult);
        }
        return results;
    } else {
        // Return null if no faces are detected
        return nullptr;
    }
}


///////////////////////////
#ifdef __cplusplus
}
#endif
////////////////////////