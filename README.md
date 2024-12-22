# facialcxx

## Overview

`facialcxx` is an Android application that utilizes ONNX Runtime and OpenCV for facial recognition and spoof detection. The project includes native C++ code for processing images, leveraging machine learning models for inference.

## Features

- Facial recognition using YOLOv7 model.
- Spoof detection to prevent fraudulent activities.
- Integration with ONNX Runtime for efficient model inference.
- Utilizes OpenCV for image processing.
- Currently support only for images.

## Requirements

- Android Studio
- Android NDK
- CMake
- OpenCV
- ONNX Runtime

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/facialcxx.git
    cd ./platforms/android/app
    ```

2. **Prepare the models:**
    - Download the required model files and testing files
    - Place them in the `src/main/assets/weights` and `src/main/assets/tests` directory.


3. **Open the project in Android Studio:**
    Open Android Studio and select `Open an existing project`, then navigate to `./platforms/android/app`.

4. **Build the project:**
    Select `Build -> Make Project` in the top toolbar and ensure the project builds successfully.

5. **Run the application:**
    Connect your Android device and select `Run -> Run app` to install and run the application on your device.

## Usage
Currently on support Android (14) API 34, you shoud create a emulator with this config.