cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)

project("cpp-spoof")

## thirdlib
# OpenCV ANDROID
add_library( lib_opencv SHARED IMPORTED )
set_target_properties(
    lib_opencv PROPERTIES 
    IMPORTED_LOCATION ${OpenCV_DIR}/libs/${ANDROID_ABI}/libopencv_java4.so
)   
set(LIB_OPENCV_INCLUDES ${OpenCV_DIR}/jni/include)

# Ortcxx
add_library( lib_ortcxx SHARED IMPORTED )
set_target_properties(
    lib_ortcxx PROPERTIES 
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/ortcxx/libs/${ANDROID_ABI}/libortcxx.so
)
set(LIB_ORTCXX_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/libs/ortcxx/include)

# Onnxruntime
add_library( lib_onnxruntime SHARED IMPORTED )
set_target_properties(
    lib_onnxruntime PROPERTIES 
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/onnxruntime/jni/${ANDROID_ABI}/libonnxruntime.so
)
set(LIB_ONNXRUNTIME_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/libs/onnxruntime/headers)

# Debug
message(STATUS ${LIB_OPENCV_INCLUDES})
message(STATUS ${LIB_ORTCXX_INCLUDES})
message(STATUS ${LIB_ONNXRUNTIME_INCLUDES})

message (STATUS ${lib_opencv})
message (STATUS ${lib_ortcxx})
message (STATUS ${lib_onnxruntime})

# Other Stuff
find_library(jnigraphics-lib jnigraphics)
find_library(log-lib log)

# Main lib
set(LIB_NAME spoofing)
set(LIB_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB LIB_SRC_FILES ${LIB_SRC_DIR}/*.cpp)

# Debug
message(${LIB_SRC_FILES})
message(${LIB_HEADERS})
message(STATUS ${jnigraphics-lib})
message(STATUS ${log-lib})

## BUILD
add_library(
    ${LIB_NAME} 
    SHARED 
    ${LIB_SRC_FILES} ${LIB_HEADERS} #1 
    native.cpp ${LIB_SRC_DIR}/PillowResize.cc
)
target_link_libraries(
    ${LIB_NAME} 
    lib_ortcxx lib_onnxruntime lib_opencv #1
    ${jnigraphics-lib} #2
    ${log-lib} #3
    android #4
)
target_include_directories(${LIB_NAME} PUBLIC 
    ${LIB_OPENCV_INCLUDES} 
    ${LIB_ORTCXX_INCLUDES} 
    ${LIB_ONNXRUNTIME_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/native.cpp
)
target_include_directories(${LIB_NAME} PUBLIC ${LIB_HEADERS})