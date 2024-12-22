cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)

project("cpp-spoof")

## thirdlib
# OpenCV ANDROID
find_package(Eigen3 REQUIRED)
find_package(Iconv REQUIRED)
find_package(OpenCV REQUIRED)

# Ortcxx
add_library( lib_ortcxx SHARED IMPORTED )
set_target_properties(
    lib_ortcxx PROPERTIES 
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/ortcxx/libs/libortcxx.so
)
set(LIB_ORTCXX_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/libs/ortcxx/headers)

# Onnxruntime
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/onnxruntime_1.19.0/lib/)
set(onnxruntime_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/libs/onnxruntime_1.19.0/include)

# Debug
message(STATUS ${OpenCV_INCLUDE_DIRS})
message(STATUS ${LIB_ORTCXX_INCLUDES})
message(STATUS ${onnxruntime_INCLUDE_DIRS})

message (STATUS ${OpenCV_LIBS})
message (STATUS lib_ortcxx)
message (STATUS ${onnxruntime_LIBRARIES})


# Main lib
set(LIB_NAME spoofing)
set(LIB_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB LIB_SRC_FILES ${LIB_SRC_DIR}/*.cpp)

# Debug
message(${LIB_SRC_FILES})
message(${LIB_HEADERS})

## BUILD
add_library(
    ${LIB_NAME} 
    SHARED 
    ${LIB_SRC_FILES} ${LIB_HEADERS} #1 
)

target_link_libraries(
    ${LIB_NAME} 
    ${OpenCV_LIBS} #2
    # ${onnxruntime_LIBRARIES} #3
    onnxruntime
    lib_ortcxx #4
    # lib_onnxruntime #4
)

target_include_directories(${LIB_NAME} PUBLIC 
    ${OpenCV_INCLUDE_DIRS} 
    ${LIB_ORTCXX_INCLUDES} 
    ${onnxruntime_INCLUDE_DIRS}
)

target_include_directories(${LIB_NAME} PUBLIC ${LIB_HEADERS})

## EXECUTABLE
add_executable(
    test1 tests/ortapi.cpp
)
target_link_libraries(
    test1 ${LIB_NAME}
)
target_include_directories(
    test1 PUBLIC
    ${LIB_HEADERS} ${OpenCV_INCLUDE_DIRS} 
    ${LIB_ORTCXX_INCLUDES} ${onnxruntime_INCLUDE_DIRS}
)
