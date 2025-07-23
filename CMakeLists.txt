cmake_minimum_required(VERSION 3.12)
project(onnx_inference_project)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Whether to build shared libraries" OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 查找ONNX Runtime
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(onnxruntime)
include(kaldi-native-fbank)
include(libsamplerate)
include(nlohmann_json) 
include(asio)
include(websocketpp)
set(ONNXRUNTIME_DIR ${onnxruntime_SOURCE_DIR})
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
message(STATUS "ONNXRUNTIME_INCLUDE_DIRS: ${ONNXRUNTIME_INCLUDE_DIRS}")
include_directories(${CMAKE_SOURCE_DIR})

find_package(ALSA REQUIRED)

add_definitions(-DASIO_STANDALONE)

# 可执行文件
add_executable(infer
    main/infer.cc
    util.cpp
    clog.cpp
    asr.cpp
    sense_voice.cpp
    vad.cpp
)

# 链接库
target_link_libraries(infer
   ${onnxruntime_lib_files} 
   kaldi-native-fbank-core
   samplerate
)

# 可执行文件
add_executable(stream
    main/alsa_stream.cc
    util.cpp
    clog.cpp
    asr.cpp
    sense_voice.cpp
    vad.cpp
    resample.cc
    alsa.cc
)

# 链接库
target_link_libraries(stream
   ${onnxruntime_lib_files} 
   kaldi-native-fbank-core
   samplerate
   ALSA::ALSA
)

# 可执行文件
add_executable(server
    main/server.cc
    ws_server.cpp
    util.cpp
    clog.cpp
    asr.cpp
    batch_sense_voice.cpp
    sense_voice.cpp
    vad.cpp
    onnx_engine.cpp
    pad-sequence.cc
)

# 链接库
target_link_libraries(server
   ${onnxruntime_lib_files} 
   kaldi-native-fbank-core
   samplerate
)
