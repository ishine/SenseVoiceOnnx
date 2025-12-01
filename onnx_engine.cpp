/*************************************************************************
    > File Name: OnnxWrapper.cpp
    > Author: frank
    > Mail: 1216451203@qq.com
    > Created Time: 2025年07月08日 星期二 15时34分18秒
 ************************************************************************/
#include "onnx_engine.h"
#include "util.h"
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <string>

Ort::Value make_tensor(std::vector<std::shared_ptr<Request>> &reqs,
                       int feat_id) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  const int32_t batch_size = static_cast<int32_t>(reqs.size());
  bool isInt = reqs[0]->_input_arrays[feat_id].isInt;

  std::vector<Ort::Value> values;
  Ort::AllocatorWithDefaultOptions allocator;

  if (isInt) {
    for (auto &req : reqs) {
      auto x_ort = Ort::Value::CreateTensor(
          memory_info, req->_input_arrays[feat_id].data_int32.data(),
          req->_input_arrays[feat_id].data_int32.size(),
          req->_input_arrays[feat_id].shape.data(),
          req->_input_arrays[feat_id].shape.size());
      values.push_back(std::move(x_ort));
    }
    return PadSequence<int32_t>(allocator, values, 0);
  } else {
    for (auto &req : reqs) {
      auto x_ort = Ort::Value::CreateTensor(
          memory_info, req->_input_arrays[feat_id].data_float.data(),
          req->_input_arrays[feat_id].data_float.size(),
          req->_input_arrays[feat_id].shape.data(),
          req->_input_arrays[feat_id].shape.size());
      values.push_back(std::move(x_ort));
    }
    return PadSequence<float>(allocator, values, 0);
  }
}

void OnnxSession::forward(std::vector<std::shared_ptr<Request>> &reqs) {
  PLOGD << "Audio Batch Size: " << reqs.size();
  std::vector<Ort::Value> input_orts;
  for (int i = 0; i < _input_names.size(); ++i) {
    input_orts.push_back(make_tensor(reqs, i));
  }

  auto output_tensors = std::make_shared<std::vector<Ort::Value>>(_session->Run(
      {}, _input_names.data(), input_orts.data(), input_orts.size(),
      _output_names.data(), _output_names.size()));

  for (int i = 0; i < reqs.size(); ++i) {
    PLOGI << "assign:" << i << " " << output_tensors.get();
    reqs[i]->_output_arrays = output_tensors;
    reqs[i]->_output_index = i;
  }
  _notice_cv.notify_all();
}

OnnxSession::OnnxSession(const std::string &model_path) {

  _env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
  _session_options.SetIntraOpNumThreads(8);
  _session_options.SetInterOpNumThreads(1);
  _session = std::make_unique<Ort::Session>(_env, model_path.c_str(),
                                            _session_options);
}

void OnnxSession::addReq(std::shared_ptr<Request> req) {
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _reqs.push_back(req);
    _cv.notify_all();
  }

  {
    std::unique_lock<std::mutex> lock(_notice_mutex);
    _notice_cv.wait_for(lock, std::chrono::milliseconds(60000),
                        [&] { return req->_output_arrays != nullptr; });
  }
}

void OnnxSession::init() {
  setupIO();
  getCustomMetadataMap(_meta_data);
  for (int i = 0; i < 8; ++i) {
    _loops.emplace_back([this] {
      while (_running.load()) {
        std::vector<std::shared_ptr<Request>> reqs;
        {
          std::unique_lock<std::mutex> lock(_mutex);
          // _cv.wait_for(lock, std::chrono::milliseconds(20),
          //              [&] { return _reqs.size() > 2; });
          _cv.wait(lock, [&] { return _reqs.size() > 0; });
          reqs.swap(_reqs);
        }
        if (reqs.size() > 0) {
          forward(reqs);
        }
      }
    });
  }
}

void OnnxSession::setupIO() {
  Ort::AllocatorWithDefaultOptions allocator;

  // 获取输入信息
  size_t num_input_nodes = _session->GetInputCount();
  _input_names.reserve(num_input_nodes);

  for (size_t i = 0; i < num_input_nodes; i++) {
    auto input_name = _session->GetInputNameAllocated(i, allocator);

    char *dest = new char[strlen(input_name.get()) + 1]; // +1 用于空终止符
    _input_names.push_back(dest);
    strcpy(dest, input_name.get());

    Ort::TypeInfo type_info = _session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> input_dims = tensor_info.GetShape();
    PLOGI << "Input " << i << " name: " << dest;
    PLOGI << "Input shape: ";
    for (auto dim : input_dims) {
      PLOGI << dim << " ";
    }
    _input_dims.push_back(input_dims);
  }

  // 获取输出信息
  size_t num_output_nodes = _session->GetOutputCount();
  _output_names.reserve(num_output_nodes);

  for (size_t i = 0; i < num_output_nodes; i++) {
    auto output_name = _session->GetOutputNameAllocated(i, allocator);
    char *dest = new char[strlen(output_name.get()) + 1];
    strcpy(dest, output_name.get());
    _output_names.push_back(dest);

    Ort::TypeInfo type_info = _session->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> output_dims = tensor_info.GetShape();
    PLOGI << "Output " << i << " name: " << dest;
    PLOGI << "Output shape: ";
    for (auto dim : output_dims) {
      PLOGI << dim << " ";
    }
  }
}

void OnnxSession::getCustomMetadataMap(
    std::map<std::string, std::string> &data) {
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::ModelMetadata model_metadata = _session->GetModelMetadata();

  // 获取自定义元数据数量
  auto keys = model_metadata.GetCustomMetadataMapKeysAllocated(allocator);
  std::cout << "\nCustom Metadata (" << keys.size() << " items):" << std::endl;

  // 遍历所有自定义元数据
  for (size_t i = 0; i < keys.size(); ++i) {
    const char *key = keys[i].get();
    auto value =
        model_metadata.LookupCustomMetadataMapAllocated(key, allocator);
    std::cout << key << ":" << value.get() << std::endl;
    data[std::string(key)] = std::string(value.get());
  }
}

void OnnxEngine::addModel(const std::string &name,
                          std::unique_ptr<OnnxSession> &&session) {

  _sessions[name] = std::move(session);
}
