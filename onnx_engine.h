/*************************************************************************
    > File Name: OnnxWrapper.h
    > Author: frank
    > Mail: 1216451203@qq.com
    > Created Time: 2025年07月08日 星期二 15时34分00秒
 ************************************************************************/
#pragma once
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <onnxruntime_cxx_api.h>
#include <queue>
#include <string>
#include <thread>

struct ArrayWithShape {
  std::vector<float> data_float;
  std::vector<int32_t> data_int32;
  std::vector<int64_t> shape;
  // std::string name;
  bool isInt = false;
};

struct Request {
  Request() {}

  Request(Request &&req) { _input_arrays.swap(req._input_arrays); }

  // input
  std::vector<ArrayWithShape> _input_arrays;

  // output
  std::shared_ptr<std::vector<Ort::Value>> _output_arrays = nullptr;
  int _output_index = 0;
};

template <typename T>
Ort::Value PadSequence(OrtAllocator *allocator, std::vector<Ort::Value> &values,
                       float padding_value) {
  int32_t batch_size = static_cast<int32_t>(values.size());

  std::vector<int64_t> shape0 =
      values[0].GetTensorTypeAndShapeInfo().GetShape();
  assert(shape0.size() == 2 or (shape0.size() == 1 and shape0[0] == 1));
  if (shape0.size() == 1) {
    std::array<int64_t, 1> ans_shape{batch_size};
    Ort::Value ans = Ort::Value::CreateTensor<T>(allocator, ans_shape.data(),
                                                 ans_shape.size());
    T *dst = ans.GetTensorMutableData<T>();
    std::fill(dst, dst + batch_size, padding_value);
    for (const auto &v : values) {
      const T *src = v.GetTensorData<T>();
      auto shape = v.GetTensorTypeAndShapeInfo().GetShape();
      std::copy(src, src + 1, dst);
      dst += 1;
    }
    return ans;
  }

  auto max_T = shape0[0];
  auto feature_dim = shape0[1];
  for (int32_t i = 1; i != batch_size; ++i) {
    auto shape = values[i].GetTensorTypeAndShapeInfo().GetShape();
    assert(shape.size() == 2);
    assert(shape[1] == feature_dim);
    max_T = std::max(max_T, shape[0]);
  }

  std::array<int64_t, 3> ans_shape{batch_size, max_T, feature_dim};

  Ort::Value ans = Ort::Value::CreateTensor<T>(allocator, ans_shape.data(),
                                               ans_shape.size());
  T *dst = ans.GetTensorMutableData<T>();
  std::fill(dst, dst + batch_size * max_T * feature_dim, padding_value);

  for (auto &v : values) {
    const T *src = v.GetTensorData<T>();
    auto shape = v.GetTensorTypeAndShapeInfo().GetShape();
    std::copy(src, src + shape[0] * shape[1], dst);
    dst += max_T * feature_dim;
  }

  return ans;

  // TODO(fangjun): Check that the returned value is correct.
}

class OnnxSession {
public:
  OnnxSession(const std::string &model_path); // json format
  virtual ~OnnxSession() {
    _running = false;
    //_loop.join();
    for (auto &th : _loops) {
      th.join();
    }
  };

  virtual void init();

  virtual void forward(std::vector<std::shared_ptr<Request>> &reqs);

  std::atomic<bool> _running = true;
  Ort::Env _env;
  Ort::SessionOptions _session_options;
  std::unique_ptr<Ort::Session> _session;
  void addReq(std::shared_ptr<Request> req);
  std::vector<std::shared_ptr<Request>> _reqs;
  // std::thread _loop;
  std::vector<std::thread> _loops;
  std::mutex _mutex;
  std::condition_variable _cv;
  std::string _name;

  std::vector<const char *> _input_names;
  std::vector<std::vector<int64_t>> _input_dims;
  std::vector<const char *> _output_names;
  std::map<std::string, std::string> _meta_data;

  void setupIO();
  void getCustomMetadataMap(std::map<std::string, std::string> &data);

  std::mutex _notice_mutex;
  std::condition_variable _notice_cv;
};

class OnnxEngine {
public:
  void addModel(const std::string &name, std::unique_ptr<OnnxSession> &&);
  void request(const std::string &name, std::shared_ptr<Request> req) {
    _sessions[name]->addReq(req);
  }

  static OnnxEngine *get_inst() {
    static OnnxEngine inst;
    return &inst;
  }

  std::map<std::string, std::unique_ptr<OnnxSession>> _sessions;

private:
  OnnxEngine() {}
};
