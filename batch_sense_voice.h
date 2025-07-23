#pragma once
#include "onnx_engine.h"
#include <map>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

class BatchSenseVoice {
public:
  BatchSenseVoice() {};
  void init(const std::string &model_path, const std::string &token_path);
  ~BatchSenseVoice();

  std::string recog(const std::vector<float> &wav);

  int32_t window_size_;
  int32_t window_shift_;
  int32_t with_itn_;
  int32_t without_itn_;

  std::map<std::string, int32_t> lang_id_;
  std::vector<float> neg_mean_;
  std::vector<float> inv_stddev_;
  std::map<std::string, std::string> tokens_;
};
