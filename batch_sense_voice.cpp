/*************************************************************************
    > File Name: infer.cpp
    > Author: frank
    > Mail: 1216451203
    > Created Time: 2025年05月07日 星期三 17时32分49秒
 ************************************************************************/
#include "batch_sense_voice.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

// for time cost
#include "util.h"

void BatchSenseVoice::init(const std::string &model_path,
                           const std::string &tokens_path) {
  auto *inst = OnnxEngine::get_inst();
  auto ptr = std::make_unique<OnnxSession>(model_path);
  ptr->init();

  std::map<std::string, std::string> meta;
  ptr->getCustomMetadataMap(meta);
  inst->addModel("SenseVoice", std::move(ptr));

  auto get_int32 = [&meta](const std::string &key) { return stoi(meta[key]); };

  window_size_ = get_int32("lfr_window_size");
  window_shift_ = get_int32("lfr_window_shift");

  std::vector<std::string> keys{"lang_zh", "lang_en", "lang_ja", "lang_ko",
                                "lang_auto"};

  for (auto &key : keys) {
    lang_id_[key] = get_int32(key);
  }

  with_itn_ = get_int32("with_itn");
  without_itn_ = get_int32("without_itn");

  auto tmp = splitString(meta["neg_mean"], ',');
  for (auto f : tmp) {
    neg_mean_.push_back(stof(f));
  }
  tmp = splitString(meta["inv_stddev"], ',');
  for (auto f : tmp) {
    inv_stddev_.push_back(stof(f));
  }

  std::ifstream fin(tokens_path);
  std::string line;
  while (std::getline(fin, line)) {
    auto arr = splitString(line, ' ');
    if (arr.size() == 2) {
      tokens_[arr[1]] = arr[0];
    }
  }
}

BatchSenseVoice::~BatchSenseVoice() {}

std::string BatchSenseVoice::recog(const std::vector<float> &data) {
  Timer cost("AsrCost");
  // extract fbank
  knf::FbankOptions opts;
  opts.frame_opts.dither = 0;
  opts.frame_opts.snip_edges = false;
  opts.frame_opts.window_type = "hamming";
  opts.frame_opts.samp_freq = 16000;
  opts.mel_opts.num_bins = 80;

  knf::OnlineFbank fbank(opts);
  fbank.AcceptWaveform(16000, data.data(), data.size());
  fbank.InputFinished();

  int32_t n = fbank.NumFramesReady();
  std::vector<float> feats;

  for (int i = 0; i + window_size_ <= n; i += window_shift_) {
    for (int k = i * 80; k < (i + window_size_) * 80; k++) {
      double value = fbank.GetFrame(k / 80)[k % 80];
      feats.push_back((value + neg_mean_[k % 560]) * inv_stddev_[k % 560]);
    }
  }
  auto req = std::make_shared<Request>();
  // feature bank
  ArrayWithShape bank;
  bank.isInt = false;
  bank.data_float = feats;
  int32_t feat_dim = 80 * window_size_;
  int64_t num_frames = feats.size() / feat_dim;
  bank.shape = {num_frames, feat_dim};
  req->_input_arrays.push_back(bank);
  //
  ArrayWithShape length;
  length.isInt = true;
  length.data_int32.push_back(num_frames);
  length.shape.push_back(1);
  req->_input_arrays.push_back(length);
  //
  ArrayWithShape lang;
  lang.isInt = true;
  lang.data_int32.push_back(0);
  lang.shape.push_back(1);
  req->_input_arrays.push_back(lang);
  //
  ArrayWithShape text_norm;
  text_norm.isInt = true;
  text_norm.data_int32.push_back(with_itn_);
  text_norm.shape.push_back(1);
  req->_input_arrays.push_back(text_norm);

  // req._feats.swap(feats);
  OnnxEngine::get_inst()->request("SenseVoice", req);

  // convert to text
  auto asr = std::string("");
  if (req->_output_arrays) {
    auto &val = req->_output_arrays->at(0);
    int output_index = req->_output_index;

    auto info = val.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();
    std::string shape_info;
    shape_info += "shape: ";
    for (auto s : shape) {
      shape_info += std::to_string(s);
      shape_info += " ";
    }
    PLOGI << shape_info;
    size_t element_count = info.GetElementCount() / shape[0];

#if 0
    size_t dim_count = info.GetDimensionsCount(); // 获取维度数量
    std::cout << "dim_count:" << dim_count << std::endl;
#endif

    float *logits_data =
        val.GetTensorMutableData<float>() + output_index * element_count;
    int64_t last_dim = shape.empty() ? 1 : shape.back();
    size_t num_rows = element_count / last_dim;

    // 5. 为结果分配空间
    std::vector<int64_t> result(num_rows);

    // 6. 对每行计算 argmax
    for (size_t i = 0; i < num_rows; ++i) {
      float *row_start = logits_data + i * last_dim;
      result[i] = std::distance(
          row_start, std::max_element(row_start, row_start + last_dim));
    }

    std::vector<int64_t> final = unique_consecutive<int64_t>(result);
    for (const auto f : final) {
      if (f > 0 and f < 24884) {
        asr += tokens_[std::to_string(f)];
      }
    }
    if (asr.size() > 0) {
      PLOGI << asr;
    } else {
      PLOGE << "empty asr";
    }
  } else {
    PLOGE << "empty output array";
  }
  return asr;
}
