/*************************************************************************
    > File Name: main.cpp
    > Author: frank
    > Mail: 1216451203@qq.com
    > Created Time: 2025年05月07日 星期三 17时50分30秒
 ************************************************************************/
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "asr.h"
#include "clog.h"
#include "util.h"

void onAsr(const std::string &asr) {
  std::cout << "asr:" << asr << "\n------------" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    PLOGE << "usage: " << argv[0] << " xx.wav";
    return -1;
  }
  try {
    // 初始化推理引擎
    std::string asr_onnx =
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx";
    std::string tokens =
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt";
    std::string vad_onnx = "silero_vad.onnx";

    std::string wav = argv[1];

    auto asr = std::make_unique<Asr>(asr_onnx, tokens, vad_onnx);
    asr->_onAsr = onAsr;

    std::vector<float> data;
    int32_t sampling_rate = 16000;
    load_wav_file(wav.c_str(), &sampling_rate, data);

    std::vector<float> tmp;
    for (auto i : data) {
      tmp.push_back(i * 37268);
    }
    std::cout << "AsrResult:" << asr->_sence_voice->recog(tmp) << std::endl;

    for (int i = 0; i < data.size() / 512; ++i) {
      std::vector<float> tmp(data.begin() + i * 512,
                             data.begin() + i * 512 + 512);
      asr->push_data(tmp, sampling_rate);
    }
    asr->wait_finish();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
