/*************************************************************************
    > File Name: config.h
    > Author: frank
    > Mail: 1216451203@qq.com
    > Created Time: 2025年07月21日 星期一 15时59分30秒
 ************************************************************************/
#pragma once
#include <string>

namespace CONFIG {
// onnx
const std::string asr_onnx =
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx";
const std::string tokens =
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt";
const std::string vad_onnx = "silero_vad.onnx";
const int max_batch = 5;
const int max_utterance_length = 10; // 10s

// ws
const u_int16_t ws_port = 6001;
const int32_t num_io_threads = 4;
const int32_t num_work_threads = 4;
} // namespace CONFIG
