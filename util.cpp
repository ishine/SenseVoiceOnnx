#include "util.h"
#include <fstream>
#include <iomanip>

std::vector<std::string> splitString(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);

  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }

  return tokens;
}

bool load_wav_file(const char *filename, int32_t *sampling_rate,
                   std::vector<float> &data) {
  struct WaveHeader header{};

  std::ifstream is(filename, std::ifstream::binary);
  is.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!is) {
    std::cout << "Failed to read " << filename;
    return false;
  }

  if (!header.Validate()) {
    return false;
  }

  header.SeekToDataChunk(is);
  if (!is) {
    return false;
  }

  *sampling_rate = header.sample_rate;
  // header.subchunk2_size contains the number of bytes in the data.
  // As we assume each sample contains two bytes, so it is divided by 2 here
  auto speech_len = header.subchunk2_size / 2;
  data.resize(speech_len);

  auto speech_buff = (int16_t *)malloc(sizeof(int16_t) * speech_len);

  if (speech_buff) {
    memset(speech_buff, 0, sizeof(int16_t) * speech_len);
    is.read(reinterpret_cast<char *>(speech_buff), header.subchunk2_size);
    if (!is) {
      std::cout << "Failed to read " << filename;
      return false;
    }

    float scale = 32768;
    // float scale = 1.0;
    for (int32_t i = 0; i != speech_len; ++i) {
      data[i] = (float)speech_buff[i] / scale;
    }
    free(speech_buff);
    return true;
  } else {
    free(speech_buff);
    return false;
  }
}

std::string getCurrentTime() {
  // 获取当前时间点
  auto now = std::chrono::system_clock::now();
  // 转换为time_t，以便使用put_time
  auto now_c = std::chrono::system_clock::to_time_t(now);
  // 转换为time_point，以便获取毫秒部分
  auto duration = now.time_since_epoch();
  auto millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  // 使用stringstream来构建完整的日期时间字符串
  std::stringstream ss;
  ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
  ss << "." << std::setw(3) << std::setfill('0') << (millis % 1000); // 毫秒部分

  return ss.str();
}
