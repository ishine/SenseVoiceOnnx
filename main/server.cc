/*************************************************************************
    > File Name: main.cpp
    > Author: frank
    > Mail: 1216451203@qq.com
    > Created Time: 2025年05月07日 星期三 17时50分30秒
 ************************************************************************/
#include "config.h"
#include "ws_server.h"

int main(int argc, char *argv[]) {
  asio::io_context io_conn; // for network connections
  asio::io_context io_work; // for neural network and decoding
                            //
  OfflineWebsocketServer s(io_conn, io_work);
  s.Run(CONFIG::ws_port);

  // give some work to do for the io_work pool
  auto work_guard = asio::make_work_guard(io_work);

  std::vector<std::thread> io_threads;

  // decrement since the main thread is also used for network communications
  for (int32_t i = 0; i < CONFIG::num_io_threads - 1; ++i) {
    io_threads.emplace_back([&io_conn]() { io_conn.run(); });
  }

  std::vector<std::thread> work_threads;
  for (int32_t i = 0; i < CONFIG::num_work_threads; ++i) {
    work_threads.emplace_back([&io_work]() { io_work.run(); });
  }

  io_conn.run();

  for (auto &t : io_threads) {
    t.join();
  }

  for (auto &t : work_threads) {
    t.join();
  }
  return 0;
}
