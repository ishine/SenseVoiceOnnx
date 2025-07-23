/*************************************************************************
    > File Name: ws_server.h
    > Author: frank
    > Mail: 1216451203@qq.com
    > Created Time: 2025年07月21日 星期一 15时57分12秒
 ************************************************************************/
#pragma once
#include "batch_sense_voice.h"
#include "clog.h"
#include "sense_voice.h"
#include "util.h"
#include <memory>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

typedef websocketpp::server<websocketpp::config::asio> server;
using websocketpp::connection_hdl;
using websocketpp::lib::bind;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;

struct ConnectionData {
  // Sample rate of the audio samples the client
  int32_t sample_rate;

  // Number of expected bytes sent from the client
  int32_t expected_byte_size = 0;

  // Number of bytes received so far
  int32_t cur = 0;

  // It saves the received samples from the client.
  // We will **reinterpret_cast** it to float.
  // We expect that data.size() == expected_byte_size
  std::vector<int8_t> data;

  void Clear() {
    sample_rate = 0;
    expected_byte_size = 0;
    cur = 0;
    data.clear();
  }
};
using ConnectionDataPtr = std::shared_ptr<ConnectionData>;

class OfflineWebsocketServer {
public:
  OfflineWebsocketServer(asio::io_context &io_conn, asio::io_context &io_work);
  ~OfflineWebsocketServer();

  server &GetServer() { return server_; }

  void Run(uint16_t port);

private:
  // When a websocket client is connected, it will invoke this method
  // (Not for HTTP)
  void OnOpen(connection_hdl hdl);

  // When a websocket client is disconnected, it will invoke this method
  void OnClose(connection_hdl hdl);

  void OnMessage(connection_hdl hdl, server::message_ptr msg);

  // Close a websocket connection with given code and reason
  void Close(connection_hdl hdl, websocketpp::close::status::value code,
             const std::string &reason);

  void Send(connection_hdl hdl, const std::string &text);

  bool Contains(connection_hdl hdl);

private:
  server server_;

  std::map<connection_hdl, ConnectionDataPtr, std::owner_less<connection_hdl>>
      connections_;
  std::mutex mutex_;

  std::unique_ptr<BatchSenseVoice> _batch_sense_voice;
  // std::unique_ptr<SenseVoice> _batch_sense_voice;
  asio::io_context &io_conn_;
  asio::io_context &io_work_;
};
