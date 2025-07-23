#include "ws_server.h"
#include "config.h"
#include <mutex>

OfflineWebsocketServer::~OfflineWebsocketServer() {}

OfflineWebsocketServer::OfflineWebsocketServer(asio::io_context &io_context,
                                               asio::io_context &work_context)
    : io_conn_(io_context), io_work_(work_context) {
  // server_.init_asio(&io_conn_);
  server_.init_asio(&io_conn_);

  server_.set_access_channels(websocketpp::log::alevel::none);
  server_.set_error_channels(websocketpp::log::elevel::info);

  server_.set_open_handler([this](connection_hdl hdl) { OnOpen(hdl); });

  server_.set_close_handler([this](connection_hdl hdl) { OnClose(hdl); });

  server_.set_message_handler(
      [this](connection_hdl hdl, server::message_ptr msg) {
        OnMessage(hdl, msg);
      });

  _batch_sense_voice = std::make_unique<BatchSenseVoice>();
  _batch_sense_voice->init(CONFIG::asr_onnx, CONFIG::tokens);
  // _batch_sense_voice =
  //     std::make_unique<SenseVoice>(CONFIG::asr_onnx, CONFIG::tokens);
}

void OfflineWebsocketServer::OfflineWebsocketServer::OnOpen(
    connection_hdl hdl) {

  std::lock_guard<std::mutex> lock(mutex_);
  connections_.emplace(hdl, std::make_shared<ConnectionData>());

  PLOGI << "Number of active connections: "
        << static_cast<int32_t>(connections_.size());
}

void OfflineWebsocketServer::OnClose(connection_hdl hdl) {
  std::lock_guard<std::mutex> lock(mutex_);
  connections_.erase(hdl);

  PLOGI << "Number of active connections: "
        << static_cast<int32_t>(connections_.size());
}

void OfflineWebsocketServer::Close(connection_hdl hdl,
                                   websocketpp::close::status::value code,
                                   const std::string &reason) {
  auto con = server_.get_con_from_hdl(hdl);

  std::ostringstream os;
  os << "Closing " << con->get_remote_endpoint() << " with reason: " << reason
     << "\n";

  websocketpp::lib::error_code ec;
  server_.close(hdl, code, reason, ec);
  if (ec) {
    os << "Failed to close" << con->get_remote_endpoint() << ". "
       << ec.message() << "\n";
  }
  server_.get_alog().write(websocketpp::log::alevel::app, os.str());
}

void OfflineWebsocketServer::Send(connection_hdl hdl, const std::string &text) {
  websocketpp::lib::error_code ec;
  if (!Contains(hdl)) {
    return;
  }

  server_.send(hdl, text, websocketpp::frame::opcode::text, ec);
  if (ec) {
    server_.get_alog().write(websocketpp::log::alevel::app, ec.message());
  }
}

bool OfflineWebsocketServer::Contains(connection_hdl hdl) {
  std::lock_guard<std::mutex> lock(mutex_);
  return connections_.count(hdl);
}

void OfflineWebsocketServer::OnMessage(connection_hdl hdl,
                                       server::message_ptr msg) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto connection_data = connections_.find(hdl)->second;
  lock.unlock();
  const std::string &payload = msg->get_payload();

  switch (msg->get_opcode()) {
  case websocketpp::frame::opcode::text:
    if (payload == "Done") {
      // The client will not send any more data. We can close the
      // connection now.
      Close(hdl, websocketpp::close::status::normal, "Done");
    } else {
      Close(hdl, websocketpp::close::status::normal,
            std::string("Invalid payload: ") + payload);
    }
    break;

  case websocketpp::frame::opcode::binary: {
    auto p = reinterpret_cast<const int8_t *>(payload.data());

    if (connection_data->expected_byte_size == 0) {
      if (payload.size() < 8) {
        Close(hdl, websocketpp::close::status::normal, "Payload is too short");
        break;
      }

      connection_data->sample_rate = *reinterpret_cast<const int32_t *>(p);

      connection_data->expected_byte_size =
          *reinterpret_cast<const int32_t *>(p + 4);

      // int32_t max_byte_size_ = decoder_.GetConfig().max_utterance_length *
      //                          connection_data->sample_rate * sizeof(float);
      int32_t max_byte_size_ = CONFIG::max_utterance_length *
                               connection_data->sample_rate * sizeof(float);
      if (connection_data->expected_byte_size > max_byte_size_) {
        float num_samples = connection_data->expected_byte_size / sizeof(float);

        float duration = num_samples / connection_data->sample_rate;

        std::ostringstream os;
        os << "Max utterance length is configured to "
           << CONFIG::max_utterance_length << " seconds, received length is "
           << duration << " seconds. "
           << "Payload is too large!";
        Close(hdl, websocketpp::close::status::message_too_big, os.str());
        break;
      }

      connection_data->data.resize(connection_data->expected_byte_size);
      std::copy(payload.begin() + 8, payload.end(),
                connection_data->data.data());
      connection_data->cur = payload.size() - 8;
    } else {
      std::copy(payload.begin(), payload.end(),
                connection_data->data.data() + connection_data->cur);
      connection_data->cur += payload.size();
    }

    if (connection_data->expected_byte_size == connection_data->cur) {
      auto d = std::make_shared<ConnectionData>(std::move(*connection_data));
      // Clear it so that we can handle the next audio file from the client.
      // The client can send multiple audio files for recognition without
      // the need to create another connection.
      connection_data->sample_rate = 0;
      connection_data->expected_byte_size = 0;
      connection_data->cur = 0;

      // decoder_.Push(hdl, d);

      connection_data->Clear();

      // asio::post(io_work_, [this]() { decoder_.Decode(); });
      asio::post(io_work_, ([this, hdl, d]() {
                   std::vector<float> data(d->data.size());
                   auto samples = reinterpret_cast<const float *>(&d->data[0]);
                   auto num_samples = d->expected_byte_size / sizeof(float);
                   for (int i = 0; i < num_samples; ++i) {
                     data[i] = samples[i];
                   }
                   auto asr = this->_batch_sense_voice->recog(data);
                   this->Send(hdl, asr);
                 }));
    }
    break;
  }

  default:
    // Unexpected message, ignore it
    break;
  }
}

void OfflineWebsocketServer::Run(uint16_t port) {
  server_.set_reuse_addr(true);
  server_.listen(asio::ip::tcp::v4(), port);
  server_.start_accept();
}
