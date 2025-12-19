/*!
 *  Copyright (c) 2023 by Contributors
 * \file .engine.h
 */

#ifndef MLC_CLI_CHAT_ENGINE_H
#define MLC_CLI_CHAT_ENGINE_H

#include <json_ffi/json_ffi_engine.h>
#include <picojson.h>
#include <tvm/runtime/module.h>

#include <condition_variable>
#include <fstream>
#include <functional>
#include <memory>
#include <queue>
#include <thread>
#include <vector>

#include "base.h"

using tvm::ffi::Function;
using tvm::ffi::TypedFunction;
using namespace tvm::runtime;

class EngineStateCli {
 public:
  std::queue<std::string> sync_queue;
  bool last_chunk_arrived = false;
  std::string output;
  std::string finish_reason;
  std::shared_ptr<std::mutex> queue_mutex;
  std::shared_ptr<std::condition_variable> queue_cv;
  double decode_tokens_per_s;
  double prefill_tokens_per_s;
  double prompt_tokens;
  double completion_tokens;

  EngineStateCli();
  std::function<void(const std::string&)> get_request_stream_callback();
  std::string handle_chat_completion(ffi::Module mod, const std::string& request_json,
                                     bool include_usage, const std::string& request_id);
  void getStats();
};

class Completions {
 public:
  std::shared_ptr<EngineStateCli> engine_state;
  ffi::Module __mod;

  explicit Completions(std::shared_ptr<EngineStateCli> engine_state, ffi::Module mod);

  inline std::string GenerateUUID(size_t length);

  std::string create(std::vector<Message>& messages, int max_tokens = -1);
};

class Chat {
 public:
  Completions completions;

  explicit Chat(std::shared_ptr<EngineStateCli> engine_state, ffi::Module mod);
};

class BackgroundLoops {
 private:
  // Default threads
  std::thread background_loop_thread;
  std::thread background_stream_back_loop_thread;
  bool terminated = false;
  std::optional<ffi::Module> __mod;

 public:
  // Parametrized constructor
  explicit BackgroundLoops(ffi::Module mod);
  ~BackgroundLoops();

  void terminate();
};

class JSONFFIEngineWrapper {
 public:
  std::shared_ptr<Chat> chat;
  std::shared_ptr<EngineConfig> engine_config;
  std::optional<ffi::Module> mod;
  std::shared_ptr<EngineStateCli> engine_state;
  std::shared_ptr<BackgroundLoops> background_loops;

  explicit JSONFFIEngineWrapper(std::string model_path, std::string model_lib_path,
                                std::string mode, std::string device, int device_id);
  void Reset();
};

#endif
