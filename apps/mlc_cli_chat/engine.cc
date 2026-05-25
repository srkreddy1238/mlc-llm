/*!
 *  Copyright (c) 2023 by Contributors
 * \file engine.cc
 */

#include "engine.h"

#include <iostream>

#include "base.h"

/// Helper function to get the json format of messages
std::string messagesToString(const std::vector<Message>& messages) {
  std::string result{""};
  for (size_t i = 0; i < messages.size(); ++i) {
    const auto& msg = messages[i];
    result += "{";

    bool firstItem = true;
    for (const auto& [key, value] : msg.content) {
      if (!firstItem) {
        result += ",";
      }
      result += "\"" + key + "\":";

      if (key == "role") {
        if (value == "user") {
          result += "\"" + value + "\"";
        } else if (value == "assistant") {
          result += "\"" + value + "\"";
        }
      } else if (key == "content") {
        if (i % 2 == 0) {
          result += "\"" + value + "\"";
        } else {
          result += value;
        }
      }
      firstItem = false;
    }
    result += "}";

    if (i != messages.size() - 1) {
      result += ",";
    }
  }
  return result;
}

// Helper function to print History
void printHistory(std::vector<Message> history) {
  for (int i = 0; i < history.size(); i++) {
    auto msg = history[i];
    std::cout << " content " << msg.content["content"];
  }
  std::cout << "\n";
}

EngineStateCli::EngineStateCli()
    : queue_cv(std::make_shared<std::condition_variable>()),
      queue_mutex(std::make_shared<std::mutex>()) {}

std::function<void(const std::string&)> EngineStateCli::get_request_stream_callback() {
  return [this](const std::string& response) -> void {
    {
      this->sync_queue.push(response);
      queue_cv->notify_one();
    }
  };
}

std::string EngineStateCli::handle_chat_completion(ffi::Module mod, const std::string& request_json,
                                                   bool include_usage,
                                                   const std::string& request_id, bool silent) {
  // Clear the queue making sure that queue is empty
  // Not really required since this process should ideally make the queue empty
  {
    std::lock_guard<std::mutex> lock(*queue_mutex);
    std::queue<std::string> empty;
    std::swap(sync_queue, empty);
  }
  // TVM Global Function which generates the responses
  bool success = mod->GetFunction("chat_completion").value()(request_json, request_id).cast<bool>();
  if (!success) {
    std::cerr << "Failed to start chat completion" << std::endl;
  }

  try {
    last_chunk_arrived = false;

    // Clear the ouput after every chat completion
    output = "";

    while (!last_chunk_arrived) {
      std::string json_str;
      std::unique_lock<std::mutex> lock(*queue_mutex);

      // Wait until the queue is not empty
      queue_cv->wait(lock, [this] { return !sync_queue.empty(); });
      std::string response = sync_queue.front();
      sync_queue.pop();
      tvm::ffi::String err;
      // Parse the JSON
      auto v = tvm::ffi::json::Parse(response, &err);
      // Check for errors
      if (!err.empty()) {
        std::cerr << "JSON parsing error: " << err << std::endl;
      }

      // parsing successful, navigate through the array
      auto arr = v.cast<tvm::ffi::json::Array>();
      for (auto& item : arr) {
        auto obj = item.cast<tvm::ffi::json::Object>();

        // Extract 'delta' content if available
        if (obj.find("choices") != obj.end() &&
            !obj["choices"].cast<tvm::ffi::json::Array>().empty()) {
          auto choices =
              obj["choices"].cast<tvm::ffi::json::Array>()[0].cast<tvm::ffi::json::Object>();
          if (choices.find("delta") != choices.end()) {
            auto delta = choices["delta"].cast<tvm::ffi::json::Object>();
            if (delta.find("content") != delta.end()) {
              std::string content = delta["content"].cast<std::string>();

              if (!silent) std::cout << content << std::flush;
              output += content;
            }
          }
        }
        // Extract 'usage' details if available
        if (obj.find("usage") != obj.end()) {
          last_chunk_arrived = true;
          if (!silent) std::cout << std::endl;
          auto usage = obj["usage"].cast<tvm::ffi::json::Object>();

          // Access the 'usage' details
          double prompt_tokens = usage["prompt_tokens"].cast<double>();
          double completion_tokens = usage["completion_tokens"].cast<double>();
          double total_tokens = usage["total_tokens"].cast<double>();

          // Access the 'extra' details
          auto extra = usage["extra"].cast<tvm::ffi::json::Object>();
          double prefill_tokens_per_s = extra["prefill_tokens_per_s"].cast<double>();
          double decode_tokens_per_s = extra["decode_tokens_per_s"].cast<double>();
          double end_to_end_latency_s = extra["end_to_end_latency_s"].cast<double>();

          // fill the stats details
          this->decode_tokens_per_s = decode_tokens_per_s;
          this->prefill_tokens_per_s = prefill_tokens_per_s;
          this->prompt_tokens = prompt_tokens;
          this->completion_tokens = completion_tokens;
        }
      }
    }
  } catch (const std::exception& exception) {
    mod->GetFunction("abort").value()(request_id);
    throw;
  }
  return output;
}

void EngineStateCli::getStats() {
  std::cout << " decode : " << this->decode_tokens_per_s << " tok/sec (" << this->completion_tokens
            << " tokens in " << this->completion_tokens / this->decode_tokens_per_s << " sec)"
            << ", prefill : " << this->prefill_tokens_per_s << " tok/sec (" << this->prompt_tokens
            << " tokens in " << this->prompt_tokens / this->prefill_tokens_per_s << " sec)"
            << std::endl;
}

// Parametrized constructor
BackgroundLoops::BackgroundLoops(ffi::Module mod) : __mod(std::move(mod)), terminated(false) {
  auto background_loop = __mod.value()->GetFunction("run_background_loop");
  auto background_stream_back_loop =
      __mod.value()->GetFunction("run_background_stream_back_loop").value();

  background_loop_thread = (std::thread)(*background_loop);
  background_stream_back_loop_thread = (std::thread)(background_stream_back_loop);
}
BackgroundLoops::~BackgroundLoops() { terminate(); }

void BackgroundLoops::terminate() {
  if (!terminated) {
    terminated = true;

    try {
      __mod.value()->GetFunction("exit_background_loop").value()();
    } catch (const std::exception& e) {
      std::cerr << "Error calling exit_background_loop: " << e.what() << std::endl;
    }

    if (background_loop_thread.joinable()) {
      background_loop_thread.join();
    }
    if (background_stream_back_loop_thread.joinable()) {
      background_stream_back_loop_thread.join();
    }
  }
}

Completions::Completions(std::shared_ptr<EngineStateCli> engine_state, ffi::Module mod)
    : __mod(std::move(mod)), engine_state(std::move(engine_state)) {}

// Method to generate a unique string for each process
inline std::string Completions::GenerateUUID(size_t length) {
  auto randchar = []() -> char {
    const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[rand() % max_index];
  };
  std::string str(length, 0);
  std::generate_n(str.begin(), length, randchar);
  return str;
}

std::string Completions::create(std::vector<Message>& messages, int max_tokens,
                                int max_prompt_length, bool silent) {
  std::string request_id{""};
  // Method to generate random string
  std::string generate_random_string{GenerateUUID(16)};

  // Unique ID for each chat completion process
  request_id = "chatcmpl-" + generate_random_string;

  std::string history_string{""};
  std::string left_braces{"{"};
  std::string right_braces{"}"};
  std::string prompt = messagesToString(messages);
  std::string jsonStart = R"({)";
  std::string message_str = R"("messages":[)" + prompt + R"(])";
  std::string max_token_str = R"(, "max_tokens":)" + std::to_string(max_tokens);
  std::string max_prompt_length_str =
      (max_prompt_length > 0) ? (R"(, "max_prompt_length":)" + std::to_string(max_prompt_length))
                              : "";
  std::string jsonEnd = R"(})";

  std::string request_str =
      jsonStart + message_str + max_token_str + max_prompt_length_str + jsonEnd;
  std::string output_res =
      engine_state->handle_chat_completion(__mod, request_str, true, request_id, silent);
  return output_res;
}

Chat::Chat(std::shared_ptr<EngineStateCli> engine_state, ffi::Module mod)
    : completions(std::move(Completions(engine_state, mod))) {}

// Device str to DLDevice map
DLDeviceType GetDevice(std::string device) {
  if ("cuda" == device) {
    return kDLCUDA;
  } else if ("cpu" == device || "llvm" == device) {
    return kDLCPU;
  } else if ("opencl" == device) {
    return kDLOpenCL;
  } else if ("vulkan" == device) {
    return kDLVulkan;
  } else if ("metal" == device) {
    return kDLMetal;
  } else {
    LOG(FATAL) << "Unsupported device :" << device;
  }
}

JSONFFIEngineWrapper::JSONFFIEngineWrapper(std::string model_path, std::string model_lib_path,
                                           std::string mode, std::string device, int device_id = 0,
                                           int prefill_chunk_size = -1,
                                           int context_window_size = -1)
    : chat(nullptr),
      engine_config(nullptr),
      mod(std::nullopt),  // not constructed yet
      background_loops(nullptr),
      engine_state(nullptr) {
  // Create an instance of EngineStateCli
  this->engine_state = std::make_shared<EngineStateCli>();

  Optional<Function> engine = Function::GetGlobal("mlc.json_ffi.CreateJSONFFIEngine");
  if (engine == nullptr) {
    std::cout << "\nError: Unable to access TVM global registry mlc.json_ffi.CreateJSONFFIEngine"
              << std::endl;
  }

  auto module_tvm = (*engine)().cast<ffi::Module>();

  this->mod = module_tvm;

  // We can give mod as an argument to this
  background_loops = std::make_shared<BackgroundLoops>(this->mod.value());

  this->engine_config = std::make_shared<EngineConfig>(tvm::ffi::make_object<EngineConfigNode>());
  (*engine_config)->model = model_path;
  (*engine_config)->model_lib = model_lib_path;
  (*engine_config)->verbose = false;
  if (mode == "interactive") {
    (*engine_config)->mode = EngineMode::kInteractive;
  } else if (mode == "local") {
    (*engine_config)->mode = EngineMode::kLocal;
  } else if (mode == "server") {
    (*engine_config)->mode = EngineMode::kServer;
  }
  const std::string file_path = model_path + "/mlc-chat-config.json";
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open " << file_path << std::endl;
    // return 1;
  }

  std::string config_content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());

  // Parse the JSON object
  tvm::ffi::String err;
  auto config_object = tvm::ffi::json::Parse(config_content, &err);
  if (!err.empty()) {
    std::cerr << "Error: Unable to parse the JSON object: " << err << std::endl;
  }
  // Accessing the parsed data
  if (config_object.try_cast<tvm::ffi::json::Object>()) {
    const tvm::ffi::json::Object& model_config = config_object.cast<tvm::ffi::json::Object>();
    if (context_window_size == -1 &&
        model_config.find("context_window_size") != model_config.end()) {
      context_window_size = model_config.at("context_window_size").cast<double>();
    }
    if (prefill_chunk_size == -1 && model_config.find("prefill_chunk_size") != model_config.end()) {
      prefill_chunk_size = model_config.at("prefill_chunk_size").cast<double>();
    }
    (*engine_config)->prefill_chunk_size = prefill_chunk_size;
    (*engine_config)->max_total_sequence_length = context_window_size;
    (*engine_config)->max_single_sequence_length = context_window_size;
  } else {
    std::cerr << "Error: Invalid JSON format" << std::endl;
  }

  auto call_back = engine_state->get_request_stream_callback();

  // Typecasting to the TVM Packed Function
  auto tvm_callback = TypedFunction<void(std::string)>(call_back);
  // Call to Initialise Background Engine
  mod.value()
      ->GetFunction("init_background_engine")
      .value()(static_cast<int>(GetDevice(device)), device_id, tvm_callback);
  std::string engine_config_json_str{(*engine_config)->AsJSONString()};
  // Call to Reload Function of JSONFFIEngineImpl
  mod.value()->GetFunction("reload").value()(engine_config_json_str);
  chat = std::make_shared<Chat>(engine_state, mod.value());
}

void JSONFFIEngineWrapper::Reset() { mod.value()->GetFunction("reset").value()(); }
