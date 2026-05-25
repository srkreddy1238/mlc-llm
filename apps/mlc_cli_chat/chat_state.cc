/*!
 *  Copyright (c) 2023 by Contributors
 * \file chat_state.cc
 */

#include "chat_state.h"

#include <iostream>
#include <string>

#include "base.h"
#include "engine.h"

void print_help_str() {
  std::string help_string = R"("""You can use the following special commands:
  /help               print the special commands
  /exit               quit the cli
  /reset              reset the history
  /stats              print out stats of last request (token/sec)
  Multi-line input: Use escape+enter to start a new line.
""")";

  std::cout << help_string << std::endl;
}

ChatState::ChatState(std::string model_path, std::string model_lib_path, std::string mode,
                     std::string device, int device_id, int prefill_chunk_size,
                     int context_window_size) {
  history_window_begin = 0;
  __json_wrapper = std::make_shared<JSONFFIEngineWrapper>(
      model_path, model_lib_path, mode, device, device_id, prefill_chunk_size, context_window_size);
}

void ChatState::slide_history() {
  size_t history_window_size = history.size() - history_window_begin;
  history_window_begin += ((history_window_size + 3) / 4) * 2;
}

std::vector<Message> ChatState::get_current_history_window() {
  return std::vector<Message>(history.begin() + history_window_begin, history.end());
}

void ChatState::warmup(int max_prompt_length) {
  std::cerr << "[mlc_cli_chat] Running warm-up iteration..." << std::endl;
  Message warm_msg;
  warm_msg.content["role"] = "user";
  warm_msg.content["content"] = "Hello";
  std::vector<Message> warm_window = {warm_msg};
  // Run a short generation silently (max_tokens=10, silent=true)
  (*__json_wrapper)
      .chat->completions.create(warm_window, /*max_tokens=*/10, max_prompt_length, /*silent=*/true);
  reset();
  std::cerr << "[mlc_cli_chat] Warm-up complete." << std::endl;
}

int ChatState::generate(const std::string& prompt, int max_tokens, int max_prompt_length) {
  // setting back the finish_reason_length
  bool finish_reason_length = false;

  // User Message
  Message new_message;
  new_message.content["role"] = "user";
  new_message.content["content"] = prompt;
  history.push_back(new_message);

  auto curr_window = get_current_history_window();

  std::string output_text{""};

  output_text =
      (*__json_wrapper).chat->completions.create(curr_window, max_tokens, max_prompt_length);

  if (__json_wrapper->engine_state->finish_reason == "length") {
    finish_reason_length = true;
  }

  if (finish_reason_length) {
    std::cout << "\n[output truncated due to context length limit...]";
  }

  Message assistant_response;
  assistant_response.content["role"] = "assistant";

  tvm::ffi::json::Value val(output_text);

  std::string output_json_str = tvm::ffi::json::Stringify(val);

  assistant_response.content["content"] = output_json_str;
  history.push_back(assistant_response);

  if (finish_reason_length) {
    slide_history();
  }
  return 0;
}

void ChatState::reset() {
  history.clear();
  history_window_begin = 0;
  this->__json_wrapper->Reset();
}

int ChatState::chat(std::string prompt, int max_tokens, int repeat, int max_prompt_length) {
  print_help_str();
  // Run a silent warm-up iteration to prime the engine before real work.
  warmup(max_prompt_length);
  // Get the prompt message
  if (!prompt.empty()) {
    int ret = 0;
    for (int i = 0; i < repeat; i++) {
      ret = generate(prompt, max_tokens, max_prompt_length);
      this->__json_wrapper->engine_state->getStats();
      reset();
    }
    __json_wrapper->background_loops->terminate();
    return 0;
  }
  std::string cin_prompt;
  while (true) {
    std::cout << ">>> ";
    std::getline(std::cin, cin_prompt);
    if (std::cin.eof() || cin_prompt == "/exit") {
      __json_wrapper->background_loops->terminate();
      break;
    } else if (cin_prompt == "/help") {
      print_help_str();
    } else if (cin_prompt == "/reset") {
      reset();
    } else if (cin_prompt == "/stats") {
      this->__json_wrapper->engine_state->getStats();
    } else {
      generate(cin_prompt, max_tokens, max_prompt_length);
    }
  }
  return 0;
}
