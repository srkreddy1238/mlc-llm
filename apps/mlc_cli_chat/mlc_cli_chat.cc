/*!
 *  Copyright (c) 2023 by Contributors
 * \file mlc_cli_chat.cc
 */

#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "chat_state.h"
#include "engine.h"

static std::string read_stdin_all() {
  std::ostringstream oss;
  oss << std::cin.rdbuf();
  return oss.str();
}

static std::string read_file_all(const std::string& path, size_t max_bytes = 10 * 1024 * 1024) {
  struct stat st {};
  if (stat(path.c_str(), &st) != 0) {
    throw std::runtime_error("Cannot stat file: " + path);
  }
  if (static_cast<size_t>(st.st_size) > max_bytes) {
    throw std::runtime_error("File too large (limit " + std::to_string(max_bytes) +
                             " bytes): " + path);
  }
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::string data;
  data.resize(static_cast<size_t>(st.st_size));
  if (!ifs.read(&data[0], data.size())) {
    throw std::runtime_error("Failed to read file: " + path);
  }

  if (data.size() >= 3 && (unsigned char)data[0] == 0xEF && (unsigned char)data[1] == 0xBB &&
      (unsigned char)data[2] == 0xBF) {
    data.erase(0, 3);
  }

  std::string out;
  out.reserve(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    if (data[i] == '\r') {
      if (i + 1 < data.size() && data[i + 1] == '\n') continue;
    }
    out.push_back(data[i]);
  }

  if (!out.empty() && out.back() == '\n') out.pop_back();
  return out;
}

struct Args {
  std::string model;
  std::string model_lib_path;
  std::string device = "auto";
  bool evaluate = false;
  int eval_prompt_len = 128;
  int max_tokens = -1;
  std::string prompt;
  int repeat = 1;
  // New field to carry a file path if provided
  std::string prompt_file;
};

// Help Prompt
void printHelp() {
  std::cout
      << "MLCChat CLI is the command line tool to run MLC-compiled LLMs out of the box.\n"
      << "Note: the --model argument is required. It can either be the model name with its "
      << "quantization scheme or a full path to the model folder. In the former case, the "
      << "provided name will be used to search for the model folder over possible paths. "
      << "--model-lib-path argument is optional. If unspecified, the --model argument will be used "
      << "to search for the library file over possible paths.\n\n"
      << "Usage: mlc_cli_chat [options]\n"
      << "Options:\n"
      << "  --model             [required] the model to use\n"
      << "  --model-lib         [optional] the full path to the model library file to use\n"
      << "  --device            (default: auto)\n"
      << "  --with-prompt       [optional] runs one session with given prompt\n"
      << "  --max-tokens        [optional] generate given number of token [default: -1 "
         "(infinite)]\n"
      << "  --repeat            [optional] Repeat the application with desire interation (default "
         "1) "
         "by reseting history. it is ignore for chat mode."
      << "  --with-prompt-file <path>  [optional] read prompt from a text file\n"
      << "  --help              [optional] Tool usage information\n"
      /*
      << "  --evaluate          (flag, default: false)\n"
      << "  --eval-prompt-len   (default: 128)\n"
      << "  --eval-gen-len      (default: 1024)\n"
      */
      ;
}

// Method to parse the args
Args parseArgs(int argc, char* argv[]) {
  Args args;

  // Taking the arguments after the exectuable
  std::vector<std::string> arguments(argv + 1, argv + argc);

  for (size_t i = 0; i < arguments.size(); ++i) {
    if (arguments[i] == "--model" && i + 1 < arguments.size()) {
      args.model = arguments[++i];
    } else if (arguments[i] == "--model-lib" && i + 1 < arguments.size()) {
      args.model_lib_path = arguments[++i];
    } else if (arguments[i] == "--device" && i + 1 < arguments.size()) {
      args.device = arguments[++i];
    } else if (arguments[i] == "--evaluate") {
      args.evaluate = true;
    } else if (arguments[i] == "--max-tokens" && i + 1 < arguments.size()) {
      args.max_tokens = std::stoi(arguments[++i]);
    } else if (arguments[i] == "--repeat" && i + 1 < arguments.size()) {
      args.repeat = std::stoi(arguments[++i]);
    } else if (arguments[i] == "--with-prompt" && i + 1 < arguments.size()) {
      args.prompt = arguments[++i];
    } else if (arguments[i].rfind("--with-prompt=", 0) == 0) {
      args.prompt = arguments[i].substr(std::string("--with-prompt=").size());
    }
    // New flag forms for file ===
    else if (arguments[i] == "--with-prompt-file" && i + 1 < arguments.size()) {
      args.prompt_file = arguments[++i];
    } else if (arguments[i].rfind("--with-prompt-file=", 0) == 0) {
      args.prompt_file = arguments[i].substr(std::string("--with-prompt-file=").size());
    } else if (arguments[i] == "--help") {
      printHelp();
      exit(0);
    } else {
      printHelp();
      throw std::runtime_error("Unknown or incomplete argument: " + arguments[i]);
    }
  }

  if (args.model.empty()) {
    printHelp();
    throw std::runtime_error("Invalid arguments: --model is required");
  }
  // -------------------------------
  // Resolve prompt source(s)
  // -------------------------------
  if (!args.prompt.empty() && !args.prompt_file.empty()) {
    throw std::runtime_error("Use either --with-prompt or --with-prompt-file, not both.");
  }
  if (!args.prompt_file.empty()) {
    // explicit file wins
    args.prompt = read_file_all(args.prompt_file);
    args.prompt_file.clear();
  } else if (!args.prompt.empty()) {
    // shortcuts on existing flag
    if (args.prompt == "-") {
      args.prompt = read_stdin_all();
    } else if (!args.prompt.empty() && args.prompt.front() == '@') {
      args.prompt = read_file_all(args.prompt.substr(1));
    }
  }

  return args;
}

// Method to detect the device
static std::pair<std::string, int> DetectDevice(std::string device) {
  std::string device_name;
  int device_id;
  auto delimiter_pos = device.find(":");

  // cuda:0 which means the device name is cuda and the device id is 0
  if (delimiter_pos == std::string::npos) {
    device_name = device;
    device_id = 0;
  } else {
    device_name = device.substr(0, delimiter_pos);
    device_id = std::stoi(device.substr(delimiter_pos + 1, device.length()));
  }
  return {device_name, device_id};
}

int main(int argc, char* argv[]) {
  Args args = parseArgs(argc, argv);
  // model path
  std::string model_path = args.model;

  // model-lib path
  std::string model_lib_path = args.model_lib_path;

  // Get the device name and device id
  auto [device_name, device_id] = DetectDevice(args.device);
  // mode of interaction
  std::string mode{"interactive"};

  ChatState chat_state(model_path, model_lib_path, mode, device_name, 0);

  return chat_state.chat(args.prompt, args.max_tokens, args.repeat);
}
