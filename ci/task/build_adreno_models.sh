#!/bin/sh
set -e

export MODEL_LOCAL_BASE=$1
if [ -d "$2" ] ; then
    export MODEL_ARTIFACTS_PATH=$2
else
    export MODEL_ARTIFACTS_PATH="./"
fi

# Artifacts folder
mkdir ${MODEL_ARTIFACTS_PATH}/dist/libs -p

ACCL=0
if [ "$3" = "ACCL" ] || [ "$2" = "ACCL" ] ; then
  ACCL=1
fi

build_model() {
    model=$1
    quantization=$2
    template=$3
    addl_args=$4

    python3 -m  mlc_llm gen_config ${MODEL_LOCAL_BASE}/${model} --quantization ${quantization} --conv-template ${template} ${addl_args} -o ${MODEL_ARTIFACTS_PATH}/dist/${model}-${quantization}-MLC
    python3 -m mlc_llm convert_weight ${MODEL_LOCAL_BASE}/${model} --quantization ${quantization} -o ${MODEL_ARTIFACTS_PATH}/dist/${model}-${quantization}-MLC/ --device cuda
    python3 -m mlc_llm compile ${MODEL_ARTIFACTS_PATH}/dist/${model}-${quantization}-MLC/mlc-chat-config.json --device android:cl-adreno-so -o ${MODEL_ARTIFACTS_PATH}/dist/libs/${model}-${quantization}-adreno-opencl.so
    if [ $ACCL -eq 1 ] ; then
      python3 -m mlc_llm compile ${MODEL_ARTIFACTS_PATH}/dist/${model}-${quantization}-MLC/mlc-chat-config.json --device android:cl-adreno-so --opt "adrenoaccl=1" -o ${MODEL_ARTIFACTS_PATH}/dist/libs/${model}-${quantization}-adreno-accl.so
    fi
    python3 -m mlc_llm compile ${MODEL_ARTIFACTS_PATH}/dist/${model}-${quantization}-MLC/mlc-chat-config.json --device android:vk-adreno-so -o ${MODEL_ARTIFACTS_PATH}/dist/libs/${model}-${quantization}-adreno-vulkan.so
    python3 -m mlc_llm compile ${MODEL_ARTIFACTS_PATH}/dist/${model}-${quantization}-MLC/mlc-chat-config.json --device android:vk-qcom-adreno-so -o ${MODEL_ARTIFACTS_PATH}/dist/libs/${model}-${quantization}-adreno-vulkan-qcom.so
}

# LLaMa-v3-8B-Instruct
build_model Meta-Llama-3-8B-Instruct q4f16_0 llama-3 "--prefill-chunk-size 256"

# LLaMa-v3.2-3B-Instruct
build_model Llama-3.2-3B-Instruct q4f16_0 llama-3 "--prefill-chunk-size 256"

# Qwen-7B
build_model Qwen-7B-Chat q4f16_0 chatml "--model-type qwen --prefill-chunk-size 256 --context-window-size 4096"

# Mistral-Instruct-7B
build_model Mistral-7B-Instruct-v0.2 q4f16_0 mistral_default "--sliding-window-size 1024 --prefill-chunk-size 256"

# Gemma-2G-it
build_model gemma-2b-it q4f16_0 gemma_instruction  "--prefill-chunk-size 256 --context-window-size 4096"

# Phi-2
build_model phi-2 q4f16_0 phi-2 "--prefill-chunk-size 256 --context-window-size 4096"

# Phi-3.5-mini-instruct
build_model Phi-3.5-mini-instruct q4f16_0 phi-3 "--prefill-chunk-size 256 --context-window-size 4096"

# llava-1.5-7b-hf
build_model llava-1.5-7b-hf q4f16_0 llava "--prefill-chunk-size 256 --context-window-size 4096"

# DeepSeek-R1-Distill-Qwen-1.5B
build_model DeepSeek-R1-Distill-Qwen-1.5B q4f16_0 deepseek_r1_qwen "--prefill-chunk-size 256 --context-window-size 4096"

# DeepSeek-R1-Distill-Llama-8B
build_model DeepSeek-R1-Distill-Llama-8B q4f16_0 deepseek_r1_llama "--prefill-chunk-size 256 --context-window-size 4096"

# Qwen2.5-1.5B-Instruct
build_model Qwen2.5-1.5B-Instruct q4f16_0 qwen2 "--model-type qwen2 --prefill-chunk-size 256 --context-window-size 4096"

# Qwen2.5-0.5B-Instruct
build_model Qwen2.5-0.5B-Instruct q4f16_0 qwen2 "--model-type qwen2 --prefill-chunk-size 256 --context-window-size 4096"
