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

ACCL=1
if [ "$3" == "NOACCL" ] || [ "$2" == "NOACCL" ] ; then
  ACCL=0
fi

build_model() {
    model=$1
    quantization=$2
    template=$3
    addl_args=$4

    python3 -m  mlc_llm gen_config ${MODEL_LOCAL_BASE}/${model} --quantization ${quantization} --conv-template ${template} ${addl_args} -o ${MODEL_ARTIFACTS_PATH}/dist/${model}-${quantization}-MLC
    python3 -m mlc_llm convert_weight ${MODEL_LOCAL_BASE}/${model} --quantization ${quantization} -o ${MODEL_ARTIFACTS_PATH}/dist/${model}-${quantization}-MLC/ --device cuda
    python3 -m mlc_llm compile ${MODEL_ARTIFACTS_PATH}/dist/${model}-${quantization}-MLC/mlc-chat-config.json --device android:adreno-so -o ${MODEL_ARTIFACTS_PATH}/dist/libs/${model}-${quantization}-adreno.so
    if [ $ACCL -eq 1 ] ; then
      python3 -m mlc_llm compile ${MODEL_ARTIFACTS_PATH}/dist/${model}-${quantization}-MLC/mlc-chat-config.json --device android:adreno-so --opt "adrenoaccl=1" -o ${MODEL_ARTIFACTS_PATH}/dist/libs/${model}-${quantization}-adreno-accl.so
    fi
}

# LLaMa-v2-7B
build_model Llama-2-7b-chat-hf q4f16_0 llama-2 "--prefill-chunk-size 256"

# LLaMa-v3-8B-Instruct
build_model Meta-Llama-3-8B-Instruct q4f16_0 llama-3 "--prefill-chunk-size 256"

# Qwen-7B
build_model Qwen-7B-Chat q4f16_0 chatml "--model-type qwen --prefill-chunk-size 256 --context-window-size 4096"

# Mistral-Instruct-7B
build_model Mistral-7B-Instruct-v0.2 q4f16_0 mistral_default "--sliding-window-size 1024 --prefill-chunk-size 256"

# Gemma-2G-it
build_model gemma-2b-it q4f16_0 gemma_instruction  "--prefill-chunk-size 256 --context-window-size 4096"

# Phi-2
build_model phi-2 q4f16_0 phi-2 "--prefill-chunk-size 256 --context-window-size 4096"

# Phi-3-mini-4k-instruct
build_model Phi-3-mini-4k-instruct q4f16_0 phi-3 "--prefill-chunk-size 256 --context-window-size 4096"

# llava-1.5-7b-hf
build_model llava-1.5-7b-hf q4f16_0 llava "--prefill-chunk-size 256 --context-window-size 4096"

# Baichuan-7B
# build_model Baichuan-7B q4f16_0 chatml "--model-type baichuan --prefill-chunk-size 256 --context-window-size 4096"

# DeepSeek-R1-Distill-Qwen-1.5B
build_model DeepSeek-R1-Distill-Qwen-1.5B q4f16_0 deepseek_r1_qwen "--prefill-chunk-size 256 --context-window-size 4096"

# DeepSeek-R1-Distill-Llama-8B
build_model DeepSeek-R1-Distill-Llama-8B q4f16_0 deepseek_r1_llama "--prefill-chunk-size 256 --context-window-size 4096"

# DeepSeek-R1-Distill-Qwen-7B - Weight conversion seem to have an error
#build_model DeepSeek-R1-Distill-Qwen-7B q4f16_0 deepseek_r1_qwen "--prefill-chunk-size 256 --context-window-size 4096"

