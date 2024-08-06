#!/bin/sh
set -e

export ANDROID_SERIAL=$1

test_model() {
    model=$1

    adb shell "rm -rf /data/local/tmp/mlc-ci/models"
    adb shell "mkdir -p /data/local/tmp/mlc-ci/models"

    adb push ./dist/${model}-q4f16_0-MLC /data/local/tmp/mlc-ci/models/
    adb push ./dist/libs/${model}-q4f16_0-adreno.so /data/local/tmp/mlc-ci/models/
    adb push ./dist/libs/${model}-q4f16_0-adreno-accl.so /data/local/tmp/mlc-ci/models/

    adb shell "cd /data/local/tmp/mlc-ci; \
        LD_LIBRARY_PATH=./lib/ \
        ./bin/mlc_cli_chat \
        --model /data/local/tmp/mlc-ci/models/${model}-q4f16_0-MLC \
        --model-lib /data/local/tmp/mlc-ci/models/${model}-q4f16_0-adreno.so \
        --device opencl \
        --with-prompt hello"

    #adb shell "cd /data/local/tmp/mlc-ci; \
    #    LD_LIBRARY_PATH=./lib/ \
    #    ./bin/mlc_cli_chat \
    #    --model /data/local/tmp/mlc-ci/models/${model}-q4f16_0-MLC \
    #    --model-lib /data/local/tmp/mlc-ci/models/${model}-q4f16_0-adreno-accl.so \
    #    --device opencl \
    #    --with-prompt hello"
}

# Setup target
adb shell "rm -rf /data/local/tmp/mlc-ci"
adb shell "rm -rf /data/local/tmp/mlc-ci"
adb push build-arm64/mlc_llm-utils-linux-arm64 /data/local/tmp/mlc-ci/

MODELS="Llama-2-7b-chat-hf \
       Meta-Llama-3-8B-Instruct \
       Qwen-7B-Chat \
       Mistral-7B-Instruct-v0.2 \
       gemma-2b-it \
       phi-2 \
       Phi-3-mini-4k-instruct \
       llava-1.5-7b-hf \
       Baichuan-7B"

for i in ${MODELS}
do
  test_model $i
done
