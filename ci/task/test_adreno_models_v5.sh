#!/bin/sh
set -e

export ANDROID_SERIAL=$1
if [ -d "$2" ] ; then
    export MODEL_ARTIFACTS_PATH=$2
else
    export MODEL_ARTIFACTS_PATH="."
fi

test_model() {
    model=$1

    adb shell "rm -rf /data/local/tmp/mlc-ci/models"
    adb shell "mkdir -p /data/local/tmp/mlc-ci/models"

    adb push $MODEL_ARTIFACTS_PATH/dist/${model}-q4f16_0-MLC /data/local/tmp/mlc-ci/models/
    adb push $MODEL_ARTIFACTS_PATH/dist/libs/${model}-q4f16_0-adreno-vulkan-qcom.so /data/local/tmp/mlc-ci/models/
   
    adb shell "cd /data/local/tmp/mlc-ci; \
        LD_LIBRARY_PATH=./lib/ \
        ./bin/mlc_cli_chat \
        --model /data/local/tmp/mlc-ci/models/${model}-q4f16_0-MLC \
        --model-lib /data/local/tmp/mlc-ci/models/${model}-q4f16_0-adreno-vulkan-qcom.so \
        --max-tokens 100 \
        --context-window-size 1024 \
        --device vulkan \
        --with-prompt \"write a poem about moon in 100 words\""
}

# Setup target
adb shell "rm -rf /data/local/tmp/mlc-ci"
adb shell "rm -rf /data/local/tmp/mlc-ci"
adb push build-arm64/mlc_llm-utils-linux-arm64 /data/local/tmp/mlc-ci/

MODELS="Llama-3.2-3B-Instruct \
       Mistral-7B-Instruct-v0.2 \
       gemma-2b-it \
       phi-2 \
       Phi-3.5-mini-instruct \
       llava-1.5-7b-hf \
       DeepSeek-R1-Distill-Qwen-1.5B \
       Qwen2.5-1.5B-Instruct"

for i in ${MODELS}
do
  test_model $i
  sleep 5
done
