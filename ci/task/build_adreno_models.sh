#!/bin/sh
__conda_setup="$('/usr/local/workspace/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/workspace/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/local/workspace/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/workspace/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate mlc-build-venv

# Environment setup
export PYTHONPATH=./python:$PYTHONPATH
export PYTHONPATH=$PWD/3rdparty/tvm/python:$PYTHONPATH
export PATH=/usr/local/cuda-12.2/bin:$PATH
python -c "import mlc_llm; print(mlc_llm.__path__)"
export TVM_NDK_CC=`cat /etc/tvm-ndk-cc`
export MODEL_LOCAL_BASE=`cat /etc/mlc-model-repo`
export MODEL_ARTIFACTS_PATH=`cat /etc/mlc-artifacts-path`

# Artifacts folder
mkdir ${MODEL_ARTIFACTS_PATH}/dist -p
mkdir ${MODEL_ARTIFACTS_PATH}/dist/libs/ -p

# LLaMa-v2-7B
python3 -m  mlc_llm gen_config ${MODEL_LOCAL_BASE}/Llama-2-7b-chat-hf --quantization q4f16_0 --conv-template llama-2 -o ${MODEL_ARTIFACTS_PATH}/dist/Llama-2-7b-chat-hf-q4f16_0-MLC
python3 -m mlc_llm convert_weight ${MODEL_LOCAL_BASE}/Llama-2-7b-chat-hf --quantization q4f16_0 -o ${MODEL_ARTIFACTS_PATH}/dist/Llama-2-7b-chat-hf-q4f16_0-MLC/ --device cuda
python3 -m mlc_llm compile ${MODEL_ARTIFACTS_PATH}/dist/Llama-2-7b-chat-hf-q4f16_0-MLC/mlc-chat-config.json --device android -o ${MODEL_ARTIFACTS_PATH}/dist/libs/Llama-2-7b-chat-hf-q4f16_0-android.so

# Mistral-Instruct-7B
python3 -m  mlc_llm gen_config ${MODEL_LOCAL_BASE}/Mistral-7B-Instruct-v0.2/ --quantization q4f16_0 --conv-template mistral_default --sliding-window-size 1024 --prefill-chunk-size 128 -o ${MODEL_ARTIFACTS_PATH}/dist/Mistral-7B-Instruct-v0.2-q4f16_0-MLC/
python3 -m  mlc_llm convert_weight ${MODEL_LOCAL_BASE}/Mistral-7B-Instruct-v0.2/ --quantization q4f16_0 -o ${MODEL_ARTIFACTS_PATH}/dist/Mistral-7B-Instruct-v0.2-q4f16_0-MLC --device cuda
python3 -m  mlc_llm compile ${MODEL_ARTIFACTS_PATH}/dist/Mistral-7B-Instruct-v0.2-q4f16_0-MLC/mlc-chat-config.json --device android -o ${MODEL_ARTIFACTS_PATH}/dist/libs/Mistral-7B-Instruct-v0.2-q4f16_0-android.so
