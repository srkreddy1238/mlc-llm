#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}
export PYTHONPATH="./python":${PYTHONPATH:-""}

if [[ -n ${MLC_CI_SETUP_DEPS:-} ]]; then
    echo "MLC_CI_SETUP_DEPS=1 start setup deps"
    # TVM Unity is a dependency to this testing
    pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu
    pip install apache-tvm-ffi
    pip install requests triton
    pip install --quiet --pre -U cuda-python
    pip install --quiet --pre -U pydantic openai fastapi shortuuid
fi

pylint --jobs $NUM_THREADS ./python/
pylint --jobs $NUM_THREADS --recursive=y ./tests/python/
pylint --jobs $NUM_THREADS --recursive=y ./examples/python/
