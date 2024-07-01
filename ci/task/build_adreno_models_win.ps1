$ENV:PYTHONPATH = "$pwd\3rdparty\tvm\python;python"
python -c "import mlc_llm; print(mlc_llm.__path__)"

New-Item -ItemType Directory -Force -Path "./dist"
New-Item -ItemType Directory -Force -Path "./dist/libs"

python3 -m mlc_llm gen_config C:\sivb\MLC\models\Llama-2-7b-chat-hf --quantization q4f16_0 --conv-template llama-2 -o ./dist/Llama-2-7b-chat-hf-q4f16_0-MLC
python3 -m mlc_llm compile ./dist/Llama-2-7b-chat-hf-q4f16_0-MLC/mlc-chat-config.json --device opencl -o ./dist/libs/Llama-2-7b-chat-hf-q4f16_0-windows.dll
