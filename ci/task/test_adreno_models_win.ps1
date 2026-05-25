# Set the error handling to stop execution on error
$ErrorActionPreference = "Stop"

# Define the base path for the model files
if (Test-Path -Path $args[0] -PathType Container) {
    $MODEL_ARTIFACTS_PATH = $args[0]
} else {
    $MODEL_ARTIFACTS_PATH = "./"
}

Get-Item -LiteralPath .\mlc_llm-utils-win-arm64-target\ -ErrorAction SilentlyContinue -ErrorVariable errs | Remove-Item -Recurse -Verbose
Get-Item -LiteralPath .\mlc_llm-utils-win-x86-target\ -ErrorAction SilentlyContinue -ErrorVariable errs | Remove-Item -Recurse -Verbose
Copy-Item -Path $MODEL_ARTIFACTS_PATH\dist\libs -Destination libs -Recurse

Expand-Archive -Path "$MODEL_ARTIFACTS_PATH/mlc_llm-utils-win-arm64-target.zip" -DestinationPath ./mlc_llm-utils-win-arm64-target
Expand-Archive -Path "$MODEL_ARTIFACTS_PATH/mlc_llm-utils-win-x86-target.zip" -DestinationPath ./mlc_llm-utils-win-x86-target

function run_cmd {
    param(
        [string]$cmd
    )
    Write-Host $cmd
    for ($i = 1; $i -le 4; $i++) {
        $global:LASTEXITCODE = 0
        Invoke-Expression -Command $cmd -ErrorAction "Stop"
        if ($LASTEXITCODE -eq 0) {
            break
        }
    }
}

# Function to build the model
function test-model {
    param(
        [string]$model
    )
    # Arm64 Test
    $global:LASTEXITCODE = 0
    run_cmd ".\mlc_llm-utils-win-arm64-target\bin\mlc_cli_chat.exe --model C:\CI\LLM-Weights\$model-q4f16_0-MLC --model-lib .\libs\$model-q4f16_0-opencl-adreno-arm64.dll --device opencl --max-tokens 100 --context-window-size 1024 --with-prompt `"What is the capital of India ?`""
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $global:LASTEXITCODE = 0
    run_cmd ".\mlc_llm-utils-win-arm64-target\bin\mlc_cli_chat.exe --model C:\CI\LLM-Weights\$model-q4f16_0-MLC --model-lib .\libs\$model-q4f16_0-adreno-clml-arm64.dll --device opencl --max-tokens 100 --context-window-size 1024 --with-prompt `"What is the capital of India ?`""
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }


    # x64 Test
    $global:LASTEXITCODE = 0
    run_cmd ".\mlc_llm-utils-win-x86-target\bin\mlc_cli_chat.exe --model C:\CI\LLM-Weights\$model-q4f16_0-MLC --model-lib .\libs\$model-q4f16_0-opencl-adreno-x86.dll --device opencl --max-tokens 100 --context-window-size 1024 --with-prompt `"What is the capital of India ?`""
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $global:LASTEXITCODE = 0
    run_cmd ".\mlc_llm-utils-win-x86-target\bin\mlc_cli_chat.exe --model C:\CI\LLM-Weights\$model-q4f16_0-MLC --model-lib .\libs\$model-q4f16_0-adreno-clml-x86.dll --device opencl --max-tokens 100 --context-window-size 1024 --with-prompt `"What is the capital of India ?`""
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # Remove weights as CI may get full soon
    # Remove-Item -Path "$MODEL_ARTIFACTS_PATH\dist\$model-q4f16_0-MLC" -Recurse -Force
}

# Build the models

test-model Meta-Llama-3-8B-Instruct
test-model Llama-3.2-3B-Instruct
test-model Mistral-7B-Instruct-v0.2
test-model gemma-2b-it
test-model phi-2
test-model Phi-3.5-mini-instruct
test-model llava-1.5-7b-hf
test-model DeepSeek-R1-Distill-Qwen-1.5B
test-model Qwen2.5-1.5B-Instruct
