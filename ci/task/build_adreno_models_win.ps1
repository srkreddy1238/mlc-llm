$ErrorActionPreference = "Stop"

$MODEL_LOCAL_BASE = $args[0]
if (Test-Path -Path $args[1] -PathType Container) {
    $MODEL_ARTIFACTS_PATH = $args[1]
} else {
    $MODEL_ARTIFACTS_PATH = "./"
}

New-Item -ItemType Directory -Path "${MODEL_ARTIFACTS_PATH}/dist/libs" -Force

$CLML = 0
ForEach ($arg in $args){
  if ($arg -eq "CLML") {
    $CLML = 1
  }
}

# ---------------------------------------------------------------------------
# Capture parent context to pass into Start-Job isolated runspaces.
# ---------------------------------------------------------------------------

$parentModelLocalBase     = $MODEL_LOCAL_BASE
$parentModelArtifactsPath = $MODEL_ARTIFACTS_PATH
$parentCLML               = $CLML
# Start-Job always starts in $HOME - capture the real invocation directory.
$parentWorkDir            = $PWD.Path

# Start-Job serializes args via CLIXML; hashtables deserialize as PSObject
# and lose their enumerator. Serialize env as "KEY=VALUE" lines instead -
# plain strings survive CLIXML and are restored inside each job.
$rawEnv = [System.Environment]::GetEnvironmentVariables()
$parentEnvString = ($rawEnv.Keys | ForEach-Object {
    $val = $rawEnv[$_]
    if ($val -notmatch "`n") { "$_=$val" }
}) -join "`n"

$buildScript = {
    param(
        [string]$model,
        [string]$quantization,
        [string]$template,
        [string]$addl_args,
        [string]$MODEL_LOCAL_BASE,
        [string]$MODEL_ARTIFACTS_PATH,
        [int]   $CLML,
        [string]$parentEnvString,
        [string]$parentWorkDir
    )

    # Continue prevents python stderr lines triggering NativeCommandError
    # through the job remoting layer. Failures are caught via $LASTEXITCODE.
    $ErrorActionPreference = "Continue"

    # Restore parent environment (PATH, PYTHONPATH, conda activation, etc.)
    foreach ($line in ($parentEnvString -split "`n")) {
        $sep = $line.IndexOf('=')
        if ($sep -gt 0) {
            $envKey = $line.Substring(0, $sep)
            $envVal = $line.Substring($sep + 1)
            [System.Environment]::SetEnvironmentVariable($envKey, $envVal, 'Process')
        }
    }

    Set-Location -LiteralPath $parentWorkDir  # Start-Job starts in $HOME

    function build-model {
        param(
            [string]$model,
            [string]$quantization,
            [string]$template,
            [string]$addl_args
        )

        $distDir = "${MODEL_ARTIFACTS_PATH}\dist\${model}-${quantization}-MLC"
        $cfgFile = "${MODEL_ARTIFACTS_PATH}\dist\${model}-${quantization}-MLC\mlc-chat-config.json"
        $libsDir = "${MODEL_ARTIFACTS_PATH}\dist\libs"

        # Step 1: gen_config must finish before any compile step.
        Write-Host "[${model}] Generating config..."
        $genArgs = @("-m", "mlc_llm", "gen_config",
                     "${MODEL_LOCAL_BASE}\${model}",
                     "--quantization", $quantization,
                     "--conv-template", $template) +
                   ($addl_args -split ' ') +
                   @("-o", $distDir)
        & python @genArgs 2>&1
        $genExitCode = $LASTEXITCODE
        if ($genExitCode -ne 0) {
            Write-Host "[${model}] ERROR: gen_config failed (exit $genExitCode)" -ForegroundColor Red
            Write-Output "EXIT_CODE:$genExitCode"  # sentinel - job State carries no numeric code
            exit $genExitCode
        }

        # Step 2: compile all targets in parallel as inner sub-jobs.
        $compileTargets = [System.Collections.Generic.List[object]]::new()
        $compileTargets.Add(@{
            label    = "opencl-adreno-x86"
            device   = "windows:cl-adreno_x86"
            outDll   = "${libsDir}\${model}-${quantization}-opencl-adreno-x86.dll"
            optValue = ""
        })
        $compileTargets.Add(@{
            label    = "opencl-adreno-arm64"
            device   = "windows:cl-adreno_arm64"
            outDll   = "${libsDir}\${model}-${quantization}-opencl-adreno-arm64.dll"
            optValue = ""
        })
        $compileTargets.Add(@{
            label    = "vulkan-qcom-adreno-x86"
            device   = "windows:vk-qcom-adreno_x86"
            outDll   = "${libsDir}\${model}-${quantization}-vulkan-qcom-adreno-x86.dll"
            optValue = ""
        })
        $compileTargets.Add(@{
            label    = "vulkan-qcom-adreno-arm64"
            device   = "windows:vk-qcom-adreno_arm64"
            outDll   = "${libsDir}\${model}-${quantization}-vulkan-qcom-adreno-arm64.dll"
            optValue = ""
        })
        if ($CLML -eq 1) {
            $compileTargets.Add(@{
                label    = "adreno-clml-x86"
                device   = "windows:cl-adreno_x86"
                outDll   = "${libsDir}\${model}-${quantization}-adreno-clml-x86.dll"
                optValue = "openclml=1"
            })
            $compileTargets.Add(@{
                label    = "adreno-clml-arm64"
                device   = "windows:cl-adreno_arm64"
                outDll   = "${libsDir}\${model}-${quantization}-adreno-clml-arm64.dll"
                optValue = "openclml=1"
            })
        }

        $compileScript = {
            param(
                [string]$model,
                [string]$label,
                [string]$cfgFile,
                [string]$device,
                [string]$outDll,
                [string]$optValue,
                [string]$parentEnvString,
                [string]$parentWorkDir
            )
            $ErrorActionPreference = "Continue"
            foreach ($line in ($parentEnvString -split "`n")) {
                $sep = $line.IndexOf('=')
                if ($sep -gt 0) {
                    [System.Environment]::SetEnvironmentVariable(
                        $line.Substring(0, $sep),
                        $line.Substring($sep + 1),
                        'Process')
                }
            }
            Set-Location -LiteralPath $parentWorkDir

            Write-Host "[${model}] Compiling ${label}..."
            # Build args as an array - each element is one argv token.
            # Passing "--opt openclml=1" as a single string would fail.
            $pyArgs = @("-m", "mlc_llm", "compile", $cfgFile,
                        "--device", $device)
            if ($optValue -ne "") {
                $pyArgs += @("--opt", $optValue)
            }
            $pyArgs += @("-o", $outDll)
            & python @pyArgs 2>&1
            $compileExitCode = $LASTEXITCODE
            if ($compileExitCode -ne 0) {
                Write-Host "[${model}] ERROR: ${label} failed (exit $compileExitCode)" -ForegroundColor Red
                Write-Output "EXIT_CODE:$compileExitCode"  # sentinel for parent
                exit $compileExitCode
            }
            Write-Host "[${model}] Done: ${label}"
            Write-Output "EXIT_CODE:0"
        }

        $subJobs = [System.Collections.Generic.List[object]]::new()
        foreach ($tgt in $compileTargets) {
            $subJobs.Add(
                (Start-Job -Name "${model}-$($tgt.label)" `
                           -ScriptBlock $compileScript `
                           -ArgumentList `
                               $model,
                               $tgt.label,
                               $cfgFile,
                               $tgt.device,
                               $tgt.outDll,
                               $tgt.optValue,
                               $parentEnvString,
                               $parentWorkDir)
            )
        }

        $failedTargets = @()
        $worstExitCode = 0
        foreach ($sj in $subJobs) {
            $output       = $sj | Wait-Job | Receive-Job
            $sentinelLine = $output | Where-Object { $_ -match '^EXIT_CODE:' } | Select-Object -Last 1
            $subExitCode  = if ($sentinelLine) { [int]($sentinelLine -replace '^EXIT_CODE:', '') } else { 0 }
            if ($sj.State -eq 'Failed' -or $subExitCode -ne 0) {
                Write-Host "[${model}] Sub-job FAILED: $($sj.Name) (exit $subExitCode)" -ForegroundColor Red
                $sj.ChildJobs | ForEach-Object {
                    $_.Error | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
                }
                $failedTargets += $sj.Name
                if ($subExitCode -gt $worstExitCode) { $worstExitCode = $subExitCode }
            }
        }
        $subJobs | Remove-Job -Force -ErrorAction SilentlyContinue

        if ($failedTargets.Count -gt 0) {
            Write-Host "[${model}] $($failedTargets.Count) compile target(s) failed." -ForegroundColor Red
            Write-Output "EXIT_CODE:$worstExitCode"
            exit $worstExitCode
        }

        Write-Host "[${model}] All targets built successfully."
        Write-Output "EXIT_CODE:0"
    }

    build-model -model $model -quantization $quantization -template $template -addl_args $addl_args
}

$modelConfigs = @(
    @{ model = "Meta-Llama-3-8B-Instruct";       quantization = "q4f16_0"; template = "llama-3";           addl_args = "--prefill-chunk-size 256 --context-window-size 4096" },
    @{ model = "Llama-3.2-3B-Instruct";          quantization = "q4f16_0"; template = "llama-3";           addl_args = "--prefill-chunk-size 256 --context-window-size 4096" },
    @{ model = "Mistral-7B-Instruct-v0.2";       quantization = "q4f16_0"; template = "mistral_default";   addl_args = "--sliding-window-size 1024 --prefill-chunk-size 256 --context-window-size 4096" },
    @{ model = "gemma-2b-it";                    quantization = "q4f16_0"; template = "gemma_instruction"; addl_args = "--prefill-chunk-size 256 --context-window-size 4096" },
    @{ model = "Phi-3.5-mini-instruct";          quantization = "q4f16_0"; template = "phi-3";             addl_args = "--prefill-chunk-size 256 --context-window-size 4096" },
    @{ model = "llava-1.5-7b-hf";                quantization = "q4f16_0"; template = "llava";             addl_args = "--prefill-chunk-size 256 --context-window-size 4096" },
    @{ model = "DeepSeek-R1-Distill-Qwen-1.5B";  quantization = "q4f16_0"; template = "deepseek_r1_qwen"; addl_args = "--prefill-chunk-size 256 --context-window-size 4096" },
    @{ model = "Qwen2.5-1.5B-Instruct";         quantization = "q4f16_0"; template = "qwen2";             addl_args = "--model-type qwen2 --prefill-chunk-size 256 --context-window-size 4096" }
    @{ model = "Qwen3.5-0.8B";                   quantization = "q4f16_0"; template = "qwen3_5";             addl_args = "--model-type qwen3_5 --prefill-chunk-size 256 --context-window-size 4096" }
    @{ model = "Qwen1.5-MoE-A2.7B-Chat";         quantization = "q4f16_0"; template = "qwen2";             addl_args = "--model-type qwen2_moe --prefill-chunk-size 256 --context-window-size 4096" }
)

# Stops all running jobs, kills their python child processes, and cleans up.
# Called from the Ctrl+C handler and the finally block.
function Stop-AllJobs {
    param([System.Collections.Generic.List[object]]$jobList)

    if ($jobList.Count -eq 0) { return }

    Write-Host "`n==> Stopping all running jobs..." -ForegroundColor Yellow
    foreach ($job in $jobList) {
        if ($job.State -eq 'Running') {
            Write-Host "    Stopping job: $($job.Name)" -ForegroundColor Yellow
            # Kill the worker process and its python child tree.
            $job.ChildJobs | ForEach-Object {
                if ($_.ProcessId) {
                    try {
                        $workerPid = $_.ProcessId
                        $workerProc = Get-Process -Id $workerPid -ErrorAction SilentlyContinue
                        if ($workerProc) {
                            Get-CimInstance Win32_Process |
                                Where-Object { $_.ParentProcessId -eq $workerPid } |
                                ForEach-Object {
                                    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
                                }
                            Stop-Process -Id $workerPid -Force -ErrorAction SilentlyContinue
                        }
                    } catch {}
                }
            }
            Stop-Job -Job $job
        }
        Receive-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
    }
    $jobList | Remove-Job -Force -ErrorAction SilentlyContinue
    $jobList.Clear()
    Write-Host "==> All jobs stopped and cleaned up." -ForegroundColor Yellow
}

# Use List (not array) so the Ctrl+C closure captures the live reference.
$jobs = [System.Collections.Generic.List[object]]::new()

# Register Ctrl+C handler before launching jobs.
# Cancel=true prevents immediate process kill, giving cleanup time to run.
$ctrlCHandler = {
    $args[1].Cancel = $true
    Write-Host "`n==> Ctrl+C detected - cancelling all build jobs..." -ForegroundColor Yellow
    Stop-AllJobs -jobList $jobs
    [System.Environment]::Exit(130)
}
[Console]::add_CancelKeyPress($ctrlCHandler)

Write-Host "==> Launching $($modelConfigs.Count) parallel build jobs..."
foreach ($cfg in $modelConfigs) {
    Write-Host "    Starting job: build-model $($cfg.model) $($cfg.quantization) $($cfg.template)"
    $jobs.Add(
        (Start-Job -Name $cfg.model -ScriptBlock $buildScript -ArgumentList `
            $cfg.model,
            $cfg.quantization,
            $cfg.template,
            $cfg.addl_args,
            $parentModelLocalBase,
            $parentModelArtifactsPath,
            $parentCLML,
            $parentEnvString,
            $parentWorkDir)
    )
}

# finally block runs on normal exit, exception, and Ctrl+C pipeline-stop.
$failedJobs    = @()
$finalExitCode = 0
try {
    Write-Host "==> Waiting for all $($jobs.Count) jobs to complete... (Ctrl+C to cancel)"
    foreach ($job in $jobs) {
        $output       = $job | Wait-Job | Receive-Job
        # job.State carries no numeric code - parse the EXIT_CODE sentinel.
        $sentinelLine = $output | Where-Object { $_ -match '^EXIT_CODE:' } | Select-Object -Last 1
        $jobExitCode  = if ($sentinelLine) { [int]($sentinelLine -replace '^EXIT_CODE:', '') } else { 0 }
        if ($job.State -eq 'Failed' -or $jobExitCode -ne 0) {
            Write-Host "ERROR: Job FAILED for model: $($job.Name) (exit $jobExitCode)" -ForegroundColor Red
            $job.ChildJobs | ForEach-Object {
                $_.Error | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
            }
            $failedJobs += $job
            if ($jobExitCode -gt $finalExitCode) { $finalExitCode = $jobExitCode }
        } else {
            Write-Host "    Completed: $($job.Name)" -ForegroundColor Green
        }
    }
} finally {
    Stop-AllJobs -jobList $jobs
    [Console]::remove_CancelKeyPress($ctrlCHandler)
}

if ($failedJobs.Count -gt 0) {
    Write-Host "==> $($failedJobs.Count) build job(s) failed. Exiting with code $finalExitCode." -ForegroundColor Red
    [System.Environment]::Exit($finalExitCode)
}

Write-Host "==> All model builds completed successfully."

#Remove-Item -Path "./dist" -Recurse -Force
