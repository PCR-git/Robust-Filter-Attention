# Run a list of training jobs one after another (one GPU, one job at a time).
#
#   powershell -ExecutionPolicy Bypass -File run_queue.ps1
#
# Each run logs to $outDir\logs\<name>.log (stdout), <name>.err (stderr: progress bars, warnings)
# and <name>.gpu.csv (GPU clocks, temperature, power and throttle reasons, once a minute).
# Results (metrics.txt, plots, checkpoints) go to $outDir\<dataset>\<name>__<date>\ (outside OneDrive).
# Before each run, waits until the GPU is nearly free (e.g. until notebook kernels release their memory).
# Close the window to stop the queue.
# -Skip name1,name2 skips runs by name (e.g. ones already finished).
# To continue an interrupted run, add --resume <run dir>\resume.pt (plus its original --set overrides) to its arguments.

param([string[]]$Skip = @())

$python = "C:\Users\pcrac\miniconda3\envs\ai\python.exe"
$repo = $PSScriptRoot
$outDir = "C:\rfa_runs"  # outside OneDrive

# dataset, preset, name, extra arguments to train_lm.py (defaults match the published runs)
# Run names below use the numbering before 2026-10-28: run "M2.7" is now preset M2.8 and vice versa.
# Done 2026-10-18/19: wikitext103 B1, M2, M2.7_rope_spec, M2.7, B3, B4. M1 was stopped after epoch 6 on 2026-10-19.
# Done 2026-10-19/21: babylm B1; wikitext103 M1 (resumed), M3, M2.1-M2.6. Done 2026-10-23: wikitext103 M2.8.
# 2026-10-28: B2 rerun for its per-epoch validation curve (the published history is lost).
$runs = @(
    ,@("wikitext103", "B2",   "B2",            "")  # leading comma: keeps a one-run list from being flattened
)
$evalAfter = $true  # run eval_extrapolation.py on each finished run

$logDir = Join-Path $outDir "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

function Wait-ForFreeGpu([int]$maxUsedMiB = 2000) {
    while ($true) {
        $used = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
        if ($used -lt $maxUsedMiB) { return }
        Write-Host "$(Get-Date -Format 'HH:mm:ss')  GPU memory in use: $used MiB; waiting for it to be freed..."
        Start-Sleep -Seconds 60
    }
}

$gpuQuery = "timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used,clocks_throttle_reasons.active"

foreach ($run in $runs) {
    $dataset, $preset, $runName, $extra = $run
    $name = "${dataset}_$runName"
    if ($Skip -contains $name) { Write-Host "skip  $name"; continue }
    Wait-ForFreeGpu
    $runArgs = "--dataset $dataset --preset $preset --name $runName --out-dir $outDir --compile $extra".Trim()
    $logName = if ($extra -match "--resume") { "${name}_resumed" } else { $name }  # keep the interrupted run's logs
    $log = Join-Path $logDir "$logName.log"
    $err = Join-Path $logDir "$logName.err"
    $gpuLog = Join-Path $logDir "$logName.gpu.csv"
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  START $name  ($runArgs)"
    $p = Start-Process -FilePath $python -ArgumentList "-u train_lm.py $runArgs" -WorkingDirectory $repo `
         -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
    $null = $p.Handle  # keeps the exit code available after the process ends
    & nvidia-smi --query-gpu=$gpuQuery --format=csv | Set-Content $gpuLog
    while (-not $p.WaitForExit(60000)) {
        & nvidia-smi --query-gpu=$gpuQuery --format=csv,noheader | Add-Content $gpuLog
    }
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  END   $name  (exit code $($p.ExitCode))"
    Get-Content $log | Select-String -Pattern "^Epoch|^Test" | Select-Object -Last 2 | ForEach-Object { Write-Host "        $_" }
    if ($evalAfter -and $p.ExitCode -eq 0) {
        $runDir = Get-ChildItem (Join-Path $outDir $dataset) -Directory -Filter "${runName}__*" |
                  Sort-Object LastWriteTime | Select-Object -Last 1
        Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  EVAL  $name  ($($runDir.FullName))"
        $evalLog = Join-Path $logDir "$name.extrapolation.log"
        & $python -u (Join-Path $repo "eval_extrapolation.py") --dataset $dataset --data-dir (Join-Path $repo "datasets") $runDir.FullName *> $evalLog
        Get-Content $evalLog | Select-Object -Last 6 | ForEach-Object { Write-Host "        $_" }
    }
}
Write-Host "Queue finished."
