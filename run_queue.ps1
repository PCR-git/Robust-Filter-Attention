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
# Done 2026-10-18: wikitext103 B1, M2. M2.7_rope_spec was cut off by a Windows Update reboot after epoch 8.
$runs = @(
    @("wikitext103", "M2.7", "M2.7_rope_spec", "--set use_SC_RoPE=False --resume C:\rfa_runs\wikitext103\M2.7_rope_spec__2026-10-18\resume.pt"),
    @("wikitext103", "M2.7", "M2.7",          ""),
    @("wikitext103", "B3",   "B3",            ""),
    @("wikitext103", "B4",   "B4",            ""),
    @("wikitext103", "M1",   "M1",            ""),
    @("wikitext103", "M3",   "M3",            ""),
    @("wikitext103", "M2.1", "M2.1",          ""),
    @("wikitext103", "M2.2", "M2.2",          ""),
    @("wikitext103", "M2.3", "M2.3",          ""),
    @("wikitext103", "M2.4", "M2.4",          ""),
    @("wikitext103", "M2.5", "M2.5",          ""),
    @("wikitext103", "M2.6", "M2.6",          ""),
    @("babylm",      "B1",   "B1",            ""),
    @("babylm",      "B3",   "B3",            ""),
    @("babylm",      "B4",   "B4",            "")
)

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
}
Write-Host "Queue finished."
