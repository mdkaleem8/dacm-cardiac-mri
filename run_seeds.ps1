# Runs after run_followups.ps1: Swin-UNETR zero-conditioning control, then
# two extra seeds (1, 2) of the key U-Net versions for seed-to-seed variation.
$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py    = Join-Path $root ".venv\Scripts\python.exe"
$work  = Join-Path $root "work"
$cache = Join-Path $work "acdc_cache"; $calib = Join-Path $work "calib.json"
$estd  = Join-Path $work "estimator"; $tset = Join-Path $work "testset"; $runs = Join-Path $work "runs"
$paper = Join-Path $root "..\literature_review\results"

while (-not ((Test-Path (Join-Path $work "followups.log")) -and
             (Select-String -Path (Join-Path $work "followups.log") -Pattern "Follow-ups finished" -Quiet))) { Start-Sleep -Seconds 120 }

function Run($name, $backbone, $mode, $extra, $lr, $seed) {
  $out = Join-Path $runs $name
  if (-not (Test-Path (Join-Path $out "done.json"))) {
    Write-Host "===== train $name  $(Get-Date -Format 'HH:mm:ss') ====="
    $a = @("scripts\train_seg.py", "--cache", $cache, "--calib", $calib, "--out", $out, "--backbone", $backbone,
           "--mode", $mode, "--estimator", $estd, "--epochs", "100", "--lr", $lr, "--seed", $seed, "--workers", "6") + $extra
    & $py @a
    if ($LASTEXITCODE -ne 0) { throw "$name training failed" }
  }
  if (-not (Test-Path (Join-Path $out "eval_seg.csv"))) {
    Write-Host "===== eval $name  $(Get-Date -Format 'HH:mm:ss') ====="
    & $py scripts\evaluate.py --cache $cache --testset $tset --estimator $estd --runs $out
    if ($LASTEXITCODE -ne 0) { throw "$name evaluation failed" }
  }
}

Run "swinunetr_dacmzero" "swinunetr" "dacm" @("--placement", "norm", "--zero_cond") "2e-4" "0"
& $py scripts\make_tables.py --runs_root $runs --estimator $estd --paper_results $paper --tag pilot
foreach ($s in 1, 2) {
  Run "unet_aug_s$s"      "unet" "aug"  @()                                    "1e-3" "$s"
  Run "unet_dacmnorm_s$s" "unet" "dacm" @("--placement", "norm")               "1e-3" "$s"
  Run "unet_dacmzero_s$s" "unet" "dacm" @("--placement", "norm", "--zero_cond") "1e-3" "$s"
}
Write-Host "Seeds finished $(Get-Date -Format 'HH:mm:ss')"
