# Revision runs, Swin-UNETR queue: seeds 3 and 4 of the four key versions (five seeds in total).
$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py    = Join-Path $root ".venv\Scripts\python.exe"
$work  = Join-Path $root "work"
$cache = Join-Path $work "acdc_cache"; $calib = Join-Path $work "calib.json"
$estd  = Join-Path $work "estimator"; $tset = Join-Path $work "testset"; $runs = Join-Path $work "runs"

function Run($name, $mode, $extra, $seed) {
  $out = Join-Path $runs $name
  if (-not (Test-Path (Join-Path $out "done.json"))) {
    Write-Host "===== train $name  $(Get-Date -Format 'HH:mm:ss') ====="
    $a = @("scripts\train_seg.py", "--cache", $cache, "--calib", $calib, "--out", $out, "--backbone", "swinunetr",
           "--mode", $mode, "--estimator", $estd, "--epochs", "100", "--lr", "2e-4", "--seed", $seed, "--workers", "6") + $extra
    & $py @a
    if ($LASTEXITCODE -ne 0) { throw "$name training failed" }
  }
  if (-not (Test-Path (Join-Path $out "eval_seg.csv"))) {
    Write-Host "===== eval $name  $(Get-Date -Format 'HH:mm:ss') ====="
    & $py scripts\evaluate.py --cache $cache --testset $tset --estimator $estd --runs $out
    if ($LASTEXITCODE -ne 0) { throw "$name evaluation failed" }
  }
}

foreach ($s in 3, 4) {
  Run "swinunetr_aug_s$s"      "aug"  @()                                       "$s"
  Run "swinunetr_dacmnorm_s$s" "dacm" @("--placement", "norm")                  "$s"
  Run "swinunetr_dacmzero_s$s" "dacm" @("--placement", "norm", "--zero_cond")   "$s"
  Run "swinunetr_dacmshuf_s$s" "dacm" @("--placement", "norm", "--shuffle_cond") "$s"
}
Write-Host "Swin revision queue finished $(Get-Date -Format 'HH:mm:ss')"
