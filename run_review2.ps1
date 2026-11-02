# Follow-up to run_review.ps1: E2 conditioning variants for the new DACM (stage) seeds 1-4.
# Waits until run_review.ps1 has finished, so that only one GPU job runs at a time.
$root  = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py    = Join-Path $root ".venv\Scripts\python.exe"
$work  = Join-Path $root "work"
$cache = Join-Path $work "acdc_cache"; $estd = Join-Path $work "estimator"; $tset = Join-Path $work "testset"
$runs  = Join-Path $work "runs"
$env:PYTHONPATH = $root
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_review\.ps1' }) { Start-Sleep -Seconds 120 }
$lvl = @("motion_1.0", "noise_1.0", "blur_1.0", "bias_1.0", "undersample_1.0")
foreach ($b in "unet", "swinunetr") { foreach ($s in 1..4) {
  $out = Join-Path $runs "$($b)_dacm_s$s"
  Write-Host "===== E2 $($b)_dacm_s$s  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
  if ((Test-Path (Join-Path $out "model.pt")) -and -not (Test-Path (Join-Path $out "eval_seg_e2.csv"))) {
    & $py -u scripts\evaluate.py --cache $cache --testset $tset --estimator $estd --runs $out --e2 --conditions @lvl --tag _e2
    if ($LASTEXITCODE -ne 0) { Write-Host "!!!!! E2 $($b)_dacm_s$s exited with code $LASTEXITCODE" }
  }
}}
Write-Host "Review follow-up finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
