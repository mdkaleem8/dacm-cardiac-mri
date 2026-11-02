# Runs after run_pipeline.ps1 has finished (waits for it):
# Swin-UNETR DACM (norm placement), slice-position analysis, and table refresh.
$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py    = Join-Path $root ".venv\Scripts\python.exe"
$work  = Join-Path $root "work"
$cache = Join-Path $work "acdc_cache"; $calib = Join-Path $work "calib.json"
$estd  = Join-Path $work "estimator"; $tset = Join-Path $work "testset"; $runs = Join-Path $work "runs"
$paper = Join-Path $root "..\literature_review\results"

# Wait until the main pipeline has evaluated Swin-UNETR (stage) and written tables.
while (-not (Select-String -Path (Join-Path $work "pipeline.log") -Pattern "Pipeline finished" -Quiet)) { Start-Sleep -Seconds 120 }
# Wait for the U-Net zero-conditioning control to finish its evaluation.
while (-not (Test-Path (Join-Path $runs "unet_dacmzero\eval_seg.csv"))) { Start-Sleep -Seconds 120 }

$out = Join-Path $runs "swinunetr_dacmnorm"
if (-not (Test-Path (Join-Path $out "done.json"))) {
  & $py scripts\train_seg.py --cache $cache --calib $calib --out $out --backbone swinunetr --mode dacm --placement norm --estimator $estd --epochs 100 --lr 2e-4
  if ($LASTEXITCODE -ne 0) { throw "swin dacmnorm training failed" }
}
if (-not (Test-Path (Join-Path $out "eval_seg.csv"))) {
  & $py scripts\evaluate.py --cache $cache --testset $tset --estimator $estd --e2 --runs $out
  if ($LASTEXITCODE -ne 0) { throw "swin dacmnorm evaluation failed" }
}
& $py scripts\eval_slices.py --cache $cache --testset $tset --estimator $estd --runs_root $runs --paper_results $paper
& $py scripts\make_tables.py --runs_root $runs --estimator $estd --paper_results $paper --tag pilot
Write-Host "Follow-ups finished $(Get-Date -Format 'HH:mm:ss')"
