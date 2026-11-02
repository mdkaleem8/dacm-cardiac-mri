# Full DACM pipeline on ACDC. Run from the dacm folder:
#   powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 [-Epochs 100] [-EstEpochs 40] [-Backbones unet,swinunetr]
param(
  [int]$Epochs = 100,
  [int]$EstEpochs = 40,
  [string[]]$Backbones = @("unet", "swinunetr"),
  [string]$Tag = "pilot"
)
$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $MyInvocation.MyCommand.Path
$py    = Join-Path $root ".venv\Scripts\python.exe"
$acdc  = Join-Path $root "..\data\ACDC\ACDC\database"
$work  = Join-Path $root "work"
$cache = Join-Path $work "acdc_cache"
$calib = Join-Path $work "calib.json"
$estd  = Join-Path $work "estimator"
$tset  = Join-Path $work "testset"
$runs  = Join-Path $work "runs"
$paper = Join-Path $root "..\literature_review\results"
New-Item -ItemType Directory -Force $work, $runs | Out-Null

function Step($name, [scriptblock]$cmd) {
  Write-Host "`n===== $name  $(Get-Date -Format 'HH:mm:ss') ====="
  & $cmd
  if ($LASTEXITCODE -ne 0) { throw "$name failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path (Join-Path $cache "splits.json"))) { Step "prepare" { & $py scripts\prepare_acdc.py --acdc $acdc --out $cache } }
if (-not (Test-Path $calib)) { Step "calibrate" { & $py scripts\calibrate.py --cache $cache --out $calib } }
if (-not (Test-Path (Join-Path $estd "estimator.pt"))) { Step "estimator" { & $py scripts\train_estimator.py --cache $cache --calib $calib --out $estd --epochs $EstEpochs } }
if (-not (Test-Path (Join-Path $tset "conditions.json"))) { Step "testset" { & $py scripts\build_testset.py --cache $cache --calib $calib --out $tset --combos } }

foreach ($b in $Backbones) {
  $lr = if ($b -eq "swinunetr") { "2e-4" } else { "1e-3" }
  foreach ($m in @("default", "aug", "dacm")) {
    $out = Join-Path $runs "${b}_${m}"
    if (Test-Path (Join-Path $out "done.json")) { continue }
    $extra = if ($m -eq "dacm") { @("--estimator", $estd) } else { @() }
    Step "train $b $m" { & $py scripts\train_seg.py --cache $cache --calib $calib --out $out --backbone $b --mode $m --epochs $Epochs --lr $lr @extra }
  }
  $runDirs = @("default", "aug", "dacm") | ForEach-Object { Join-Path $runs "${b}_$_" } | Where-Object { -not (Test-Path (Join-Path $_ "eval_seg.csv")) }
  if ($runDirs) { Step "evaluate $b" { & $py scripts\evaluate.py --cache $cache --testset $tset --estimator $estd --e2 --runs @runDirs } }
  Step "tables" { & $py scripts\make_tables.py --runs_root $runs --estimator $estd --paper_results $paper --tag $Tag }
}
Write-Host "`nPipeline finished $(Get-Date -Format 'HH:mm:ss')"
