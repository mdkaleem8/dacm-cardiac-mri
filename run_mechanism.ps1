# Experiments on why conditioning adds little (all seeds 0, 1, 2):
#  A. conditioning shuffled across the batch during training (U-Net, then Swin-UNETR)
#  B. reduced training set: 4 subjects per group (20 of 80), 300 epochs (U-Net)
#  C. training severity capped at 0.5, tested up to 1.0 (U-Net)
$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py    = Join-Path $root ".venv\Scripts\python.exe"
$work  = Join-Path $root "work"
$cache = Join-Path $work "acdc_cache"; $calib = Join-Path $work "calib.json"
$estd  = Join-Path $work "estimator"; $tset = Join-Path $work "testset"; $runs = Join-Path $work "runs"
$paper = Join-Path $root "..\literature_review\results"
$SEEDS = @(@("", "0"), @("_s1", "1"), @("_s2", "2"))

function Run($name, $backbone, $mode, $extra, $lr, $seed, $epochs, $e2) {
  $out = Join-Path $runs $name
  if (-not (Test-Path (Join-Path $out "done.json"))) {
    Write-Host "===== train $name  $(Get-Date -Format 'HH:mm:ss') ====="
    $a = @("scripts\train_seg.py", "--cache", $cache, "--calib", $calib, "--out", $out, "--backbone", $backbone,
           "--mode", $mode, "--estimator", $estd, "--epochs", $epochs, "--lr", $lr, "--seed", $seed, "--workers", "6") + $extra
    & $py @a
    if ($LASTEXITCODE -ne 0) { throw "$name training failed" }
  }
  if (-not (Test-Path (Join-Path $out "eval_seg.csv"))) {
    Write-Host "===== eval $name  $(Get-Date -Format 'HH:mm:ss') ====="
    $ea = @("scripts\evaluate.py", "--cache", $cache, "--testset", $tset, "--estimator", $estd, "--runs", $out)
    if ($e2) { $ea += "--e2" }
    & $py @ea
    if ($LASTEXITCODE -ne 0) { throw "$name evaluation failed" }
  }
}

$norm = @("--placement", "norm")
$zero = @("--placement", "norm", "--zero_cond")
$shuf = @("--placement", "norm", "--shuffle_cond")

# A. shuffled conditioning during training, U-Net
foreach ($s in $SEEDS) { Run "unet_dacmshuf$($s[0])" "unet" "dacm" $shuf "1e-3" $s[1] "100" ($s[1] -eq "0") }
& $py scripts\seed_summary.py --runs_root $runs --paper_results $paper --backbone unet --versions aug dacmnorm dacmzero dacmshuf

# B. reduced training set
$low = @("--train_per_group", "4")
foreach ($s in $SEEDS) {
  Run "unet_aug_low$($s[0])"      "unet" "aug"  $low            "1e-3" $s[1] "300" $false
  Run "unet_dacmnorm_low$($s[0])" "unet" "dacm" ($norm + $low)  "1e-3" $s[1] "300" $false
  Run "unet_dacmzero_low$($s[0])" "unet" "dacm" ($zero + $low)  "1e-3" $s[1] "300" $false
}
& $py scripts\seed_summary.py --runs_root $runs --paper_results $paper --backbone unet --tag _low

# C. capped training severity
$cap = @("--max_sev", "0.5")
foreach ($s in $SEEDS) {
  Run "unet_aug_cap$($s[0])"      "unet" "aug"  $cap            "1e-3" $s[1] "100" $false
  Run "unet_dacmnorm_cap$($s[0])" "unet" "dacm" ($norm + $cap)  "1e-3" $s[1] "100" $false
  Run "unet_dacmzero_cap$($s[0])" "unet" "dacm" ($zero + $cap)  "1e-3" $s[1] "100" $false
}
& $py scripts\seed_summary.py --runs_root $runs --paper_results $paper --backbone unet --tag _cap

# A. shuffled conditioning during training, Swin-UNETR
foreach ($s in $SEEDS) { Run "swinunetr_dacmshuf$($s[0])" "swinunetr" "dacm" $shuf "2e-4" $s[1] "100" ($s[1] -eq "0") }
& $py scripts\seed_summary.py --runs_root $runs --paper_results $paper --backbone swinunetr --versions aug dacmnorm dacmzero dacmshuf
Write-Host "Mechanism experiments finished $(Get-Date -Format 'HH:mm:ss')"
