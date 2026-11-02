# Revision runs, U-Net queue.
#  1. Estimators retrained under each stress-test restriction (20 subjects; severity cap 0.5),
#     so that the conditioned model gets no training data the controls did not see.
#  2. DACM (norm) with the restricted estimator ("dacmnormr"), five seeds per stress test.
#  3. Seeds 3 and 4 of the main U-Net versions and of the stress-test controls (five seeds in total).
$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py    = Join-Path $root ".venv\Scripts\python.exe"
$work  = Join-Path $root "work"
$cache = Join-Path $work "acdc_cache"; $calib = Join-Path $work "calib.json"
$estd  = Join-Path $work "estimator"; $tset = Join-Path $work "testset"; $runs = Join-Path $work "runs"
$estLow = Join-Path $work "estimator_low"; $estCap = Join-Path $work "estimator_cap"
$SEEDS = @(@("", "0"), @("_s1", "1"), @("_s2", "2"), @("_s3", "3"), @("_s4", "4"))
$NEW   = @(@("_s3", "3"), @("_s4", "4"))

function Est($out, $extra) {
  if (-not (Test-Path (Join-Path $out "estimator_log.json"))) {
    Write-Host "===== estimator $out  $(Get-Date -Format 'HH:mm:ss') ====="
    $a = @("scripts\train_estimator.py", "--cache", $cache, "--calib", $calib, "--out", $out,
           "--epochs", "100", "--workers", "6") + $extra
    & $py @a
    if ($LASTEXITCODE -ne 0) { throw "estimator $out failed" }
  }
}

function Run($name, $mode, $extra, $seed, $epochs, $est, $e2) {
  $out = Join-Path $runs $name
  if (-not (Test-Path (Join-Path $out "done.json"))) {
    Write-Host "===== train $name  $(Get-Date -Format 'HH:mm:ss') ====="
    $a = @("scripts\train_seg.py", "--cache", $cache, "--calib", $calib, "--out", $out, "--backbone", "unet",
           "--mode", $mode, "--estimator", $est, "--epochs", $epochs, "--lr", "1e-3", "--seed", $seed, "--workers", "6") + $extra
    & $py @a
    if ($LASTEXITCODE -ne 0) { throw "$name training failed" }
  }
  if (-not (Test-Path (Join-Path $out "eval_seg.csv"))) {
    Write-Host "===== eval $name  $(Get-Date -Format 'HH:mm:ss') ====="
    $ea = @("scripts\evaluate.py", "--cache", $cache, "--testset", $tset, "--estimator", $est, "--runs", $out)
    if ($e2) { $ea += "--e2" }
    & $py @ea
    if ($LASTEXITCODE -ne 0) { throw "$name evaluation failed" }
  }
}

$norm = @("--placement", "norm")
$zero = @("--placement", "norm", "--zero_cond")
$shuf = @("--placement", "norm", "--shuffle_cond")
$low  = @("--train_per_group", "4")
$cap  = @("--max_sev", "0.5")

# 1. restricted estimators
Est $estLow $low
Est $estCap $cap

# 2. DACM (norm) with the restricted estimator, five seeds
foreach ($s in $SEEDS) { Run "unet_dacmnormr_low$($s[0])" "dacm" ($norm + $low) $s[1] "300" $estLow ($s[1] -eq "0") }
foreach ($s in $SEEDS) { Run "unet_dacmnormr_cap$($s[0])" "dacm" ($norm + $cap) $s[1] "100" $estCap ($s[1] -eq "0") }

# 3. seeds 3 and 4: main versions, then stress-test controls
foreach ($s in $NEW) {
  Run "unet_aug$($s[0])"      "aug"  @() $s[1] "100" $estd $false
  Run "unet_dacmnorm$($s[0])" "dacm" $norm $s[1] "100" $estd $false
  Run "unet_dacmzero$($s[0])" "dacm" $zero $s[1] "100" $estd $false
  Run "unet_dacmshuf$($s[0])" "dacm" $shuf $s[1] "100" $estd $false
}
foreach ($s in $NEW) {
  Run "unet_aug_low$($s[0])"      "aug"  $low            $s[1] "300" $estd $false
  Run "unet_dacmzero_low$($s[0])" "dacm" ($zero + $low)  $s[1] "300" $estd $false
  Run "unet_aug_cap$($s[0])"      "aug"  $cap            $s[1] "100" $estd $false
  Run "unet_dacmzero_cap$($s[0])" "dacm" ($zero + $cap)  $s[1] "100" $estd $false
}
Write-Host "U-Net revision queue finished $(Get-Date -Format 'HH:mm:ss')"
