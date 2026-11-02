# Runs requested by the internal IEEE Access review (2026-10-30). One GPU job at a time.
# Most important first, so the key results exist even if a later step fails.
#  A1  Swin-UNETR augmentation-only with affine instance norms (control for the z=0 module), 5 seeds
#  A2  E2 conditioning variants (oracle, noisy, shuffled) for seeds 1-4 of DACM (norm) and shuffled-training
#  A3  U-Net DACM (norm) conditioned on type only, 5 seeds
#  B1  Default version, seeds 1-4, both backbones
#  B2  Linear probes, seeds 1-4
#  B3  DACM (stage), seeds 1-4, both backbones
#  B4  Slice position, seeds 1-4
#  B5  CMRxMotion and M&Ms for the new runs
# Every step skips work that is already done, so the script can simply be restarted.
$root  = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py    = Join-Path $root ".venv\Scripts\python.exe"
$work  = Join-Path $root "work"
$cache = Join-Path $work "acdc_cache"; $calib = Join-Path $work "calib.json"
$estd  = Join-Path $work "estimator"; $tset = Join-Path $work "testset"; $vset = Join-Path $work "valset"
$runs  = Join-Path $work "runs"
$env:PYTHONPATH = $root

function Step($label, [scriptblock]$body) {
  Write-Host "===== $label  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
  try { & $body; if ($LASTEXITCODE -ne 0) { Write-Host "!!!!! $label exited with code $LASTEXITCODE" } }
  catch { Write-Host "!!!!! $label failed: $_" }
}

function Train($name, $backbone, $mode, $extra, $seed) {
  $out = Join-Path $runs $name
  $lr = if ($backbone -eq "unet") { "1e-3" } else { "2e-4" }
  Step "train $name" {
    if (-not (Test-Path (Join-Path $out "done.json"))) {
      $a = @("-u", "scripts\train_seg.py", "--cache", $cache, "--calib", $calib, "--out", $out, "--backbone", $backbone,
             "--mode", $mode, "--estimator", $estd, "--epochs", "100", "--lr", $lr, "--seed", "$seed", "--workers", "6") + $extra
      & $py @a
    }
  }
  Step "eval $name" {
    if ((Test-Path (Join-Path $out "done.json")) -and -not (Test-Path (Join-Path $out "eval_seg.csv"))) {
      & $py -u scripts\evaluate.py --cache $cache --testset $tset --estimator $estd --runs $out
    }
  }
}
function Sfx($s) { if ($s -eq 0) { "" } else { "_s$s" } }

# ---- A1: Swin-UNETR augmentation-only with affine instance norms
foreach ($s in 0..4) { Train "swinunetr_augaff$(Sfx $s)" "swinunetr" "aug" @("--norm_affine") $s }

# ---- A2: E2 variants at severity 1.0 for seeds 1-4 (seed 0 already has them)
$lvl = @("motion_1.0", "noise_1.0", "blur_1.0", "bias_1.0", "undersample_1.0")
foreach ($b in "unet", "swinunetr") { foreach ($v in "dacmnorm", "dacmshuf") { foreach ($s in 1..4) {
  $out = Join-Path $runs "$($b)_$v$(Sfx $s)"
  Step "E2 $($b)_$v$(Sfx $s)" {
    if (-not (Test-Path (Join-Path $out "eval_seg_e2.csv"))) {
      & $py -u scripts\evaluate.py --cache $cache --testset $tset --estimator $estd --runs $out --e2 --conditions @lvl --tag _e2
    }
  }
}}}

# ---- A3: U-Net, conditioning on degradation type only
foreach ($s in 0..4) { Train "unet_dacmtype$(Sfx $s)" "unet" "dacm" @("--placement", "norm", "--cond_part", "type") $s }

# ---- B1: Default, seeds 1-4
foreach ($b in "unet", "swinunetr") { foreach ($s in 1..4) { Train "$($b)_default$(Sfx $s)" $b "default" @() $s } }

# ---- B2: probes for seeds 1-4 (outputs kept out of the paper folder; aggregated by the stats script)
foreach ($s in 1..4) {
  $pdir = Join-Path $work "probe_seeds\s$s"
  Step "probes s$s" {
    if (-not (Test-Path (Join-Path $pdir "numbers_probe.tex"))) {
      & $py -u scripts\probe_val.py --valset $vset --testset $tset --estimator $estd --runs_root $runs --paper_results $pdir --suffix "_s$s"
    }
  }
}

# ---- B3: DACM (stage), seeds 1-4
foreach ($b in "unet", "swinunetr") { foreach ($s in 1..4) { Train "$($b)_dacm$(Sfx $s)" $b "dacm" @("--placement", "stage") $s } }

# ---- B4: slice position, seeds 1-4 (writes slices.csv into each run folder)
foreach ($s in 1..4) {
  Step "slices s$s" {
    & $py -u scripts\eval_slices.py --cache $cache --testset $tset --estimator $estd --runs_root $runs `
        --paper_results (Join-Path $work "slices_seeds\s$s") --suffix "_s$s"
  }
}

# ---- B5: external test sets for the new runs (existing outputs are skipped by the scripts)
$new = @()
foreach ($s in 0..4) { $new += "work\runs\swinunetr_augaff$(Sfx $s)"; $new += "work\runs\unet_dacmtype$(Sfx $s)" }
foreach ($b in "unet", "swinunetr") { foreach ($s in 1..4) { $new += "work\runs\$($b)_default_s$s" } }
Step "external CMRxMotion" {
  & $py -u scripts\eval_cmrx.py --data_root ..\data\CMRxMotion\CMRxMotion_Training --cache work\cmrx_cache --estimator $estd --out work\cmrx_eval --runs $new
}
Step "external M&Ms" {
  & $py -u scripts\eval_mnms.py --data_root ..\data\MnMs\OpenDataset --cache work\mnms_cache --estimator $estd --out work\mnms_eval --runs $new
}
Write-Host "Review queue finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
