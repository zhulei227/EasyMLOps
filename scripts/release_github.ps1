# -*- coding: utf-8 -*-
# EasyMLOps GitHub/PyPI 发布脚本（v0.2.4 起简化版）
#
# 背景: 本仓库(easymlops-trae)自 v0.2.4 起为独立 git 仓库, 本地 main 分支
#       直接跟踪远程 github/main(git@github.com:zhulei227/EasyMLOps.git),
#       发布即普通 git push(历史连续, 无需任何快照/工作树操作)。
#
# 用法:
#   1. 修改 easymlops/__init__.py 的 __version__
#   2. powershell -ExecutionPolicy Bypass -File scripts/release_github.ps1 -Version 0.2.6 [-UploadPyPI] [-SkipPush]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [switch]$UploadPyPI,
    [switch]$SkipPush
)
$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot

# ---- 0. 前置校验: 版本号一致 ----
$init = Get-Content (Join-Path $projectDir "easymlops/__init__.py") -Raw
$expected = '__version__ = "{0}"' -f $Version
if ($init.Contains($expected) -eq $false) {
    throw "easymlops/__init__.py 中 __version__ 与 -Version $Version 不一致, 请先修改版本号"
}

# ---- 1. 构建 sdist/wheel ----
Set-Location $projectDir
python setup.py sdist bdist_wheel
if ($LASTEXITCODE -ne 0) { throw "build failed" }

# ---- 2. 提交并推送(普通快进 push) ----
if (-not $SkipPush) {
    git add -A
    git commit -m "release: v$Version" 2>$null
    git push github main
    if ($LASTEXITCODE -ne 0) { throw "push main failed" }
    # tag 重指到当前提交并推送
    $commit = git rev-parse HEAD
    git tag -f $Version $commit | Out-Null
    git push -f github $Version
    if ($LASTEXITCODE -ne 0) { throw "push tag failed" }
}

# ---- 3. 可选: 上传 PyPI(需 ~/.pypirc 凭证) ----
if ($UploadPyPI) {
    python -m twine upload "dist/easymlops-$Version-py3-none-any.whl" "dist/easymlops-$Version.tar.gz"
}

Write-Host ""
Write-Host "release v$Version done" -ForegroundColor Green
