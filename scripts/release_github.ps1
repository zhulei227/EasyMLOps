# -*- coding: utf-8 -*-
# EasyMLOps GitHub 发布脚本(固化 v0.2.4 验证过的推送逻辑)
#
# 背景:
#   本地 git 仓库根是 D:/datas/share(多项目混合仓库, 本地 master 含全部项目历史),
#   而远程 github(=git@github.com:zhulei227/EasyMLOps.git) 的 main 分支是
#   "easymlops 单项目"布局(项目文件在仓库根)。两者没有共同祖先,
#   直接 push master 会尝试上传 441MB 混合历史并被拒(LFS 缺失/非快进)。
#
#   因此发布采用"增量快照"方式: 基于远程 main 建临时工作树,
#   同步 easymlops-trae 内容后提交推送 —— 每次只传 easymlops 增量(几 MB)。
#
# 用法:
#   1. 修改 easymlops/__init__.py 的 __version__
#   2. powershell -File scripts/release_github.ps1 -Version 0.2.5 [-UploadPyPI]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [switch]$UploadPyPI
)
$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot          # easymlops-trae 根
$wt = Join-Path $env:TEMP "easymlops_release"

# ---- 0. 前置校验: 版本号一致 ----
$init = Get-Content (Join-Path $projectDir "easymlops/__init__.py") -Raw
if ($init -notmatch [regex]::Escape("__version__ = `"$Version`"")) {
    throw "easymlops/__init__.py 中 __version__ 与 -Version $Version 不一致, 请先修改版本号"
}

# ---- 1. 构建 sdist/wheel ----
Set-Location $projectDir
python setup.py sdist bdist_wheel
if ($LASTEXITCODE -ne 0) { throw "build failed" }

# ---- 2. 基于远程 main 建发布工作树 ----
Set-Location $projectDir
git fetch github main
git worktree prune
if (Test-Path $wt) { Remove-Item -Recurse -Force $wt }
git branch -D "release-$Version" 2>$null | Out-Null
git worktree add -b "release-$Version" $wt github/main
if (-not (Test-Path (Join-Path $wt ".git"))) { throw "worktree .git link missing" }

# ---- 3. 同步项目内容(dist/ 随 git 树发布; 排除本地杂项; /XF .git 保护工作树链接) ----
robocopy $projectDir $wt /MIR `
    /XD .git build easymlops.egg-info .codebuddy .idea __pycache__ .pytest_cache `
    /XF .git "*.pyc" /NFL /NDL /NJH | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with code $LASTEXITCODE" }
if (-not (Test-Path (Join-Path $wt ".git"))) { throw "robocopy deleted .git link" }

# ---- 4. 提交并推送 main(应为快进; 若远程有变动需人工介入) ----
Set-Location $wt
git add -A
git commit -m "release: v$Version"
git push github "release-${Version}:main"
if ($LASTEXITCODE -ne 0) { throw "push main failed" }

# ---- 5. tag 重指到发布提交并推送 ----
$commit = git rev-parse HEAD
git tag -f $Version $commit | Out-Null
git push -f github $Version
if ($LASTEXITCODE -ne 0) { throw "push tag failed" }

# ---- 6. 清理工作树与临时分支 ----
Set-Location $projectDir
git worktree remove --force $wt
git worktree prune
git branch -D "release-$Version" | Out-Null

# ---- 7. 可选: 上传 PyPI ----
if ($UploadPyPI) {
    Set-Location $projectDir
    python -m twine upload "dist/easymlops-$Version-py3-none-any.whl" "dist/easymlops-$Version.tar.gz"
}

Write-Host "`nrelease v$Version done: main -> $(git rev-parse --short HEAD), tag $Version" -ForegroundColor Green
