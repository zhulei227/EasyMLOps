# GitHub / PyPI 发布流程

> **v0.2.5 起本目录(easymlops-trae)为独立 git 仓库**，本地 `main` 分支直接跟踪
> 远程 `github/main`（`git@github.com:zhulei227/EasyMLOps.git`），发布即普通 `git push`。
> （历史方案——基于远程 main 的快照工作树同步——已废弃，仅 v0.2.4 发布时使用过，
> 当时本地仓库是多项目混合结构且与远程历史不连续。）

## 一键发布

```powershell
# 1. 修改 easymlops/__init__.py 的 __version__（脚本会校验一致性）
# 2. 运行发布脚本（-UploadPyPI 同时上传 PyPI，需 ~/.pypirc 凭证；-SkipPush 只构建不推送）
powershell -File scripts/release_github.ps1 -Version 0.2.5 -UploadPyPI
```

等价手动步骤：

```powershell
# ① 构建
python setup.py sdist bdist_wheel
# ② 提交推送（普通快进 push，本地 main 已绑定远程）
git add -A; git commit -m "release: v0.2.5"; git push github main
# ③ tag 重指到发布提交后推送
git tag -f 0.2.5; git push -f github 0.2.5
# ④ PyPI（可选）
python -m twine upload dist/easymlops-0.2.5*
```

## 绑定关系

| 引用 | 说明 |
|------|------|
| 本地 `main`（easymlops-trae 仓库） | 开发与发布分支，`git branch -u github/main main` 已设上游，直接 `git push` / `git pull` |
| 远程 `github/main` | EasyMLOps 唯一发布分支（easymlops 单项目布局，项目文件在仓库根） |
| 远程 tag `<version>` | 随发布重指到发布提交（`git tag -f` + `git push -f`） |
| `dist/` | 随 git 树发布（含当期 wheel/sdist）；发布前建议清理旧版本安装包，只保留当前版本 |

## 已知坑

1. **本地环境**：`dask` 必须 >= 2023.3.0（requirements 已声明）；旧版 dask(2021.x) 与 pandas 2.x 不兼容，会导致 `import lightgbm` 报 `pandas.core.strings has no attribute StringMethods`。
2. **git 版本较老**（< 2.28）：`git init -b main` 不可用，需 `git init` + `git symbolic-ref HEAD refs/heads/main`。
3. **PyPI 版本不可覆盖**：同名版本上传一次后即锁定，需递增版本号。
4. **twine 1.6.0**：可用但老旧；若 PyPI 拒绝（API 变更），`pip install -U twine`。
5. **kaggle_solutions/**：本地私有语料目录，已加入 `.gitignore`，不随仓库发布。
6. **.ps1 必须带 UTF-8 BOM**：Windows PowerShell 5.1 对无 BOM 文件按系统 ANSI(GBK) 解码，中文注释的字节序列会破坏字符串解析（报"字符串缺少终止符"）。`scripts/release_github.ps1` 已带 BOM，编辑后需保持。
