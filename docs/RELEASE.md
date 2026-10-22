# GitHub / PyPI 发布流程（v0.2.4 起固化）

## 背景：为什么不能直接 `git push`

| | 本地 master | 远程 github/main |
|---|---|---|
| 仓库根 | `D:/datas/share`（多项目混合：easymlops + autohome + trend_striker 等） | EasyMLOps 单项目（项目文件在仓库根） |
| easymlops 位置 | `traes/easymlops-trae/` | 仓库根 `/` |
| 体量 | ~441MB / 48k 对象 | 仅 easymlops 历史 |
| 共同祖先 | 无（历史上发生过目录迁移/重写） | — |

因此**不要**把本地 master 重命名为 main 或设为 github/main 的上游后直接 push：
会尝试上传 441MB 混合历史，且会被 LFS 钩子（历史中 trend_striker 的两个 LFS 指针对象本地已丢失）或非快进检查拒绝；强推则会破坏远程 easymlops 历史。

**正确方式：增量快照推送** —— 基于远程 main 建临时工作树，把 `easymlops-trae/` 当前内容同步进去提交推送。每次只传 easymlops 增量（几 MB），远程保持 easymlops 单项目布局与连续历史。

## 一键发布

```powershell
# 1. 修改 easymlops/__init__.py 的 __version__（脚本会校验一致性）
# 2. 运行发布脚本（可选 -UploadPyPI 同时上传 PyPI，需 ~/.pypirc 凭证）
powershell -File scripts/release_github.ps1 -Version 0.2.5 -UploadPyPI
```

脚本等价于以下手动步骤：

```powershell
# ① 构建
python setup.py sdist bdist_wheel
# ② 基于远程 main 建发布工作树
git fetch github main
git worktree add -b release-0.2.5 "$env:TEMP/easymlops_release" github/main
# ③ 同步内容（/XF .git 保护工作树链接文件；dist/ 随 git 树发布，仅排除 build/egg-info）
robocopy . "$env:TEMP/easymlops_release" /MIR /XD build easymlops.egg-info .codebuddy /XF .git *.pyc
# ④ 提交并推送（快进）
Set-Location "$env:TEMP/easymlops_release"; git add -A; git commit -m "release: v0.2.5"
git push github release-0.2.5:main
# ⑤ tag 重指到发布提交后推送
git tag -f 0.2.5; git push -f github 0.2.5
# ⑥ 清理
Set-Location D:/datas/share; git worktree remove --force "$env:TEMP/easymlops_release"; git worktree prune; git branch -D release-0.2.5
# ⑦ PyPI（可选）
python -m twine upload dist/easymlops-0.2.5*
```

## 分支/tag 绑定关系

- 远程 `github` = `git@github.com:zhulei227/EasyMLOps.git`，唯一发布分支为 **main**（不要向其推送本地 master）。
- 本地 **master**：多项目工作分支，easymlops 的日常开发在此进行（`traes/easymlops-trae/`），不与远程绑定。
- 本地 tag：随发布提交重指（`git tag -f <version> <发布提交>`），远程 tag 以最后一次发布为准。
- `0.2.4` 发布提交：`8f877317`（远程 main 与 tag 0.2.4 均指向它）。

## 已知坑

1. **LFS 缺失对象**：本地历史中 `traes/trend_striker` 的 2 个 LFS 文件对象已丢失（v0.2.4 历史重写时已从历史中移除，见备份分支 `backup-pre-filter`）。若再次遇到 pre-receive hook declined，优先检查是否有新的 LFS 指针进入推送范围。
2. **robocopy /MIR 会删除工作树的 `.git` 链接文件**：必须配 `/XF .git`（文件级排除），且同步后校验 `.git` 存在。
3. **twine 1.6.0**：可用但老旧；若 PyPI 拒绝（API 变更），升级 `pip install -U twine`。
4. **PyPI 版本不可覆盖**：同名版本上传一次后即锁定，需递增版本号。
