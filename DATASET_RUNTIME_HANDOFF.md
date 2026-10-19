# Sage 行人跟踪数据集运行源码交接

本分支 `dataset-runtime-20261019` 用于另一台机器运行新数据集 STT/DT 工具，源于 m5 的 FLUX `8393c17d96565162804e4a5c0d9df13242244634`。

公开 `main` 缺少当前所需的 `sage_utils/`，公开 `sage3d` 虽有该目录，但两个关键文件的内容与 m5 不同。m5 的8393c17当时不能从GitHub取到。本分支从公开sage3d建立，覆盖 **1389份来自m5的原始文本源码/配置**，包括sage_utils及其episode_caches，逐文件SHA在 `M5_SOURCE_MANIFEST.json`。其它既有上游文件保留，少量原始开发小报告按来源保留而不改其内容；不覆盖公共main/dev。当前Git提交是本次源码快照的新提交，不应要求 `git rev-parse HEAD` 等于原m5提交8393c17。

## 获取与校验

```bash
git clone --single-branch --branch dataset-runtime-20261019 https://github.com/Zeying-Gong/FLUX.git FLUX
cd FLUX
python3 verify_dataset_runtime.py
```

两份当前新数据集代码要求的文件SHA如下：

- `sage_utils/patched_base_command.py`: `ee2eb4c5091b22736ca76c794839c2bba7b0b8467a512aa3456bd555e9ca181b`
- `sage_utils/tracking_episode_collection_camera_only.py`: `9c14c3a3c57226fc5c470ff3d819d16138159a529ee980fd227f362870e7e1b4`

## 与数据集工具配合

新数据集oracle、审计、配方与当前主交接仍在 [TrackWAM/new_dataset](https://github.com/Zeying-Gong/TrackWAM/tree/main/new_dataset)。将其machine配置中的flux_root设为本分支目录。该工具中原有critical-file SHA校验可直接对上本分支。

本次没有上传大型场景资产、模型权重、运行日志、视频、本次STT/DT批量评测raw、凭据或m5私有配置。`assets/` 沿用公开基线已有内容；公开Git本来不包含的完整人形/机器人/场景资产仍需按项目下载说明或目标机器现有资产准备，不能把源码SHA通过当成完整资产或仿真通过。Sage3D场景由协作者自行准备；本分支的episode_caches只是源任务分类缓存，不是已通过oracle的episode。

## 镜像说明

m5已验证镜像可从下列不可变仓库摘要取得，无需由m5传离线导出包：

```text
quay.io/zeyinggong/flux@sha256:fbe832143ee4c688e5220a86bb0c106d461dd3d40befed489c4b4291aaae0071
```

实际仓库manifest的config digest与原本地image ID `sha256:754da4db87acf005ab7930a45df11648773688e1596cb694a286b46e9769718e` 一致。4090侧仍需核验其worker能拉取该镜像。离线镜像交付包只是docker save的备用运输文件，不是额外必需资产。

不要求任何情况下都使用完全相同的镜像字节；相同镜像用于减少环境差异。若改用其他Isaac镜像，需要匹配所用SDK/扩展/API，并重新做实际场景、人形动画、RGB-D、NavMesh和oracle审计验证，再记录新运行身份。当前TrackWAM迁移工具的image-ID检查仍固定原版本，不应直接绕过检查声称新环境兼容。当前迁移launcher基于SSH/Docker，K8s提交需要协作者适配执行后端，不能直接运行docker launcher就宣称已有K8s支持。

## 实际验证边界

m5源文件与本分支1389份文件逐SHA一致；Python源文件语法编译检查通过，凭据模式检查通过。新4090环境的完整仿真/DT评测尚未执行，Sage资产下载与md_ai_kit认证由该环境继续处理。
