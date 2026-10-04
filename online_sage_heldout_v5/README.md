# SAGE heldout：LightNav + Spot 动力学在线测试

本目录保留2026-10-04的Spot基线v5。这是**单条派生heldout episode的在线闭环验证**，不是整个heldout验证集的成功率。LightNav只接收实时RGB和指令，位姿和目标位置不输入模型。

[在线视频](results/online.mp4)（左：模型输入，右：第三人称） · [预览](results/preview.jpg) · [轨迹/NavMesh](results/online_navigation_map.png)

[结果与限制](results/analysis.json) · [动作记录](results/actions.json) · [验收](results/verified.json) · [当次代码快照](results/source_61004.tar.gz) · [快照哈希](results/source_hashes.json)

## 实现

使用Isaac Sim 5.0自带的无机械臂Spot USD和官方SpotFlatTerrainPolicy。12个关节以位置目标驱动，由重力、关节力和原场景碰撞网格产生运动。初始化后不直接设置底座位置或速度。

| 项目 | 设置 |
|---|---|
| 场景 | SAGE-3D 0402_840115，派生路径及来源见case_v4.json |
| 模型 | LightNav-0 trackvla接口；首个waypoint转vx/vy/wz |
| 频率 | 模型4Hz，行走50Hz，物理500Hz，动画/画面更新100Hz，视频30fps |
| 相机 | 640×480，水平FOV90°，机身平移[0.24,0,0.24]米 |
| 相机姿态 | 随机身完整姿态，实际离地0.718–0.781米，未人为稳定 |
| 人物 | 原生动作匹配步态，名义0.9m/s，实测0.892m/s，无强制逐帧位移 |
| NavMesh | 半径0.30m缓存用于审计；实际运动由物理碰撞决定，不投影修改位姿 |

## 结果与边界

Job61004 / Task71927，百度4090，两张RTX4090，已完成并通过动态闭环验收。

| 指标 | v4b运动学 | v5动力学 |
|---|---:|---:|
| 机器人路径 | 3.121m | 4.722m |
| 平均距离 | 3.694m | 3.389m |
| 最终距离 | 5.671m | 4.053m |
| 在线动作/观测数 | 30 | 30 |

v5执行3750个物理步、225帧/7.5秒视频；无跌倒，最大倾斜2.31°。**存在障碍物接触，不能算无碰撞跟随成功**：右前上腿及脚与场景网格接触；中心点36/750、三点占地153/750个采样超出旧NavMesh。

同时改变了资产、动力学、碰撞处理和相机随动；人物动画更新频率不同，实测步速从0.902变为0.892m/s。因此不能把距离变化解释为动力学的单因素收益。场景来自heldout，人物路径是通行检查后的派生路径。

## 复现

需要另备有权使用的SAGE-3D场景/碰撞资产、Isaac Sim及People扩展、LightNav代码和权重。第三方机器人资产、扩展和模型权重不随本目录发布。版本见runtime_manifest.json；外部文件哈希见asset_source_hashes.json。

脚本保留已验证NAS部署路径。迁移时检查run_suite.py的Python/LightNav路径、run_online.py的ASSETS/Isaac路径、policy_server.py的checkpoint、840115.usda的场景引用、case_v4.json的人物资产及online.kit/YAML路径。不是任意机器上一键可用的包。

~~~bash
# 在本目录配置外部运行包（包含Shapely和NVIDIA兼容扩展）
python prepare_runtime.py --runtime-source /data/nas_ray/home/zeying.gong/algorithm/repos/FLUX-sage3d/online_sage_heldout
# 通过已配置调度器提交完整两卡任务
/data/nas_ray/home/zeying.gong/algorithm/envs/md_ai_kit_submit/bin/md_ai_kit submit online_spot_v5.yaml
# 对完成结果重新验收和生成图表
python verify_rollout.py /path/to/output/speed_090
python analyze_online.py /path/to/output
~~~

完整任务命令见online_spot_v5.yaml。run_suite.py保存代码快照、哈希、模型/仿真日志和验证结果。原始NAS产物：
/data/nas_ray/project/md-ak/users/zeying.gong/job_61004/task_71927/sage_lightnav_online_spot_v5

## 调试与保存

初始0.50m高度引发腿部地面穿入和倒地，改为官方示例0.70m落地初始化后稳定。60Hz画面不整除500Hz物理，实际只有8步/帧，改为100Hz后准确3750步/7.5秒。失败日志及全部接触/关节记录保留在NAS，未混作正式结果。

Spot基线保留在本目录；Go2会用单独目录和匹配控制器，不直接套用Spot权重。
