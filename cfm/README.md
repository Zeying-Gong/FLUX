# CFM Baseline（NavDP 的 CFM 版本）

基于 [NavDP](../navdp) 复制而来，将 DDPM 推理替换为 **Conditional Flow Matching (CFM)** ODE 采样，与 InternNav 中 CFM 微调权重对接。

## 与 NavDP 的差异

- **策略网络**：`CFM_Policy`，`cond_pos_embed` 为 `memory_size*16+2`（与 CFM checkpoint 一致）
- **采样**：ODE Euler 积分（默认 5 步），替代 DDPM 10 步
- **归一化**：支持 `normalization_config.json`，推理时对动作反归一化后再 `cumsum`
- **混合目标**：`predict_ip_action` 暂仍用 DDPM，其余任务均为 CFM

## 依赖

- 与 NavDP 相同；`depth_anything` 通过符号链接复用 `../navdp/depth_anything`。

## 运行 CFM 服务器

```bash
cd baselines/cfm
python cfm_server.py \
  --checkpoint /path/to/cfm_finetune/checkpoint-XXX/pytorch_model.bin \
  --normalization-config /path/to/normalization_config.json \
  --cfm-steps 5 \
  --port 8889
```

- `--checkpoint`：CFM 微调权重（必需）
- `--normalization-config`：与训练时一致的归一化配置；若省略，不做反归一化
- `--cfm-steps`：ODE 步数（默认 5）
- `--port`：默认 8889，与 navdp 的 8888 区分

## 评估

与 NavDP 相同 API（`/navigator_reset`、`/pointgoal_step` 等），仅端口不同。例如：

```bash
# 启动 CFM 服务器后，评估时指定 --port 8889
python eval_pointgoal_wheeled.py --port 8889 ...
```

`client_utils` 的 `navigator_reset(..., port=8889)`、`pointgoal_step(..., port=8889)` 等即连到 CFM 服务。

## 文件说明

| 文件 | 说明 |
|------|------|
| `policy_network.py` | `CFM_Policy`：ODE 采样、反归一化、CFM 推理 |
| `policy_agent.py` | `CFM_Agent`：预处理、历史帧、调用策略 |
| `policy_backbone.py` | 与 NavDP 共用（符号链接 depth_anything） |
| `cfm_server.py` | Flask API，与 navdp_server 接口兼容 |
