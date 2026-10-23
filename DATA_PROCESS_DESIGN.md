# V0 数据预处理

当前先训练 `001-rope`、`051-cube`，各自独立训练并保存权重；`008-pink-cloth` 暂缓。沿用 `DATALOADER_DESIGN.md` 的数据划分、五帧预测范围和 batch 结构。

## 处理位置

```text
Dataset / DataLoader → data_process → 模型与损失
```

DataLoader 返回原始单位的数据，并完成触觉世界坐标生成；`data_process` 在内存中统一变换 `inputs` 和 `targets`。重构文件保持原始单位，参数预先确定并固定，不在每个 batch 或窗口中重新拟合。

## 坐标归一化

每个 scene 使用固定中心 `center[3]` 和正的统一长度尺度 `scale`，三个坐标轴共用一个尺度：

```text
xyz_normalized = (xyz_world - center) / scale
```

- 参数仅由训练集可作窗口起点的初始 GS 估计，按原始帧去重、排除补齐点；验证、测试和未来目标不参与拟合。
- `center` 为上述全部真实 GS 中心点的均值，按点等权。
- `scale` 为上述各帧绕自身质心的 P99 半径的中位数。逐帧质心仅用于估计长度；实际坐标变换始终使用该 scene 的固定 `center`。
- 所有窗口、条件时刻和五帧目标共享同一组 scene 参数；验证、测试复用训练参数。

根据 `audit/input-representation-v1/summary.json`，当前参数如下（单位米）：

| Scene | center | scale |
|---|---|---|
| `001-rope` | `[0.210834337570, 0.013140156253, 0.157411820790]` | `0.157037090510` |
| `051-cube` | `[0.236388585803, 0.023145916698, 0.103283534840]` | `0.042746614739` |

| 字段 | 处理规则 |
|---|---|
| `gs_xyz`、`controller_points`、`tactile_xyz`、五帧 `target_xyz` | 减去 center，再除以 scale |
| `action_state[..., :3]` | 同上，变换动作中的位置 |
| `gs_attr[..., :3]`（log_scales） | 减去 `log(scale)`，使用自然对数 |
| `action_state[..., 12]`（原始开口） | 除以 scale，不增加裁剪 |
| 旋转矩阵、单位四元数、不透明度 logits、零阶 SH | 保持原值 |

夹爪几何仍按原始米制开口及既定裁剪规则生成，再变换生成的坐标；不将缩放后的开口代入原始运动学公式。

## 触觉幅值缩放

```text
tactile_normalized = tactile / 100
```

两场景、四路通道共用固定系数 100，保留连续非负值、有效零点、幅值比例和时间变化。允许结果超过 1，不截断，不设接触幅值阈值，不逐帧归一化；`log1p` 留作后续比较。

## 掩码与反归一化

- 掩码、真实点数、帧号和时间元数据保持不变。
- 变换后按各自掩码重新填零：GS 坐标及属性用 `gs_mask`，目标用各帧 `target_mask`，动作用 `action_valid`，控制点用 `controller_mask`，触觉坐标用 `tactile_spatial_mask`，触觉幅值用 `valid_mask`。
- 有效零触觉继续有效；二维读数有效性与三维触觉有效性独立保留。
- 模型位移与位置处于归一化坐标系。展示和汇总评估时还原米制；预测中的补齐点继续由掩码排除。

```text
xyz_world  = xyz_normalized * scale + center
delta_world = delta_normalized * scale
log_scales_world = log_scales_normalized + log(scale)
opening_m = opening_normalized * scale
tactile_original = tactile_normalized * 100
```

每个 scene 的参数随对应实验及模型权重保留，保证推理和反归一化一致。本文件记录处理约定，尚未实现 `data_process` 程序。
