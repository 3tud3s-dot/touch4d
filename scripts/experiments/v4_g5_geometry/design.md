# G5-GEO-v1 输出端几何适配与冻结验收协议

本实现基于 `f978e0a29b6f0f0b6097c40684b949797a2f5d7b`，严格加载 G5-Direct E8 Best（SHA256 `067984c4c286306094efeff2f28ed57e3dcdd0c333972ecc6feb4b982ee3a40c`）。生产 V4/G4/G5、数据、Loss、Renderer、固定 RBF 均不修改。

## 计算路径

原 G5 在 eval/no_grad 下完成完整自回归。只读 hook 捕获最后一层 Cross 的 Query `[B,H,128,128]`；与原 Trace 的最终累计 Anchor 位移 `[B,H,128,3]` 拼接。输入 detach 后经过 `131→128→64→6`、两次 GELU，最后线性层权重和偏置精确零初始化。

前 3 维经 `log(1.25)*tanh`，后 3 维经 `(pi/6)/sqrt(3)*tanh`。原固定 RBF 将有界 log-scale 增量和旋转向量映射到 GS。尺度相对初态的局部主轴，旋转增量使用世界 WXYZ 左乘 `delta_q ⊗ q0`。先插值旋转向量再做 Exp，不插值四元数。

`motion` 保持原 G5 输出；另建 `render` 字典，只有 log_scales/quaternions 被替换。整个运动 rollout 已先结束，因此新属性不能进入下一步 GS Encoder。几何头无法修复中心方向、中心分布或原 Anchor 结构失真。

四元数指数使用 sinc 的解析零极限。保留原始单位四元数精度，不在输出路径重复归一化 q0，以免身份变换改变 renderer 舍入；协方差诊断显式正规化旋转。原数据单位四元数的数值误差会记录。Padding 在非线性计算前置零。

## 模式与快照

- `geometry_off` 原样返回原 G5 输出。
- `geometry_on_zero` 真正经过可微头和属性运算，要求最后投影仍精确零；若参数已更新会拒绝，而非偷偷重置权重或绕过计算。
- `geometry_on` 使用当前头。

新格式只保存 Geometry Head、独立架构配置、骨干来源与状态哈希、源码、Processor、运行身份、更新次数和可选优化器/RNG。严格拒绝跨配置、跨骨干、跨运行身份；不可覆盖已有快照。原 G5 格式不混用。

## 预先冻结 Smoke

完整 Val100（ep3 55、ep6 45，episode 等权）分别运行原 G5、off、on_zero；每个窗口/Horizon 检查原输出等价。XYZ 要求逐位一致；CD 绝对差 ≤1e-4 mm，RGB/Alpha/Joint 绝对差各 ≤1e-6。门槛不随结果调整。

真实原三相机 1280×720 gsplat、原有效深度区域与 reduction。复用 `v4_objective`，明确 `CD+RGB+0.1 Alpha`、scale_weight=0、depth_weight=0；原函数仍计算 scale_regularization 诊断量，但不优化它。

基线门槛全部通过后，只执行四次 AdamW 更新：ep2/start141、ep7/start105、ep2/start141、ep7/start105。Seed42、H15/Δ1、B1、lr=1e-4、weight_decay=1e-4、clip_norm=1。优化器仅含 Geometry Head。无自动重试、无验证选步，无 scheduler。保存初始和最终四次更新快照。

最终完整 Val100 对照和 off 回退检查；保留逐窗口、Horizon、相机、几何统计与正负变化。四项均保持或改善仅表明这次固定快照检查通过，不证明未来训练必然不退化。

尺度倍率 0.8–1.25、旋转向量范数 ≤30° 是工程限幅；不保证中心 Jacobian 一致，也不代表物理应变。GT 逐帧独立重建，不能按行号监督尺度/旋转或解释材料运动。

## 用户追加授权与后续边界

用户随后明确授权自主修复、提交、GitHub 推送、服务器同步、GPU 申请，并将最终目标延伸到达到可训练状态后完成八轮 Head-only 训练及验收。仍冻结原 G5；八轮正式入口应单独实现、测试、提交和冻结来源，不把 Smoke 的四次更新冒充正式轮次。不得通过放宽等价/隔离标准掩盖工程问题，失败记录必须保留。Smoke 不会自行触发八轮作业。
