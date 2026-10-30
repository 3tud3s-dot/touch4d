# G5-ARAP / G5-ARAP-Iso V1 冻结设计与工程门禁

## 隔离与研究问题

本目录来自原 G5 正式源码 `f978e0a29b6f0f0b6097c40684b949797a2f5d7b`。C 为原 G5-Direct 加短时 ARAP，D 只额外加入更弱的相对初态等距漂移项。推理网络、所有运动分支、固定 RBF、数据、Renderer 和原 Joint 均不修改。不引入 State、GEO 或新的位移分支，不修改正在推进的 Struct/State 实验。

参考权重只用于只读校准与验收：`/data1/userdata/tcweng/experiments/touch4d/v4-g5-direct-v1-training/run-001/best.pt`，SHA256 `067984c4c286306094efeff2f28ed57e3dcdd0c333972ecc6feb4b982ee3a40c`。两组正式训练和 Smoke 都从 Seed42 重新初始化，正式训练不加载 Smoke 状态。

## 初态空间图

精确复用 Struct 实验的初态有效 GS 16NN：排除 Self、Padding 和不超过 1e-6 的近零候选边，取每点有效候选长度中位数 s（偶数取中间两数均值）。只保留互为近邻、去重无向边，且 `max(1e-6,0.1*min(si,sj)) < length <= 2*min(si,sj)`。整个 H15 图保持固定，不读取 GT 或预测来重新建图。ARAP 将无向边展开为两个中心的邻边；Iso 每无向边只计一次，权重都是1。

空间邻域不是材料拓扑。只在 ep2/start141 与 ep7/start105 的初态图进行简洁可视化审查；记录过滤比例、最长边和跨段误连风险，不根据验证误差更改规则。不人为连接缺口、不增加绳索顺序。

## 短时 ARAP

连续序列为初态 X0 和预测 X1..X15。每个有邻边中心 i、每步 k 拟合 `e_current ≈ R e_previous`。用 detached Kabsch SVD 得到合法 SO(3)，反射时纠正最后一个奇异方向，保证 det=+1。仅 R 的求解停止梯度，当前和上一预测中心都保留计算图。

每中心损失为邻边残差平方和，除以该中心**固定初态**邻边长度平方和加 `1e-12`（归一化坐标的平方长度单位）；再对有效中心与时间平均，最后窗口等权。无边窗口贡献可微零，仍占一个窗口权重。共线与平面邻域不屏蔽；扭转可能不唯一，不解释为材料姿态。SVD 批处理失败时逐矩阵重试，个别仍失败时采用 Identity 但保留其残差，明确统计，不掩盖求解失败。校准与 Smoke 门禁要求有效中心的最终 Identity fallback 数量为零；rank1/2 等可解退化并不属于 fallback，不据此屏蔽。

## 长期 Iso

对预测边计算 `SmoothL1(log((length_current+1e-6)/(length_initial+1e-6)), beta=0.1)`，窗口内对有效无向边及 H15 平均，再窗口等权。1e-6 为归一化长度单位。Iso 不代表真实材料不可伸长；弯曲会改变空间弦长，指标下降不能单独证明物理改善。

## 一次预设梯度校准

默认 λA=0.01、λI=0.001。仅用冻结 G5 E8 在 **Train ep2/start141、ep7/start105** 的同一预测，比较原三相机 Joint、ARAP、Iso 对完整 H15 最终归一化 GS XYZ 的 L2 梯度范数。

`λA=min(0.01,0.05/max(norm_ARAP/norm_Joint))`；`λI=min(0.001,0.01/max(norm_Iso/norm_Joint))`。只下调一次，不用 Val/Test、不向上放大。零或非有限梯度先保存图/求解统计并报告工程阻塞，不以把正则权重改零绕过。C/D 共用同一 λA；C 不将 Iso 加入训练，D 使用同一冻结 λI。

## 正式配对协议

- H15/Δ1、Seed42、B1；8 Epoch，每轮1514 Train，Val ep3=55/ep6=45。
- AdamW lr=1e-4、wd=1e-4、clip norm1；原三相机1280×720 gsplat，不降低分辨率或删相机。
- 原监督 `CD+RGB+0.1 Alpha`。Scale/Rotation/Opacity/SH 11D 始终继承初态。
- C Total=Original+λA ARAP；D 再加 λI Iso。分别记录原始与加权结构项，不能将 Total 与历史 Joint 混比。
- 实际核验共享参数初始化、Manifest 和八轮完整 DataLoader 顺序，不只比较 Seed。
- Best 只按完整 Val、episode 等权的**原始 Joint**选取。每完整 Epoch 不可变快照，保留 Initial/Best/Last/E2/E4、RNG、优化器、访问数与成功更新数。
- 只支持完整 Epoch 边界恢复。中断 Epoch 回滚重跑，额外计算单独记录，不混模型、源码、Loss、配置或 run。改变数学需保留旧 run 并从头另起，不能拼接。

## CPU、Smoke 与 Slurm

数学测试覆盖刚性不变、SO(3)方向/反射、剪切伸缩、双时刻梯度、有限差分、退化/NaN/Mask/空图。训练系统测试覆盖配对初始化/顺序、严格 Checkpoint、Best 与完整 Val。独立 Review 后才冻结提交、同步运行。

在首次 GPU 运行前固定：同一预测输出的 ARAP/Iso 复算使用 `atol=1e-7, rtol=1e-5`，记录实际绝对差，容纳 CUDA scatter-add/SVD 的合法浮点舍入；不用于放宽原 G5 H15 输出、属性、模型参数与 Checkpoint 的逐位等价检查。

一个 prepare_smoke GPU 作业依次校准、C/D 各四次真实更新（ep2/141→ep7/105→重复），验证完整 H15、全部分支梯度和参数变化、结构非恒零/梯度尺度、11D 属性不变、保存恢复。低于历史八轮模型性能不是 Smoke 失败。NaN/Inf、反射、Mask/归一化错误或求解严重失稳才是工程失败，最多三轮明确修复，保留日志。

下游 C/D 正式训练以 afterok 依赖 prepare，启动时再次检查源码、配置、校准、门禁与 Manifest。每进程一张 GPU，C/D 同时最多两张；GPU 不足由正常排队串行，不使用或取消32087及其他作业。失效依赖自动终止等待，修复后仅替换本轮自己的失效任务，防止重复。执行 SHA 与后续报告 SHA 分别记录。

## 验收与解释边界

完整 Val100 比较 Copy、历史G5、C/D Best和Last，保留同 Epoch、H1..H15、逐窗口配对与 ep3/ep6 分项。除训练项外沿用独立原16NN结构代理、Near/Middle/Far GS运动、去平移残差、有符号XYZ和低变化漂移。固定4主窗口及3低变化窗口，关键4时刻、3相机RGB/Alpha、同世界尺度几何，每行最多3图。

实际检查有效运动是否保留，防止少动换规整。GT 每帧独立重建，无材料点对应；不能称空间图为材料拓扑、二维缺口为三维断裂、边长代理为真实应变。本轮不保证指标改善，不追加模型、调参或延长预算。
