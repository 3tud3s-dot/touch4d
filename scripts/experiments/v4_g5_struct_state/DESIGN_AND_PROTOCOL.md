# G5-Struct / G6-State-Struct V1

## 冻结范围

基于 G5 正式源码 `f978e0a29b6f0f0b6097c40684b949797a2f5d7b`，仅新增本独立实验目录和专项测试。
A 为原 G5-Direct 网络加弱结构 Loss；B 仅额外增加逐 Anchor GRU 状态读出。关闭 GEO，不新增属性头、A2A、接触门控或第四条 XYZ 输出。

两组均保留原 GS/LocalGS/Hand/Controller/T3 编码、两层 Cross、Base/Global/Local、18D 有符号运动、初态 FPS 和固定 RBF。
位置始终为 `X0 + W @ cumulative_d`；其余 11D 属性继承初态。只按当前已经到达的动作/触觉和预测历史递推，不输入未来 GT。

## 结构图与 Loss

只用初态有效 GS 和原 `initial_knn(...,16)` 候选（包含 Self 后再排除），保留互邻并去重的无向边。
长度 <=1e-6 的近零候选不参与局部间距；间距 `s_i` 采用普通中位数，偶数个候选取中间两项均值。
保留 `max(1e-6,0.1*min(s_i,s_j)) < length0 <= 2*min(s_i,s_j)`，全 H15 固定。
不使用 GT、人工绳索顺序或缺口桥接算法。原后端 kNN 的等距选择不变，最终边按索引排序。
空间邻图不是真实材料拓扑；折叠和重建缺口仍可能带来错误邻接，必须审查两张真实训练窗图。

`ratio=predicted_length/length0`，惩罚 `SmoothL1(relu(ratio-1.2),beta=.1) + SmoothL1(relu(.8-ratio),beta=.1)`。
先按窗口的有效边及 H 平均，再对 Batch 平均；空边窗口贡献连通的零值。结构项施加在最终 GS 中心，不是 Anchor 控制变量。
原训练 Loss 为 `CD + RGB + 0.1 Alpha`，总 Loss 仅额外加 `lambda_struct * structure`。
日志分开记录原 Loss、未加权结构项、加权结构项和总 Loss。Best 仍仅按完整 Val100 的原始 Joint 选择。

## 一次性校准

默认 lambda=.01。只读历史 G5 E8（SHA256 `067984c4c286306094efeff2f28ed57e3dcdd0c333972ecc6feb4b982ee3a40c`），
仅在 Train ep2/start141 和 ep7/start105，以同一完整 H15 预测 XYZ 为独立可微叶子，计算原 Joint 和结构项的 XYZ 梯度 L2 范数。
设最大未加权比值 `R=max(norm_struct/norm_joint)`，最终 lambda 为 `min(.01,.05/R)`，R=0 时保持 .01。
原 Joint 梯度为零而结构梯度非零时，安全系数为零；记录该不可辨识情况。绝不向上放大。
`calibration.json` 保留两个训练窗、原权重/源码、校准实际执行 SHA、组件哈希、范数和最终系数；A/B 使用同一文件 SHA。
断线重连复用校准，不根据验证结果重新选系数。5% 只约束这两个历史预测的输出空间梯度，不保证未来参数梯度或每个训练 Batch 的比例。

## B 的逐 Anchor 状态

S 明确指加时间 Embedding、Cross 之前的池化物体 Anchor 特征。每窗口 `Z0=S0`。
输入301D为 `S128 + CrossQuery128 + hand18_left + hand18_right + actual_FPS_history3 + previous_control_d3 + hand_valid2 + history_valid1`。
`LayerNorm301 -> Linear301,128 -> GELU -> GRUCell128,128`，无效手基底和无效 Anchor 严格屏蔽。
`enhanced_query = original_query + Linear128,128(Z)`，该读出权重/偏置精确零初始化。
Base/Global/Local 三处统一使用增强 Query；原 Alpha、ToolMotion12、HeadMass4 保持原规则。
状态输入的18D纯运动基底在当前 Global 输出之前计算；增强 Query 生成的实际 Global per-hand contribution 再正常交给 Local。
窗口内状态不 detach，窗口间重置。没有新的位移分支。

A620,511 参数，B775,353 参数，新增154,842。共享初始化逐项核验；B额外初始化使用独立 RNG 上下文。
A本来已包含 AR GS、累计控制和位置历史，因此 B 是额外递推记忆，不是“有记忆 vs 无记忆”。额外非线性容量也是混杂因素。
零读出可使初始 GRU 梯度为零；原 G5 运动输出层也零初始化，需在4次 Smoke 后核验读出及 GRU 的非零梯度和实际更新，不能只看 AdamW 衰减引起的参数变化。

## 正式配对协议

H15 / interval1，8 Epoch，Seed42，Batch1，AdamW lr1e-4、wd1e-4，原 norm1 裁剪。
完整1514 Train /100 Val（ep3=55、ep6=45），沿用原三相机1280x720 gsplat和掩码/归一化。
真实 DataLoader 的八轮顺序逐项核验并保存哈希；新增网络 RNG 不影响采样器。
每轮完整验证，先各 Episode 平均再等权。验证不加入结构项；不读取 ep0/ep9 Test。
从共同新初始化训练，不加载历史或 Smoke 权重。Smoke 固定 ep2/141、ep7/105、ep2/141、ep7/105，各4次更新后丢弃。

## 运行与恢复

只使用核验后的 Allocation32087、GPU UUID `GPU-4048d136-505b-fca2-631f-fda0ab7ab9af`；不创建新 GPU Allocation。
原 tmux gpu18 / Step32087.0 保留，本任务在独立 Slurm Step 和独立持久化 tmux 中执行。
先串行测两组 Smoke 峰值，再做无更新的实际双计算图并发探针。至少保留 max(15%总显存,4GiB)，另加峰值余量和CPU内存预算。
探针同时持有两组完整 H15 前向图后才放行反传；并发 OOM/显存预算不足转串行，不更改数据、Horizon、分辨率或 Loss。
所有子进程/PID启动时刻、Step、UUID、阶段、源码和完成标记持久化；同一 Launcher 使用文件锁。
完整模型/AdamW/Python/NumPy/CPU-CUDA RNG 与不可变 Epoch1..8 保存，包含 E2/E4；Best/Last 为同 Run 原始 Joint 的原子别名。
只恢复完整 Epoch；未提交 Epoch 回滚重跑，单独保留 attempted counter 和重算范围。若中断发生在待更新标记内，最多一个更新的执行情况未知，如实记录。
本轮自动恢复总数最多2次。NaN/Inf、数据/身份/因果性错误失败关闭，不调整模型或系数。资源失效则记录资源阻塞。

## 最终评价

比较 Copy、历史G5、A/B Best 和 Last，保留完整Val100、逐窗/逐H指标、同Epoch对照和负面结果。
分析初态边长度比和独立16NN代理、Near/Middle/Far GS运动及去平移残差；防止少动换规整。
固定四个主窗 H1/H5/H10/H15 三相机RGB/Alpha及同尺度几何，低变化三窗补充数值。
只对 B Best 七个代表窗做一次冻结 state_readout_off；它不是重新训练的 A。
GT为独立重建GS而非材料点标签；空间图/长度比不是接触/真实应变，渲染图不是原始照片。
训练、评估和报告源码分别记录实际执行 SHA。最终停止，不追加模型、Loss、超参搜索或训练预算。
