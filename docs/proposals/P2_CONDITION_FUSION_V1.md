# P2 Condition Fusion V1：只读审计与候选实验

状态：2026-10-26 技术审计；本轮只提交本提案，尚未选择或实现架构改动，不启动训练或 CUDA 推理。

## 1. 基准与执行边界

- 从最新 `origin/main` **`64b1320d3718e7d05d590ce854b8aeb05f14236f`** 创建 `codex/p2-condition-fusion-v1`，不是从 H5/H15 实验分支派生；只使用本地 `D:\Amax\Projects\touch4d`。
- 正式基线是 P2：14D GS → LocalGSBlock（k=16）→ 128 个 FPS 锚点的两层结构 attention → 逐 GS 条件 cross-attention → 共享锚点累计位移 Head → 固定各向同性 RBF。宽度 128、4 头；位置相对初态累计预测，其余 GS 属性继承初态。
- 基线须显式指定 `--gs-decoder anchors --gs-structure anchor_attention --gs-local-attention knn --controller-features 7 --prediction-mode autoregressive --output-mode gs_geometry`；不能以 CLI 默认 Free Head 代替 P2。
- 不引入 P3 双帧状态、可学习 RBF、属性更新、新 loss 或新监督掩码。本轮不改现有源码、数据、checkpoint 或其他文档，不创建 PR、不合并 main。
- 用户原有 6 个修改／未跟踪文档在切换前后 SHA256 一致，保留其本地状态，不纳入本次提交。旧 `DATALOADER_DESIGN.md` 的部分 V0 并行预测表述是历史设计；本提案的 AR/H 行为以所列源码为准，不顺带修改旧文档。

以下是**本地提交回执记录**，不是本轮实时作业状态查询。本轮不连接服务器、不更新任何服务器 worktree，也不读取正在发布的 checkpoint：

| 受保护实验 | 源码 SHA | 服务器执行目录 | checkpoint 输出目录 |
| --- | --- | --- | --- |
| H15，Job 30666 | `ca84a6e12035ddb85d15d1251852ca99d69495bf` | `/data1/userdata/tcweng/projects/touch4d-worktrees/p2-baseline` | 同目录 `runs/001-rope/p2-001-h15-e8-30666/` |
| H5，Job 30695 | `b9771780956c2182de47a1377baeeaba50c3c5f1` | `/data1/userdata/tcweng/projects/touch4d-worktrees/p2-001-h5-e8` | 同目录 `runs/001-rope/p2-001-h5-e8-30695/` |

回执：[H15](../../audit/p2-001-h15-e8-submission-v1/submission.json)、[H5](../../audit/p2-001-h5-e8-submission-v1/submission.json)。两提交的 `training/`、`dataloader/`、`data_process/`、固定窗口与 Oracle 实现均与本次 main 基准无差异；本地切分支不会改变服务器已绑定源码。

## 2. 当前数据流：代码事实

记 B 为 batch，N 为初态 GS 补齐点数，D=128，A≤128 为有效锚点数，H 为预测长度。条件索引 j=0…H−1 对应 t+j；预测 h=1…H 对应 t+h，只读取 j<h。目标 GS 独立重建，不假设跨帧点 ID。

H15 仍是随着动作／触觉逐步到达的在线条件预测，不宣称在 t 时刻已知未来全部触觉的开放环预测；新编码器同样只能处理当前可用前缀。

| 数据 | 当前 batch shape | 实际语义／模型入口 |
| --- | --- | --- |
| `gs_xyz` / `gs_attr` / `gs_mask` | `[B,N,3]` / `[B,N,11]` / `[B,N]` | 初态；属性为 log-scale3、WXYZ4、opacity logit1、SH DC3；每步重新编码当前预测位置与初态属性 |
| `action_state` / `action_valid` | `[B,H,2,13]` / `[B,H,2]` | `[世界位置3, 旋转矩阵按行9, 原始开口1]`，左右手；MLP 输出每时刻 2 个状态 token |
| `controller_points` / `controller_mask` | `[B,H,2,30,3]` / `[B,H,2,30]` | 每手两指各15个几何控制点；7D 编码后每时刻60个 token |
| `tactile` / `valid_mask` | `[B,H,4,12,32]`，同形 bool | 四指依次左手左指、左手右指、右手左指、右手右指；每指目前只有1个 token |
| `tactile_xyz` / `tactile_spatial_mask` | `[B,H,4,12,32,3]` / `[B,H,4,12,32]` | 传感器表面几何位置及资格；XYZ 已生成、归一化，但当前模型未读取 XYZ |
| `tactile_age_us` | `[B,H,4]` | 该路触觉相对当前 GS 时刻的样本年龄；模型除以75000 |
| `meta.observation_timestamp_us` | `[B,H+1]` | t…t+H 的真实微秒时间戳；目前未传给模型时间编码 |

数据来源：[dataset.py:326](../../scripts/dataloader/dataset.py#L326)、[collate.py:7](../../scripts/dataloader/collate.py#L7)、[geometry.py:47](../../scripts/dataloader/geometry.py#L47)。统一变换见 [processor.py:175](../../scripts/data_process/processor.py#L175)：GS、控制点和触觉 XYZ 同场景中心／尺度归一化，开口除以尺度，旋转矩阵不变，触觉仅除以100。

### 2.1 Action / Controller

`actions` 是同帧记录的夹爪位姿和开口，**不是力、力矩、控制命令或下一步动作标签**。状态来自 `T_worlds/openings`；控制点是同一位姿和裁剪开口的几何展开，不是60路独立观测。状态保留未裁剪开口，故二者并非完全重复。手身份来自 metadata，不能按原始 slot 编号猜左右手。

[model.py:223](../../scripts/training/model.py#L223) 当前7D为：

```text
[q_j (3), q_j-q_0 (3), valid_j & valid_0 (1)]
```

差分前清理无效数；初态无效仅禁用累计差分，当前有效坐标仍产生 token。已有累计运动信息，不能称其为纯静态编码。局限是：状态／控制点 MLP 各时刻独立，没有显式相邻增量或物理时间间隔；比较时间变化的工作交给后面的 GS cross-attention。

首步只有 t 时刻状态，没有 t−1 观测；它无法仅从本窗口确定真实初始速度。后续已观测前缀提供运动信息，但不包含尚未发生的控制意图。增加显式差分只能改善表达，不能凭空增加运动观测。

### 2.2 Tactile

重构按每路 GS 时刻之前最近 raw 样本配对，年龄0–75 ms有效；`max(raw-median,0)`、有限数与区域掩码不变，有效零触觉保留。同一 raw 样本可能被连续 GS 时刻复用。见 [rebuild.py:314](../../scripts/rebuild_dataset/rebuild.py#L314)、[dataset.py:184](../../scripts/dataloader/dataset.py#L184)。

[model.py:239](../../scripts/training/model.py#L239) 将每指 `384幅值 + 384 valid + 384 spatial valid + 1 age`，共1153维压成128维；加指身份及时间 embedding。Flatten 保留固定行列顺序，本身不丢信息；**1153→128 的整指压缩、缺少局部共享编码和显式三维关联才是待验证的瓶颈**。不能直接声称模型只看触觉总量。

[geometry.py:69](../../scripts/dataloader/geometry.py#L69) 用当前 GS 时刻夹爪位姿、裁剪开口及固定像素网格生成 XYZ，空间资格为 `valid_mask & action_valid & finite(XYZ)`。它表示传感器表面位置，不是物体真实接触点、遮挡标签或无滑动对应。幅值可早于位姿0–75 ms，因此定位存在时间近似；本轮方案保留 age，不擅自插值、改配对或设接触幅值阈值。没有合法位姿的有效二维读数仍应保留。

### 2.3 Cross-Attention 与 AR

每时刻共66个 token：2状态+60控制点+4触觉。条件 MLP 一次编码所有时刻，但没有跨时刻混合；第 h 步实际截取前 h 个条件时刻，另加一个始终有效的 null token，避免全无效 softmax。见 [model.py:239](../../scripts/training/model.py#L239)、[model.py:303](../../scripts/training/model.py#L303)。

两层 [CrossBlock](../../scripts/training/model.py#L35) 使用逐 Gaussian query 和条件 K/V，仅有内容点积及 key padding mask；没有 GS–控制点／触觉的相对 XYZ 偏置，没有显式局部接触关系，也没有条件 token 间的 self-attention。当前 token 数不平衡可能影响优化，但60个控制点不等于动作贡献占60/66，attention权重也不等于因果贡献。

GS 内部已有相对空间编码：[LocalGSBlock](../../scripts/training/model.py#L89) 和 [StructuralGSBlock](../../scripts/training/gs_structure.py#L20)。它们不等同于跨模态空间融合。每步 GS/Local/Structure 重新编码，前步预测计算图保留；锚点 Head 预测相对 x₀ 的累计位移，不把累计量重复叠加到上一预测。

H15 的计算瓶颈需区分理论交互量与实际 CUDA 性能。每层、每头的条件 attention 配对数为：

```text
B × N × Σ(h=1…H)(66h+1)
H5：995BN；H15：7935BN，约7.97倍。
两层四头 H15：63480BN；单步最长 memory 为991。
```

这里 N 是 batch padding 后的 N，缺失手槽位仍占张量长度；mask不等于省去计算。条件 MLP 只算一次，但 K/V投影随前缀在每层每步重算。上述不含 GS 局部／结构编码、RBF、渲染、CD及反向；`need_weights=False` 的实际内存依赖后端，不能把理论score矩阵当作已测峰值显存。

## 3. 静态倾向：现有证据与未知项

**代码事实：** 零初始化位置 Head 初始即复制初态；训练后仍可利用初态形态和 horizon embedding 拟合规律而弱化条件。这是可行的捷径，不是已证明的当前 P2 行为。AR 存在也不自动证明利用了动作，或远期梯度足够强。

**既有实验事实：** [P2 H15 固定窗口结果](../../audit/structural-gs-h15-v1/stage3_comparison.json)中，001-rope 两窗口平均 CD约1.964 mm，复制初态约7.283 mm，说明这些窗口上模型确实产生了有效运动；不证明动作／触觉利用或跨 episode 泛化。[CUDA审计](../../audit/structural-gs-h15-v1/stage3_acceptance.json)确认完整跨步图，但近初始化末步到最早 Head 的梯度很弱；不能据此断言所有已训练 checkpoint 都梯度消失。

**历史线索：** [condition-trace-v1](../../audit/condition-trace-v1/results.json)来自早期7维控制点、仅位置 AR 模型，非当前 P2。其 last 在选定训练／验证窗口上的动作冻结平均 ΔCD约 +0.0020/+0.0008 mm；触觉置零约 +0.1199/−0.0176 mm。采样是目的性诊断，未严格匹配幅度，不能把这些数值当作 P2 结论或“条件贡献率”。旧报告可复用诊断方法，不能原样复用硬编码五帧和逐点 Head 的 hook。

**尚未确认：** Job 30695/30666 的完整训练 P2 是否静态、条件是否有益、H15后段是否更依赖动作。本轮未做这些 checkpoint 的推理，也不宣称已定位根因。

## 4. 三项候选：分别验证，不一次叠加

以下是讨论方案，配置值不是已批准的实现要求。先以同一个未改 P2 为参照独立实验，有证据后才考虑组合。Loss、RBF、AR累计位移参考点、属性继承、数据划分及掩码规则均不随候选改变。

### A. 因果相邻控制点增量（优先讨论）

**假设：** 已观测动作变化需要显式近期运动表达，避免只靠注意力从累计位移／绝对位置中比较时刻。

```text
输入：[B,H,2,30,11]
[q_j(3), q_j-q_0(3), initial_pair_valid(1),
 q_j-q_(j-1)(3), adjacent_pair_valid(1)]
输出：[B,H,60,128]，总条件仍66 tokens/时刻。
```

- j=0新增差分为0、标记False；j>0仅相邻两端有效才差分。初态无效但后续相邻有效时可恢复近期运动表达。现有 batch 足够，不增加历史读取或重构数据。
- 这叫**相邻观测间位移**，不是速度；首步缺少历史的问题仍在。暂不同时增加速度、角速度、开口速度或时序网络。
- 宽128时首层增加512参数，B=1/H15额外约46.08万MAC，token交互规模不变。若后续独立测试速度，只能使用已观测条件时间的后向差分；meta的最后一个目标时刻及前向／中心差分不得进入输入。
- 对照 A0=原7D，A1=11D；保持触觉与融合原样。共享参数显式对齐，新增输入列零初始化，验证初始输出等价；同seed本身不能保证改维度后的随机数流相同。
- 新模式必须显式写入 checkpoint，旧3D/7D路径不变；不能把旧 checkpoint 静默恢复成11D。先检验动作冻结响应是否有益，再判断增量是否值得保留。

### B. 少量带三维位置的触觉 patch

**假设：** 整指摘要难以让 GS 区分不同局部压力模式的位置；小规模 patch 能暴露空间信息，同时保留已有无定位读数通路。

- 原4个 finger token保留；候选新增每指4个 `12×8` 列分段 patch，共16个。每patch输入为 `96幅值 + 96二维valid + 96空间valid + 3位置 + 1年龄`，共292D；共享MLP输出 `[B,H,4,4,128]`，加指／patch身份与条件时间。总长度66→82。
- patch位置由空间有效像素 XYZ **等权均值**得到，随观测位姿变化；不按幅值加权，不把均值叫作真实接触中心。新空间分支幅值仅取空间有效位置，`patch_keep=any(spatial_mask)`；valid zero仍保留，无空间资格的二维读数由原finger通路保留。全无效patch屏蔽，不能把填零位置当世界原点。
- 输入已有，无需改变数据集；age保留0–75ms定位近似。全高patch只给出patch级位置，行方向压力位置仍由固定顺序幅值隐式表达，这是低成本分辨率取舍，未宣称解决全部空间信息瓶颈。
- 对照 B0=同样16patch但XYZ三维置零，B1=提供XYZ；两者292D、mask、token数及初始化一致，均另与原P2比较，区分增加局部token的容量收益和三维定位收益。本轮不同时加C的相对偏置。
- 在原逐GS融合下，H15 memory配对系数7935→9855，约增加24.2%；新增patch两层MLP约12.9MMAC/B（H15、宽128，不含bias/激活），不等于总运行时间／显存比例。
- 新token即使数值为零也会改变softmax分母，不能宣称“零token初始化等价原P2”；保证配置关闭时旧行为不变，B0/B1开启时彼此配对。是否值得此成本需未来真实CUDA测量。

### C. 基于 GS Anchors 的空间条件融合

**假设：** 在与位移Decoder同一锚点层融合几何条件，可降低逐GS查询成本，并使局部条件影响更容易定位；不能把查询压缩的变化混称空间建模收益。

```text
结构编码后 GS features [B,N,128]
→ 既有初态邻域池化 [B,A,128]
→ 两层条件 cross-attention，memory [B,66h+1,128]
→ 共享 Head [B,A,3]
→ 原固定 RBF [B,N,A] → x₀ + 累计位移 [B,N,3]
```

- 首轮单独沿用原66token和A≤128，不依赖A/B候选。当前位置取当前AR状态中固定anchor索引的中心；条件位置来自各已到达时刻：状态token的夹爪位置、控制点XYZ、整指空间有效像素XYZ均值。整指均值是粗空间代表，后续有必要才结合B。
- C0只将query从N个GS移为A个anchor，沿用内容attention；C1在C0上加每头相对位置bias：`q·k/sqrt(d) + b(p_condition-p_anchor)`，候选小MLP `3→32→4`。不设距离截断、接触阈值或固定材料方向。内容有效但无法定位的token仍可参与内容attention，几何bias=0；null无位置、bias=0。query/key的资格分别检查。
- C1的bias输出层零初始化，与C0初始等价；**C0改变了池化与attention顺序，不等价于原P2**。保留P2独立配置和旧checkpoint语义，三组 P2/C0/C1 分别判断查询压缩与几何bias。初始位置Head为零使三者坐标暂时一致，不代表内部特征或后续行为相同。
- 旧冻结窗口初态N：rope3473/3782，cube1309/1212。H15两层四头条件配对数原为约220.47/240.08/83.10/76.94M，A=128后约8.13M，约为原来的3.38%–10.56%。这是该attention部分，不能直接宣称整模型加速同倍。
- C1若每层独立bias MLP，额外520参数，H15约454MMAC/B的相对bias计算（66token、两层、满128锚点）；须实测相对位置激活与完整反向开销。它替换原逐GS条件cross-attention，不额外保留一套逐GS融合再声称省算力。GS Encoder、局部结构及RBF不变。

## 5. 先验证当前完整训练 P2 的条件利用

此节是**后续获准执行的诊断协议**，本轮不执行。等当前作业自然完成、checkpoint稳定发布后，记录best/last的路径、SHA256、源码、完整config、processor、epoch与实际更新数，评估前后核对哈希。用现有 `evaluate.load_model` 加载，禁止以模型默认值重建，禁止把H15 checkpoint截成H5加载。

`epochs` 模式的 `best.pt` 按验证集联合score选取，主要为RGB + 0.1 Alpha + 1×归一化CD（render可用样本）；不是最低CD checkpoint，scale正则单独记录。见 [train.py:172](../../scripts/training/train.py#L172)、[train.py:748](../../scripts/training/train.py#L748)。best和last都报告，不能只选择有利者。

### 5.1 取样与对照

- 001训练episode保持1/2/4/5/7/8，验证3/6，不使用test0/9。先按GT运动幅度、已观测控制点增量、触觉强度／空间资格分层冻结少量训练和验证窗口，不根据本轮预测效果挑选；同层尽量匹配且报告未匹配项。
- 单列历史ep3退化窗口65/115/150及其诊断来源，不与代表性样本混成无偏汇总。H15先核实长度与监督完整性，不跨episode；不为更好结果更换窗口。
- 对每个checkpoint固定初态GS、目标、mask与相机，运行真实条件、动作固定初态、有效触觉值置零、指内固定空间排列打乱，并保留复制初态参考。打乱只覆盖整个窗口共同有效像素，跨时间同一排列；固定seed11/22/33，记录覆盖比例，不把未打乱像素视为已干预。
- 动作冻结同时冻结 `action_state` 和 `controller_points`，新差分需重算；对初态无效的手／点不能伪造参考，报告未覆盖比例。当前P2未使用XYZ；未来B/C应另标“动作数值独立扰动”和“所有位姿派生几何一致冻结”，后者同时重算tactile_xyz／patch位置，不能让真实坐标成为隐藏动作通路。即使几何一致，冻结动作与真实触觉也可能物理不相容，这属于诊断干预。
- 触觉置零保留valid、spatial、age与位置，不把有效零当缺失；空间打乱保持原位置和mask，只置换幅值。两种扰动不同于删除整个模态，不能据此声称已测得全部模态贡献。

### 5.2 逐层与输出指标

追踪 `原始条件 → MLP → 身份/时间编码 → memory LayerNorm → cross-attention更新 → 锚点累计位移 → 最终XYZ`。P2 Head是anchor级，不能沿用旧逐点Head索引；null、padding、无效手与像素不计入统计。加入相同embedding不会改变绝对差值，可能改变相对比例及后续LayerNorm响应；不同层范数不能直接比较为信息传递率。

正常AR下后步变化同时包含条件直接作用和前步预测变化。增加一次仅用于解释的固定query／固定原始AR状态重放，分离融合直接响应与递推反馈；重放使用真实条件基准预测，不读取目标GS，也不把该重放当正式预测性能。

逐horizon报告：

- 当前unsquared双向CD（mm）、RGB/Alpha MAE、`ΔCD=CD(扰动)−CD(真实)`；正值表示真实条件在该对照更有帮助。平均前先保留每窗口／episode及各步，H15标h1/5/10/15。
- 扰动前后预测位移RMS、真实条件预测相对初态的位移RMS、GT集合／区域运动与复制初态误差。预测沿自己的模板ID可算位移；GT不得按行构造“真实粒子轨迹”。检查运动端、较少运动端与局部铺展，区分真实形变、静态复制及错误运动。
- ΔCD绝对值优先；相对变化仅在非零且足够稳定的分母下给出，并保留分母。不要把某个百分比解释为动作／触觉独立贡献占比；两路相互作用，不能强制加和为100%。
- 对同层同checkpoint记录masked RMS响应，attention可作描述性辅助。敏感不等于有益，不敏感也不能单凭特征范数证明编码丢信息；必要时再提小读出器诊断，不在本轮增加训练。

H5与H15分开报告；共同起点的前五步可以作描述性比较，但两个完整训练实验的窗口数／优化次数不同，不能当作等预算的架构消融。历史400-step拟合不能代替这次完整训练条件利用结论。

## 6. 后续实验顺序与最低工程约束

建议顺序：**冻结完整训练P2的诊断 → 共同确认第一项候选 → 单项配对实验**。当前优先讨论A（7→11维）：数据已具备、增量成本小、归因直接；它是显式运动表达假设，不是已证明根因。若完整训练证据显示动作编码响应充分而融合几乎不响应，应优先考虑C，不能预先保证A更有效。B单独验证空间触觉信息的价值，不与A/C同时开始。

| 候选 | 最小对照 | 后续可能涉及的文件（本轮未改） |
| --- | --- | --- |
| A | P2 7D / 11D；触觉和融合不变 | `training/model.py`、`training/train.py` 配置／恢复及 `tests/training/` |
| B | 原P2 / 等patch无XYZ / 等patch含XYZ | `training/model.py` 编码与身份配置、`training/train.py`、`tests/training/`；现有数据接口足够 |
| C | 原P2 / anchor内容融合 / anchor相对空间bias | `training/model.py`、必要的训练内融合模块及配置／测试；复用 `gs_decoder.py` 池化和RBF |
| 冻结诊断 | 同checkpoint不同条件，无optimizer | `experiments/` 下专用诊断，复用 `evaluate.py`、renderer、loss；新报告用独立audit目录 |

后续获准的架构验证先沿用已审计固定窗口、相同H/seed/初始化共有权重/相机/loss/400更新预算，两窗口交替各200；不改变监督来追求收益。001原窗口ep1/85、ep2/73，051 ep0/109、132；只有在用户确认适用scene后执行。固定窗口检查表达能力与正确性，完整训练的条件利用／泛化需要单独同协议对照，不因单个窗口改善立即采用。

必须通过：旧3D/7D及H5/H10/H15 checkpoint原语义与输出回归；新模式配置不允许静默恢复；缺失初态、有效零触觉、无空间定位、全无效条件/null、padding、独立左右手；修改未来条件不影响较早预测；修改GT不影响预测；完整跨步梯度；A新增列/C1相对bias的初始化等价性。B只保证关闭时旧行为及B0/B1配对，不虚报新增token与基线等价。

未来CUDA验收须另行调度，记录H15前向／完整反向峰值 allocated/reserved、每秒窗口数及渲染在总耗时的占比；不靠隐式detach、缩短H或隐藏token减少来规避资源问题。正式研究采用与否由负责人决定，未获下一步授权不实现这些候选。

## 7. 本轮交付状态

已完成最新main与本地状态核对、安全建分支、源码／本地既有实验只读审计，以及本提案整理。仅本文件纳入提交；原有修改和未跟踪文档保留。源码与本次基准无差异，因此本轮不重复执行模型单测、不声称新架构已通过测试。没有提交Slurm任务、触碰服务器工作区、同步研究分支到运行目录，或修改现有H5/H15日志和checkpoint。

第一项实际架构改动待与用户共同确认；本文件不能视为三项候选均已获准实施。
