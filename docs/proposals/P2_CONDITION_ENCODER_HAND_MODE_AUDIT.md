# Step 2：Action–Tactile Condition Encoder 与单双夹爪 Masking 设计验收

审计日期：2026-10-27。研究分支：`codex/p2-condition-fusion-v1@7af22153d9bc6c4c4f1c29da8e7da6511831afea`；正式基线：`main@64b1320d3718e7d05d590ce854b8aeb05f14236f`。两者 `scripts/` 无差异。

本轮只有源码／元数据只读检查、轻量 CPU 统计与张量验证，以及本文件。没有实现新模型、修改重构数据、加载 GS／checkpoint、渲染、申请 GPU、提交 Slurm 或执行 Git 提交／推送。文中分为 **代码事实、实测事实、已确认规则、候选设计、未验证项**。

## 1. 结论与本轮确定的边界

**设计结论：按下述修正后的 Hand Mask 规则通过设计验收；新 Encoder 的学习效果与运行性能尚未验收。自动屏蔽静止手不采用。**

用户在本轮审计中明确确认：当前 **001-rope、008-pink-cloth、051-cube** 的 **ep0–4 为右手操作，ep5–9 为双手操作**。以此作为这些 episode 的操作模式约定：

| Episode | 保留的手 `[左,右]` | 确定处理 |
| --- | --- | --- |
| 0–4 | `[False, True]` | 左手 13D Action、30 个 Controller Points、两路 Tactile 和两个整指备用 Token 全部屏蔽 |
| 5–9 | `[True, True]` | 双手独立编码；不因一只手静止或触觉为零而屏蔽 |

**被屏蔽左手的触觉按本轮已确认的无关／噪声观测处理，不因其数值非零而恢复输入，也不通过 Fallback 绕过屏蔽。** 原始文件仍原样保留，屏蔽属于后续模型输入构造规则。

这项确定性约定取代“仅凭元数据不能决定是否去掉未记录手触觉”的待议状态。它是用户确认的 episode 模式，不是从触觉幅值或运动阈值推导出的标签。不得将 `episode < 5` 擅自推广到其他未审计 scene；后续实现应按完整 scene 名称核对已确认范围，并与 metadata 映射交叉校验。新 scene 没有规则时应明确未配置，不能猜测左右手。

本阶段保留 Controller 7D、2 个夹爪状态槽位、60 个联合工具槽位，并预留 4 个整指槽位。只改变 Condition Encoder 与必要的输入 Mask 传递；不改变 GS Encoder、Cross-Attention、共享锚点 Decoder、Loss、时间协议或数据划分。此前 `P2_CONDITION_FUSION_V1.md` 的 7→11D、patch Token、anchor 条件融合是旧候选，本轮不实施。

## 2. 现有仓库的代码事实

### 2.1 手的身份、参与和静止是不同概念

| 概念 | 当前证据／字段 | 能说明什么，不能说明什么 |
| --- | --- | --- |
| A：物理身份／被记录的夹爪 | `metadata.robot_slot_hands`、`physical_hand_to_robot_slot`、`binding_basis` | 确定已发布 robot slot 对应哪只物理手；不是逐帧参与、接触或硬件库存标签 |
| B：实际操作参与 | 现有文件没有专门的逐帧标签；本轮用户确认上述 episode 模式 | 当前三 scene 可以按已确认模式做整手屏蔽；不能从 `action_valid` 直接解释为参与 |
| C：静止但固定／接触物体 | 位姿、开口、触觉提供观测，没有可信的接触／固定真值 | 静止可能仍提供边界条件；不等于未参与，不自动去掉 |

`scripts/dataloader/geometry.py::hand_slots` 校验手顺序、四路顺序及两个映射的一致性。输出始终以物理 `[左手,右手]` 排列，不能把 robot slot 0 写死成左手。`splits.py::SINGLE_HAND_TRAIN` 是训练课程／episode 子集，不是逐帧活动标签；不得用它排除正常 train/validation 中的 episode。

### 2.2 Action、Controller 与各类 Mask

| 字段／源码 | 当前语义 |
| --- | --- |
| `actions` → `action_state`；`geometry.py::action_conditions` | 实测世界位置 3、按行旋转矩阵 9、原始开口 1，共 13D；它是状态观测，不是目标命令、速度或力 |
| `action_valid` | robot slot 存在、位姿／开口有限、刚体检查通过；刚体容差 `1e-6`。有限的越界开口不直接令 Action 无效 |
| `controller_mask` | `controller.npz` 的 `controller_valid_mask`；要求相应 Action 有效，坐标有限。缺失手填 0、Mask=False |
| `valid_mask` | 每路取 GS 时刻之前最近 raw，年龄 0–75 ms（沿用暂定值），raw／median 等有限，取前 12 行，第 31 列无效；`tactile=max(raw-median,0)`。零值可以有效 |
| `tactile_spatial_mask` | `valid_mask & 对应手 action_valid & world_xyz 有限`；是读数能否空间定位的资格，不是接触区域真值 |
| `tactile_xyz` | 用当帧夹爪状态计算的传感器表面采样位置；不是物体接触点，也不是每个 raw 触觉采样时刻单独插值得到的位姿 |

`dataset.py::_episode_conditions / __getitem__`、`collate.py::fixed_fields / check_batch` 保持四路触觉顺序。两个 Pose 缺失通路与二维触觉有效性原本独立：**现有实现允许缺失手的二维触觉继续有效**，相关单测特意覆盖了该行为。新方案只在输入层增加整手门控，不改动保存数据的 `valid_mask` 原始含义。

`processor.py::_transform_batch` 对 GS、Controller、`tactile_xyz` 使用相同 scene 坐标归一化；开口除以场景尺度，旋转矩阵不缩放，触觉除以 100，Mask 与微秒年龄保持其语义。008 在本轮用于数据语义审计，不据此宣称它已具备正式训练所需的全部处理配置。

### 2.3 当前 Condition Encoder

`scripts/training/model.py::controller_inputs / encode_conditions`：

- 夹爪状态 MLP：13→128，每时刻 2 个 Token。
- Controller MLP：7→128，每时刻 60 个 Token。7D 精确为 `[q_j, q_j-q_0, endpoint_valid]`；第 7 维是 **当前和初态均有效**，不是单独的当前有效标志。初态无效时仅差分和末位为 0，当前有效坐标仍可形成 Token；当前 Token 有效性另由 `controller_mask` 控制。
- 触觉 MLP：1153→128，每指 `384 值 + 384 valid_mask + 384 spatial_mask + 1 age`。`tactile_xyz` 尚未使用。
- Token 顺序为 `[2 状态, 60 Controller, 4 整指]`，共 66 个固定槽位。加入身份和时间 Embedding 后再次按 Token Mask 置零；Cross-Attention 用 key padding mask 排除无效条件，并有独立 null token。
- 整指 Token 当前只由 `valid_mask.any()` 决定，不受整手模式约束。正常单手帧若四指都有二维有效样本，当前最多 **35** 个有效 Token（1+30+4），不是 33。
- 第 k 个预测 `t+k` 只读取 `t…t+k-1` 的条件；编码器当前不跨时间混合。这条因果性应保持。

## 3. 真实数据 CPU 只读审计

数据根目录：`/data1/userdata/tcweng/projects/touch4d/datasets/touch4d-v1/`。检查三个 scene 全部 30 个 `episode_0…9`，每个只读 `metadata.json`、`controller.npz`、`tactile.npz`、`timestamps.npz`，并查看源 processed `metadata.json` 的字段。没有读逐帧 GS、图片或深度。

服务器使用现有 `.venv/bin/python -B`，单线程 CPU。调用的 Geometry 来自只读 P2 worktree `ca84a6e12035ddb85d15d1251852ca99d69495bf`；所涉 `geometry.py / dataset.py / collate.py / model.py / processor.py` 与本地对应内容的 SHA256 在统一换行后全部一致。没有同步或修改服务器工作区。

### 3.1 分布、映射和完整性

| Scene | 右手单槽 episode／帧数 | 双槽 episode／帧数 | 总帧数 | 有效已发布手-帧数 | 缺失左槽但有正触觉的帧数 |
| --- | --- | --- | ---: | ---: | ---: |
| 001-rope | ep0–4／1392 | ep5–9／1301 | 2693 | 3994 | 1392／1392 |
| 008-pink-cloth | ep0–4／2065 | ep5–9／2149 | 4214 | 6363 | 2064／2065 |
| 051-cube | ep0–4／1177 | ep5–9／1238 | 2415 | 3653 | 1177／1177 |

全部右手单槽为 `robot_slot_hands=[right]`、映射 `{left:null,right:0}`；全部双槽为 `[right,left]`、映射 `{left:1,right:0}`。这与用户确认的 episode 模式一致，但实现仍应读取并校验 metadata，不能跳过身份映射。

30 个 episode 的帧数、Action 编码、刚体有效性、Controller 与 Action Mask 关系、无效 Controller 清零、触觉有限非负及无效列检查均通过。已发布手的位姿均有效，Controller 未出现额外无效点。四路时间索引均重新按 `searchsorted(..., side=right)-1` 核对，年龄一致且有效样本不超过 75 ms；实测有效年龄范围分别为 001：12–74947 μs、008：0–42959 μs、051：12–73604 μs。各 episode 帧间隔中位数约 32.0–32.2 ms。

120 个被读元数据／NPZ 文件的读前、读后 SHA256 全部一致。统计不涉及 checkpoint、图像、报告或训练数据改写。

重构 metadata 中没有 `active_hand / participation / contact` 之类的参与标签；源 processed metadata 字段为 `WH, cameras, end_frame, fps, frame_num, intrinsics, start_frame`，也没有这种标签。本轮操作模式的确定依据是用户确认，不是发现了新的自动标签。

左槽缺失时，左手正值像素-帧计数分别为 180215、232518、99347；这些数值**不构成保留输入的理由**，按确认模式全部屏蔽。它们只解释为什么旧代码可能让未参与手进入整指通路，不用于设置接触阈值。

### 3.2 双手中的低运动与触觉

双手 episode 中，未找到任一只手相邻两帧 `T_worlds` 和 `openings` 同时逐元素完全不变的情况。不能据此认定两手一直主动运动：跟踪噪声和开口抖动也会造成非零变化。

为只读展示低运动风险，对双手连续 11 帧统计 10 个相邻步的平移路径长度、旋转测地角累加和绝对开口变化累加；在每个 scene 中分别求三项的经验分位排名，以三项排名最大值较小的区间取非重叠示例。此方法仅选审计示例，不产生线上活动标签、训练窗口或运动阈值。

| 示例（帧号从 0 编号） | 所观察手 | 10 步累计平移／旋转／开口变化 | 另一手同项 | 有正值的观测帧 | 有效触觉 P99／最大值 |
| --- | --- | --- | --- | --- | --- |
| 001 ep8，60–70（0.332056 s） | 右 | 6.408 mm／0.989°／2.609 mm | 左：34.992 mm／4.665°／8.002 mm | 11／11 | 6／42 |
| 051 ep9，122–132（0.332032 s） | 右 | 1.080 mm／0.496°／1.898 mm | 左：1.289 mm／0.634°／2.659 mm | 11／11 | 30／58 |

数值为 `max(raw-median,0)`、尚未除以 100，不能当作校准后的力。001 示例说明两手运动幅度可以明显不对称；051 示例是两手均低运动，不能称为“一手动、一手静止”。这些区间都有持续正读数，但未查看 RGB／人工接触标签，**“严格静止”“正在固定物体”“非零值全部来自接触”均未验证**。

对 ep5–9 不做自动静止手屏蔽。即使未来确认某只手静止，它的夹持位置、旋转、开口、触觉分布仍可能描述固定端和力传递边界，删掉可能使运动端／稳定端关系丢失。

## 4. 候选 Condition Encoder：输入、几何聚合与 Shape

以下是待实现设计，不是已有代码。H 保持可配置 5／10／15；每个条件时刻分别处理，无跨未来时间归一化或池化。

| 环节 | 输入／输出 Shape | 处理 |
| --- | --- | --- |
| 整手规则 | `hand_keep [B,H,2]`，bool | 由已确认 scene/episode 模式产生，整段恒定；与 metadata 校验。建议后续作为显式输入传递，不从压力或窗口内运动反推 |
| Action State | `[B,H,2,13]` → `[B,H,2,128]` | 复用状态 MLP 与物理手身份，Mask=`action_valid & hand_keep` |
| Controller 原始坐标 | `[B,H,2,30,3]`；Mask `[B,H,2,30]` | 先整手门控，再构造当前坐标和相对窗口初态的位移 |
| Controller 7D | `[B,H,60,7]` → `[B,H,60,128]` | 保持现有 endpoint-valid 语义和控制点顺序 |
| Tactile 值／二维 Mask／空间 Mask | 各 `[B,H,4,12,32]` | 分别表示连续幅值、数据有效性、可定位资格；先应用整手门控 |
| 表面坐标／年龄 | `[B,H,4,12,32,3]`；`[B,H,4]` | 几何关联使用有效的 `tactile_xyz` 与 Controller；年龄沿用每指 `age_us/75000` |
| 同指几何权重 | `[B,H,4,15,384]` | 每指只向自己的 15 点聚合；无跨指／跨手连接；无效像素不进分母 |
| 像素特征（候选小宽度） | `[B,H,4,384,dτ]` | 小型共享编码器保留幅值、传感器局部三维位置与年龄；`dτ=32` 可作为首版候选，最终配置需显式记录 |
| 聚合触觉 | `[B,H,4,15,dτ]` → `[B,H,60,dτ]` | 额外保留可用标志、有效覆盖率与年龄，区别有效零值和无观测 |
| 联合工具 Token | `[B,H,60,128]` | 融合 Controller 动作几何特征与同点聚合触觉；没有可用触觉时保留动作特征 |
| 可选整指备用 | `[B,H,4,128]` | 只限 `hand_keep=True` 的手，后文定义 |
| 最终条件 | `[B,H,66,128]`；Mask `[B,H,66]` | 顺序仍为 `[2 状态,60 工具,4 备用]`；关闭备用仍保留槽位并置 Mask=False |

Controller 7D 本身就是局部 Action 几何表示；60 个工具 Token 是它与触觉的融合，不要求再把 13D 夹爪状态重复广播 30 次。13D 状态仍保留为两个独立 Token，提供旋转、开口等信息。

推荐最小融合形式：`J_p = C_p + has_tactile_p × F(C_p, pooled_tactile_p, coverage_p, age_p)`，输出 128D，再用当前 Controller Mask 置零。这里 `C_p` 为 7D MLP 输出；标志必须来自有效性，不从幅值是否大于零推断。新增融合残差输出可零初始化，便于后续归因，但由于原整指通路改变，**不能声称整个新 Encoder 初始化与旧 P2 数值等价**。

### 4.1 384 → 15 的含义与边界

固定物理顺序：

| 触觉通道 | Controller 区间（展平后的 60 点） |
| --- | --- |
| 左手左指 0 | `0:15` |
| 左手右指 1 | `15:30` |
| 右手左指 2 | `30:45` |
| 右手右指 3 | `45:60` |

每指 Controller 网格为 `rows=(0,6,11)`、`cols=(0,8,16,23,31)`，按行优先。触觉局部 Z 为 `0.0235−0.002r`，Controller 为 `(0.025/12)(11.5−r)`，二者不完全相同，不能将某个 Controller 直接视为同索引触觉像素。

关联应使用真实几何距离，例如候选固定距离核：

`K_pi = exp(−||q_p−x_i||²/(2σ²))`

`E_pi = same_finger(p,i) & controller_keep_p & spatial_keep_i`

`W_pi = E_pi K_pi / sum_j(E_pj K_pj)`；分母为 0 时整行置 0，并令 `has_tactile_p=False`。

σ 是传感器／控制点网格的聚合尺度，应依据间距选取并在实现前冻结；不是接触幅值或运动阈值。若距离使用场景归一化坐标，σ 也必须除以同一场景尺度；不能把米制带宽直接用于归一化距离。本轮没有确定生产核参数。CPU 张量验证使用的 0.01 m 仅是测试夹具。

有效列实际最多 31×12=372 点，384 是存储槽位。**触觉第 31 列屏蔽，不代表 Controller 的 col31 点无效**；后者仍可聚合同指 col30 等邻近有效像素。六个真实 episode 的 51 个可定位指-帧验证中，世界坐标距离与夹指局部距离的最大差异约 `2.87×10⁻⁸ m`。同指刚体变换保持距离，因此可缓存局部距离表，逐帧应用有效性并重新归一化；不能缓存去掉动态 Mask 的最终权重。

空间 Mask 只用于关联资格、归一化、统计覆盖率，不再把其 384 个布尔数值直接拼入触觉 MLP。几何位置描述“这个传感器点在哪里”，**不声称该点真的接触物体**；不按压力大小选择邻域，不增加接触阈值。

## 5. 单双手与备用通路的完整 Mask 规则

记 `g=hand_keep`，`g_f` 为映射到四指的同一手门控。所有门控必须同时作用于输入数值、融合资格和最终 Token Mask。

1. **Action：** `a = action_valid & g`。无效 13D 输入先用 `where` 清零，再编码；加完身份／时间 Embedding 后再清零。不能用 `NaN*0`。
2. **Controller：** `c = controller_mask & g[...,None]`。先清理坐标，差分只在 `c_j & c_0` 下计算；当前点有效、初态无效时保留当前坐标，但差分为零且第 7 维为零。当前点无效则整个工具 Token 无效。
3. **二维触觉：** `v = valid_mask & g_f[...,None,None]`；数值先按 v 清零。年龄仅在该指存在有效二维样本时可输入，否则置零。不能用 `tactile>0` 决定 v。
4. **空间触觉：** `s = tactile_spatial_mask & v`。空间坐标按 s 清零；与 c 共同决定几何权重。当前 75 ms 规则已经包含在 `valid_mask`，不因插值或 fallback 放宽。
5. **工具融合：** 有效 Controller 无触觉时继续提供 Action；有效零压力时 `has_tactile=True`。触觉有值但 Controller 无效时不得创建虚假世界坐标工具 Token。
6. **最终 memory：** 在身份／时间编码后再次应用 `[a, c.flatten, fallback_keep]`，并传入 Cross-Attention key padding mask。所有真实条件无效时继续使用原 null token；null 不代表任何一只手。

### 5.1 四个备用 Token 是否必要

建议保留固定四槽位与独立开关，首个候选采用 **fallback-only**，不默认重复加入四个全局触觉摘要：

- 对每个像素，计算其是否已经有同指有效 Controller 的空间关联；`residual_2d = v & ~assigned_to_any_valid_controller`。
- `fallback_keep = residual_2d.any()`，输入只使用这些尚不能定位的有效二维读数、相应有效 Mask 和年龄，不伪造 world XYZ。
- 可以沿用整指展平通路的形式，但输入改为 `384 值 + 384 二维有效 Mask + 1 age = 769D`，没有 384D spatial 数值输入；未参与手始终没有 residual，始终不可形成备用 Token。
- 保留手的位姿暂时无效时，其 Action 和 Controller 失效；若二维触觉仍有效，允许通过备用通路表达“有读数但未定位”。这是保留手的异常观测处理，与 ep0–4 被屏蔽左手完全不同。
- 正常几何完整时，单手最多 31 个有效 Token，双手最多 62 个；若另做“保留全部整指摘要”的独立对照，分别最多 33、66 个。槽位均为 66，不改变 Cross-Attention 结构。

本次真实数据中已发布位姿与 Controller 均有效，没有提供“保留手位姿失效”的真实样本。备用通路的逻辑可用合成 CPU 测试检查，但其实际训练收益 **未验证**。若后续取消它，必须把丢失这类二维观测的代价作为明确消融，而不是默默改变有效性含义。

### 5.2 情况表

| 情况 | Action／工具 Token | 触觉／备用 |
| --- | --- | --- |
| ep0–4 左手 | 全部数值置零，Token=False | 两路值、几何、年龄清零；聚合与备用全部禁用 |
| ep0–4 右手、ep5–9 任一手正常 | 独立按当前有效性保留 | 连续触觉聚合，不看正值比例决定是否保留 |
| 双手中一只静止 | 保留 | 保留有效零值或非零值，不自动关闭 |
| 保留手当前位姿失效 | Action／Controller 无效 | 不能做空间聚合；可用的二维观测仅进入该手 fallback |
| Controller 初态失效、当前有效 | 当前 Token 保留，差分无效 | 当前定位有效的触觉仍可融合 |
| 时间过期／无过去触觉 | Action／Controller 可继续有效 | 触觉聚合和备用均不能使用该路读数 |
| 所有手或所有条件不可用 | 真实条件 Token 无效 | 原 null token 保证 attention 有合法 memory |

## 6. 运动判断的可行性与信息损失

**不将自动活动检测纳入首版。** 本轮已有用户确认的 episode 模式，没必要用运动阈值重新判断。只屏蔽已确认缺失／未参与手是严格对照，双手中的静止手保留。

如果以后有独立研究需求，至少应同时观察：`||p_j−p_(j−1)||/Δt`、`acos(clamp((trace(R_(j−1)^T R_j)−1)/2))/Δt`、`|w_j−w_(j−1)|/Δt`。旋转用相对旋转角，开口用原始未裁剪值，不能只看 XYZ；抖动、时间间隔和跟踪质量均需考虑。三者是运动证据，不是操作参与真值。

- 只能使用当前与过去，不使用未来速度、居中滤波、整段最大运动或未来触觉。一次编码整个条件张量时也不能让早期 Mask 读取后期数据。
- 首帧没有历史则运动状态为“未知”，按已确认手模式保留，不能把零差分当成静止。当前 Dataset 没有额外 t−1 观测，不能假装有。
- 阈值若将来需要，应由人工确认的空闲／固定／运动片段和测量噪声分布提出，在训练范围审计后冻结，专门检查固定手的误删除；本轮不提供数值门槛。
- 自动删静止手可能减少无关状态和噪声，却可能删除固定端位置、夹持姿态、开口约束、持续压力以及释放前的状态。触觉非零与否也不足以补足参与标签。

当前确定规则的收益是消除 ep0–4 无关左手条件，并让单／双手 Mask 一致；这仍是设计动机，**不等于已证明性能提高**。几何聚合会把 384 位置压缩至 15 个局部表达，可能损失细小触觉分布；需要空间扰动对照检验，不能称为无损映射。

## 7. CPU-only 初步验收与成本

### 7.1 已执行的检查

| 检查 | 结果 | 证据范围 |
| --- | --- | --- |
| 三 scene、30 episode 元数据／轻量数组 | 通过 | 共 9322 帧；无 GS／图像读取；120 文件读前后 SHA256 一致 |
| 双手存在且运动不对称 | 有实例 | 001 ep8 60–70；不把低运动认定为严格静止 |
| 静止手确实在接触／固定物体 | 未验证 | 没有逐帧接触标签，本轮无 RGB 人工核验；不作自动删手依据 |
| 现有相关单测 | **9 passed，0 skipped，0 failed** | Geometry 3 项，Dataset 5 项，Controller 7D 1 项；0.714 s 测试执行时间 |
| 候选路由的纯 CPU 张量断言 | **16 passed，0 failed** | 合成 `B=2,H=3`，单／双手、NaN 清理、Mask、有效零值、fallback、梯度与逐时刻独立性 |
| 真实同指距离表验证 | 通过 | 001／008／051 ep0、ep5 各取首／中／末帧，51 个有效指-帧，最大距离差约 2.87e−8 m |
| 新 Encoder 学习收益、完整集成、H15 显存／吞吐 | 未验证 | 尚未实现，不使用已有小测试替代 GPU 或模型性能验收 |

9 项复用测试均使用内存夹具，没有生成测试数据文件。运行方式：本地 `D:\Anaconda\envs\pytorch\python.exe -B -m unittest`，环境设置 `PYTHONDONTWRITEBYTECODE=1`、`OMP_NUM_THREADS=1`、`MKL_THREADING_LAYER=SEQUENTIAL`、`CUDA_VISIBLE_DEVICES=`。具体选择：

- `scripts.tests.dataloader.test_dataloader.GeometryTests`（全部 3 项）；
- `DatasetTests.test_all_scenes_and_metadata_reversed_slots`；
- `DatasetTests.test_missing_hand_and_zero_response_remain_valid_in_2d`；
- `DatasetTests.test_invalid_pose_is_masked_without_dropping_window`；
- `DatasetTests.test_saved_no_past_stale_and_duplicate_timestamps`；
- `DatasetTests.test_75_ms_boundary_is_inclusive_and_stale_mask_is_enforced`；
- `scripts.tests.training.test_training.TrainingTests.test_relative_controller_features_masks_and_scale`。

16 项临时内存断言覆盖：单／双手状态与 Controller 数量；同指权重归一化；无效触觉 col31 与有效 Controller col31 的区别；有效零值；不同指隔离；整手关闭无关联；Embedding 后清零；摘要模式单手 33／双手 66；正常定位 fallback 关闭；保留手失效位姿的双指 fallback；整手关闭后 fallback 禁用；全无效有限零输出；无效 NaN 清理；有效值可微与无效值梯度为零；改变后期触觉不改早期聚合；初态失效不删当前有效 Controller。此原型没有实现或训练网络，也不证明完整预测模型已经通过因果性回归。

### 7.2 初步算力判断

每个条件时刻有 `4×15×384=23040` 对几何权重；B=1、H=15 时，FP32 权重约 **1.32 MiB**。若逐像素特征宽度为 32，特征约 **2.81 MiB**；宽度 128 则约 **11.25 MiB**，均不含梯度、隐藏层和优化器。

应先编码 384 像素再聚合，复用同指固定距离表，避免展开 `[B,H,4,15,384,128]` 的逐对特征；这个张量单份就约 **168.75 MiB**。不得把上述原始张量估算当成实际训练峰值。

条件槽位上限仍是 66，H15 的 Cross-Attention 稠密维度不增加；Mask=False 不会自动让标准 dense attention 节省相同比例计算。新的像素编码／聚合会增加计算，是否比原 1153D 整指 MLP 更划算尚未实测。CPU 已验证路由和数据规模可处理，尚不能宣称 GPU 性能通过。

## 8. 后续最小对照与实施前约束

以下均为待授权的未来工作，本轮不实施。保持同一 P2 GS 主干、H、初始化共有权重、数据划分、优化器、Loss 和预算，不能同时改动作维度、Cross-Attention 或 Decoder。

| 顺序 | 对照 | 单独回答的问题 |
| --- | --- | --- |
| A | 原 P2 编码／仅增加已确认 Hand Mask | ep0–4 去掉无关左手输入本身的影响；原版只作为历史诊断，修订设计始终屏蔽左手 |
| B | A／触觉 MLP 去掉 384D spatial 数值，其余不变 | 去掉显式 spatial 向量的影响；空间有效性仍保留 |
| C | B／384→15 几何聚合与 60 个 Action–Tactile 工具 Token | 局部融合是否有益；先统一整指摘要设置，避免与备用开关混杂 |
| D | 相同 C 下，整指摘要／fallback-only／无备用 | 是否需要整指摘要、异常定位时是否需要备用；所有模式都受 Hand Mask 约束 |

若需要判断新通路是否使用空间信息，可在冻结同一模型时做触觉置零及同指固定像素排列扰动，保留有效 Mask／年龄；只作诊断，不把人为扰动的损失变化称为真实贡献百分比。联合工具 Token 应区分动作特征、触觉更新与最终响应。

评价单手与双手分开，双手中低运动端单列；同时看预测几何／渲染质量与条件响应，不能只要求敏感性更强。固定窗口只检查表达与优化，泛化需要正常验证集。008 的统计不能自动改变当前训练 scene 范围，原 train/val/test 名单不改。

后续实现最小接口工作：

- Dataset 从已确认模式产生并校验 `inputs.hand_keep [H,2]`；Collate 堆叠为 `[B,H,2]`，Processor 检查后原样传递，不对 bool 归一化。不更改 metadata、重构 NPZ 或保存的原始 Mask。
- Condition Encoder 使用它统一屏蔽 Action／Controller／触觉／备用；保留 7D endpoint 规则、已有时间前缀与 null。当前代码没有 `hand_keep`，这一点是新增要求，不是已有能力。
- 配置中显式保存新 Encoder 模式、整手策略、像素宽度、距离核／几何带宽、备用模式。旧 checkpoint 仍加载旧 Encoder，不静默切换语义；新旧通路参数 shape 不一致时不能强行恢复。
- 新增集成测试应覆盖本轮 16 项边界、全部支持 H、左右 slot 反转、缺失／无效初态、未来条件扰动、全条件失效，以及旧 checkpoint 回归。禁止以代码已审计替代这些实现测试。

**可进入下一步的范围：** 用户审核本文件后，先落实已确认的 Hand Mask 与保持 7D 的几何局部触觉融合方案；具体聚合核／带宽和备用配置应在实现配置中明确冻结。**不进入的范围：** 自动静止手筛选、未来信息推断参与状态、动作 11D、GS／Cross-Attention／Decoder／Loss 改造。

本轮最终产物仅为本文件。已有 `MODEL_ROADMAP.md` 修改、未跟踪文档、代码、数据及实验结果保留；没有提交、推送或 PR。
