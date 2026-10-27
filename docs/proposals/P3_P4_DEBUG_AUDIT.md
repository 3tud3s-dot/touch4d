# P3–P4 Debug Audit：近静态预测根因

## 冻结范围与安全

基准 `codex/p3-p4-spatial-dynamics-v1@400b4c389325efe18d4d9a54eb5412e722ab22cc`；GitHub main `64b1320d3718e7d05d590ce854b8aeb05f14236f`（2026-10-27 通过 GitHub 连接器复核，Git HTTPS 当时不可达）。独立分支 `codex/p3-p4-debug-audit-v1`，不修改原 Roadmap、模型或历史结果，不创建 PR、不合并。

- 本地源码：`D:\Amax\Projects\touch4d-worktrees\p3-p4-debug-audit-v1`。
- 本地报告：`D:\Amax\Projects\touch4d-experiments\p3-p4-debug-audit-v1`。
- 服务器结果：`/data1/userdata/tcweng/experiments/touch4d/p3-p4-debug-audit-v1/`。
- 服务器执行：`/data1/userdata/tcweng/projects/touch4d-worktrees/p3-p4-debug-<SHA>/`，每个提交固定独立路径。
- 用户交互式 `srun` Job **30903**（1 GPU）受保护，不进入或复用。本轮最多新增两个并行单卡作业。先单卡诊断，再两个配对作业；不自动重试或改变研究配置。

## 假设与执行顺序

H1 数据/对齐/Mask/单位/评价错误；H2 条件传递不足；H3 Loss/优化倾向近静态；H4 固定 D0 表达或优化瓶颈。先 CPU 审查，再 A2/B/C 真 CUDA，确认无应立即修复的确定性错误才运行 D。相关性不等于根因证明。

| 阶段 | 固定数据与控制 | 输出 |
| --- | --- | --- |
| A | CPU：ep2/start73、ep3/start115、ep1/start85，H5；真实时间、几何、Mask、归一化与完整代码链 | 数据契约、源哈希、确定性测试与代码行号 |
| A2 | 原 S0、S2-C1 seed42 的 best/last，三个窗口只读推理；H5 主诊断，H15 仅原 hold-last 外推前缀核验 | Anchor/GS 位移、非刚性残差、独立点集 CD/质心/形状、GT/Pred/Copy 三相机 |
| B | 原 S0 last，ep2/start73；每个分量均恢复同权重及原 AdamW 状态 | raw/clip=1/actual update、参数范数、梯度 cosine、预测变化；单位保持不变 |
| C | ep2/start73 H5；只优化 128 Anchor × 5 × 3 的累计位移，零初始化；初态固定 W；AdamW lr=1e-3、wd=0、clip=1、400 updates | Copy、GT 集合质心平移、D0 CD-only；另以相同预算 Full Loss 排除仅几何监督的歧义；记录 0/100/200/400 |
| D | ep2/start73 H5；S0+A0/T3+D0/F0；独立从零 seed42；AdamW lr=1e-4、wd=1e-4、clip=1、batch1、400 updates | E1 CD-only vs E2 Full；0/100/200/400 指标、参数更新、条件冻结/置零、五步图像 |

Oracle 的较高参数学习率是预先固定的直接几何优化设置，不与网络优化器的学习速度等价，不搜索或延长预算。D 的两组初始化哈希、样本、相机和成功更新数完全一致。全程使用原正式 objective、renderer、processor；Full 权重为 CD=1、RGB=1、Alpha=.1、Scale=.01，D0 的初态属性继承使 Scale 项为零。CD-only 仍记录相同渲染指标，渲染项反向权重为零。

Val ep3 仅只读诊断，不训练、不选超参数。未来 GT 仅作正常监督或明确标记的离线 Oracle，不能输入正式预测模型。独立重建 GS 无跨帧固定 ID；GT 质心变化及集合误差不能解释为材料粒子真位移。单窗口成功也不能证明条件动力学或泛化；时间编码记忆是可能的解释。

## 判据与证据

- 确定性 Bug：必须具体代码位置、最小失败复现、修复后通过及旧行为回归；无证据则写“未发现确定性 Bug”。不因性能不佳重写网络。
- D0 Oracle 与 CD-only 网络能运动：基本表达/梯度链可用；Full 更差时再结合独立梯度方向与实际更新评估 Loss 竞争。
- Oracle 能而两网络不能：条件映射、优化、初始化优先；Oracle 不能则仍需区分 GT、单位及有限预算，不直接判定不可表达。
- 两网络能拟合而原多窗口近静态：训练窗口运动覆盖、采样与可辨识性值得优先检查，不宣称已证实。
- 原始梯度、裁剪梯度、AdamW 更新分开；压力幅度、信息可恢复性和条件有效收益分开；输出变化不等于预测改善。

## 实际进度与结果

本轮完成 CPU 审计、冻结 checkpoint 验证、梯度/实际更新审计、两种 D0 Oracle 和两种 400-update 单窗口训练。**未发现原模型的确定性代码 Bug；D0 和两种 Loss 均能拟合该诊断窗口。原多窗口训练的近静态退化得到复现，但尚未唯一定位其因果根源。** 以下均为本轮实测，不能外推为跨窗口泛化或物理准确性证明。

### 执行、资源与验收

| 阶段 | 实际完成内容 | 状态与证据 |
| --- | --- | --- |
| A | 三个窗口的帧号、因果条件、Hand Mask、几何归一化、T3 路由和 D0 插值；原 24 窗采样统计 | 完成；`data_audit.json`、`sampling_audit.json` |
| A2 | S0/S2-C1 best/last × 三窗口，共 12 组；另核验两个 last 在原 24 个已见训练窗口的结果 | 完成；`diagnosis-v2/checkpoint_results.json`、`checkpoint_seen.json` |
| B | CD/RGB/Alpha/Full 的 raw/clip/AdamW 更新、参数相对变化和梯度 cosine | 完成；`gradient_audit.json`，恢复原权重与优化器状态 |
| C | CD-only 与 Full D0 Anchor Oracle，各 400 次；Copy 与 GT 集合质心平移参考 | 完成；`oracle_results.json`、`diagnosis-v2/oracle-{cd,full}/` |
| D | E1 CD-only、E2 Full，各 400 次成功更新，0/100/200/400 验收 | 完成；`E1/`、`E2/`，初始化哈希一致，checkpoint 恢复输出逐元素一致 |
| E | 未触发旧模型 Bug 修复；修复一次新审计导出器的环境依赖问题 | 原模型、Loss、Dataset 及旧实验实现未改；失败证据保留 |
| F | 数值汇总、五步 GT/Pred/Copy 几何与三相机图像、根因证据表 | `report.html`、`results.json`、`training_comparison.csv`、`acceptance.json` |

科学实验源码为 `6651b791818a65859f3edfed30228ec6bd43b05c`；补充已见窗口审计源码为 `7ad9ed7b755b88431c249f806dd97b6c2ee79ff8`。最终报告提交与实验执行 SHA 分开记录，不将后续文档提交冒充训练版本。

| Job | SHA | 执行内容 | 终态 |
| --- | --- | --- | --- |
| 30904 | `37d5efd` | 首次诊断，输出 `diagnosis/` | FAILED：新导出器导入服务器未安装的 Pillow；保留日志与原目录 |
| 30905 | `6651b79` | 改用项目现有 PNG 编码器；A2/B/C，输出 `diagnosis-v2/` | COMPLETED，约 205 秒 |
| 30906 | `7ad9ed7` | 两个旧 last 的 24 个已见窗口推理 | COMPLETED，约 52 秒 |
| 30907 | `6651b79` | E1，400 次成功更新 | COMPLETED，约 122 秒 |
| 30908 | `6651b79` | E2，400 次成功更新 | COMPLETED，约 124 秒 |

30904 的失败属于**本轮新工具的导出依赖缺陷**，不解释已有 S0/S2-C1 的近静态预测。修复复用既有 `png_bytes` 并补充禁止 PIL 导入的测试，没有安装新环境或改变实验参数，重跑使用独立新目录。

- 本地单测：271 项，245 通过、26 跳过、0 失败；跳过为 24 项 h5py 依赖及 2 项显式启用的 CUDA 单测。服务器：271 项，269 通过、2 项 CUDA 单测跳过、0 失败；实际 CUDA/gsplat 前后向另已完成。
- RTX 5090，PyTorch `2.7.1+cu128`；诊断/E1/E2 峰值 allocated 约 2.705 GiB，reserved 约 3.049 GiB。真实反向、数值有限性、历史哈希与 checkpoint 恢复验收通过。
- 新增并行 GPU 最多 2 张。未进入、取消、重排或复用用户 `srun` 30903；审计作业终态后该用户作业仍正常运行。
- S0/S2-C1 四份 best/last 在读取前后 SHA256 一致；H5 与原 H15 hold-last 外推的前五步输出完全一致。H15 仍只是旧 H5 权重外推，不是正式 H15 训练。

### A：数据、前向与旧 checkpoint

三个诊断窗口均满足输入 `t`、条件 `t…t+4`、目标 `t+1…t+5`，例如 ep2 为初态 73、条件 73–77、目标 74–78。触觉沿用最近过去样本与 0–75 ms 时间有效规则。001 单手窗口只保留右手，具有每步 1 个夹爪状态、30 个工具 Token；T3 真实进入有效工具分支，Fallback 未误开。有效零触觉不被当作无观测。

场景归一化尺度为 `0.15703709051 m`，坐标往返最大误差约 `2.98e-8 m`，没有发现重复缩放。D0 的 128 Anchor 权重行和最大误差 `2.38e-7`；ep2 的 RBF 带宽约 11.165 mm。旧模型 Anchor 自身输出已很小，不是先产生大位移再被插值统一缩小。已审查路径未发现错误 detach、参数漏入 optimizer、梯度清零顺序错误或未来目标泄漏。

代码复核入口（原实验源码相对路径与行号）：`scripts/dataloader/dataset.py:326` 取初态与未来目标；`scripts/dataloader/collate.py:180` 分离初态/逐步目标 padding；`scripts/data_process/processor.py:175` 统一 GS/Controller/Tactile XYZ 归一化；`scripts/experiments/spatial_dynamics/conditions.py:20` 有效性清理，`:76` 同指聚合，`:94` 条件融合；`scripts/experiments/spatial_dynamics/model.py:150` 因果 rollout；`scripts/training/gs_decoder.py:26` 固定初态场、`:160` 位移插值；`scripts/training/losses.py:194` 原 Loss；`scripts/experiments/spatial_dynamics/runner.py:218` 清梯度、`:224` 反向、`:230` optimizer 更新。

**证据边界：**重构数据的 GS/robot 时间轴与来源一致，但没有独立的原始 robot 硬件采集时间轴可交叉验证，因此不宣称已完全排除硬件同步误差。RGB/Alpha 指标延续既定 GT GS 渲染及有效区域协议，不是新增 raw RGB 或遮挡真值审计。

| H5 平均 CD，mm | Copy | S0 best | S0 last | S2-C1 best | S2-C1 last |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train ep2/start73 | 3.5993 | 3.5406 | 3.6615 | 3.5521 | 3.6582 |
| Val ep3/start115 | 2.2410 | 2.2354 | 2.2526 | 2.2458 | 2.2520 |
| Train ep1/start85 | 2.5661 | 2.5387 | 2.5493 | 2.5223 | 2.5484 |

ep2/start73 和 ep1/start85 虽属于训练 episode，**不在旧实验冻结的 24 个训练窗口中**。因此补充原窗口名单上的只读推理，避免把未见窗口表现误写为训练集拟合情况：

| 原 24 个已见窗口，seed42 last | 平均 CD，mm | 相对 Copy 改善 | 改善窗口数 | h5 位移 RMS，mm |
| --- | ---: | ---: | ---: | ---: |
| Copy Initial | 2.03576 | — | — | 0 |
| S0 | 2.02803 | 0.380% | 6/24 | 0.3313 |
| S2-C1 | 2.02763 | 0.399% | 6/24 | 0.3386 |

原 24 窗 h5 GT 集合质心变化范数均值约 4.555 mm；这描述独立重建点集变化，不是材料运动真值。采样中 h5 Copy CD 中位数 1.861 mm、最大 12.410 mm，运动信息并非均一。已见窗口同样近静态，故退化不能仅归因于 H15 外推或本次诊断窗口未见。

### B：梯度与实际更新

S0 last、ep2/start73 的原始 CD/RGB/Alpha/Full 分别为 `0.0233163 / 0.1312908 / 0.1473006 / 0.1693372`，保留原归一化目标单位；Scale 为零。CD-only 与 Full 的全局原始梯度范数为 `2.582 / 27.163`，均 clip 至 1。Full 的 Condition 联合参数实际 AdamW 更新范数约 `0.01315`、相对原参数范数约 `1.35e-4`；Controller、Tactile/Fusion、GS、Structural Attention 和 Anchor Head 均有有效更新，不是只有 weight decay。

| 同一 batch、同一 checkpoint 的方向对照 | 数值 | 可支持的解释 |
| --- | ---: | --- |
| Head：CD vs RGB 原始梯度 cosine | +0.834 | 没有 Head 全局强对立的证据 |
| Head：CD vs Alpha 原始梯度 cosine | +0.783 | 同上，不能把 Full 大梯度直接解释为抵消 CD |
| Tactile Pressure：CD vs RGB / Alpha cosine | −0.651 / −0.787 | 此子模块存在局部竞争，不代表整网根因 |
| Condition：CD vs Full 实际 AdamW 更新 cosine | +0.960 | 原始梯度和带历史动量的实际更新需要区分 |

恢复原 1200-step AdamW 状态后，仅作一次 CD-only 更新，CD 从 3.66152 变为 3.66690 mm；Full 单步变为 3.67131 mm。这不是梯度链错误的证据：该 batch 未在旧冻结窗口中，且历史动量、有限步长和自适应预条件不保证单个 batch 的单步 CD 单调下降。也不能把 clip 系数当作 AdamW 更新量同比缩小的比例。

初始化首步上游梯度因零位移头而为零；第 2 次更新 GS/Controller/Structural/Fusion 等路径已经打开，但受两级零初始化影响，`pressure_encoder` 第 2 次仍为零。已记录的第 100/200/400 次及旧 checkpoint 审计中，其梯度与实际更新均非零，未发现永久阻断。当前记录未覆盖第 3–99 次的逐模块统计，不能指定压力分支首次打开的精确步数。

### C/D：Oracle 与严格配对的单窗口结果

以下均为 ep2/start73、H5；除 Copy/质心参考外均报告预定第 400 次结果，不择优挑选 checkpoint。Oracle 优化目标利用未来 GT，只能用于离线容量诊断。

| 方法 | 平均 CD，mm | 相对 Copy 降幅 | h5 位移 RMS，mm | h5 去全局平移残差 RMS，mm |
| --- | ---: | ---: | ---: | ---: |
| Copy Initial | 3.5993 | — | 0 | 0 |
| GT 集合质心平移 Oracle | 3.0488 | 15.29% | 9.3869 | 0 |
| D0 Anchor CD-only Oracle | 1.5705 | 56.37% | 14.2731 | 10.0539 |
| D0 Anchor Full Oracle | 1.6630 | 53.80% | 14.3564 | 10.2491 |
| E1：S0 网络，CD-only | 1.6731 | 53.52% | 14.6159 | 10.2645 |
| E2：S0 网络，Full | 1.7067 | 52.58% | 14.4759 | 10.4163 |

Oracle 明显优于仅质心平移；在这个窗口，D0 可以表达非零局部形变，固定 RBF 不是“绝对无法运动”的瓶颈。非刚性残差表示预测位移的空间差异，仍须结合 GT 集合与可视化判断，不能单独当作真实形变恢复量。

| 网络 | CD@0 | CD@100 | CD@200 | CD@400 | RGB MAE@400 | Alpha MAE@400 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 CD-only | 3.5993 | 1.7192 | 1.6802 | 1.6731 | 0.031247 | 0.022378 |
| E2 Full | 3.5993 | 1.7939 | 1.7302 | 1.7067 | 0.028902 | 0.018650 |

E2 的平均 CD 比 E1 高 2.01%，但 RGB/Alpha 分别低 7.51%/16.66%；两组 h5 质心误差分别约 1.550/1.152 mm。证据支持几何与渲染的有限折衷，不支持 Full Loss 必然锁死近静态解。两组每步 CD 和输出均不同；h1–h5 CD 分别为 E1 `[1.5845,1.6496,1.6994,1.6975,1.7343]`、E2 `[1.6029,1.6701,1.7347,1.7387,1.7870]` mm，没有将单帧重复冒充五帧。

最终条件消融仍极弱：动作冻结引起的 h5 逐点位移差范数的均值（不是 RMS）为 E1/E2 `0.00546/0.00570 mm`，触觉置零为 `0.000346/0.000355 mm`；五步平均 ΔCD 的绝对值分别不超过 `0.000110/0.000004 mm`。因此**单窗过拟合成功不等于已学会 Action-conditioned Dynamics**；通过初态和时间记忆拟合仍是可能解释，尚未针对其机制做独立干预。

### Q1–Q6 与证据分级

1. **Q1：是否有真正的旧代码 Bug？** 在本轮覆盖的输入、因果前向、Mask、单位、D0、梯度、优化器及评价链中未发现可复现的确定性 Bug。新工具 Pillow 依赖缺陷已修复，不能作为历史退化根因。
2. **Q2：D0 能否拟合运动？** 在所选 ep2/H5 窗口可以：Anchor Oracle CD 降至 1.5705 mm，h5 位移 14.27 mm，显著优于 Copy 和质心平移参考；不证明所有形变、缺失区域或长时范围都可表达。
3. **Q3：CD-only 与 Full 谁更容易产生有效运动？** 两者都能；CD-only 几何略好，Full 渲染更好。有限折衷存在，但不足以解释旧模型几乎不动。
4. **Q4：Condition Encoder 是否有有效梯度和更新？** 有非零梯度及实际 AdamW 更新，因而排除“完全未训练”的解释；条件扰动收益依然极小，不能据此宣称条件语义已被利用。
5. **Q5：最可能的问题位于哪里？** 目前优先指向多窗口条件到轨迹映射及联合优化不足；采样、曝光量、跨窗梯度竞争和条件可辨识性尚未分离。原 1200/24=50 次每窗曝光与本轮 400 次单窗曝光不同，不能把这一差异直接当作因果证据。
6. **Q6：下一轮先改哪个最小组件？** 先做实验 Runner 的采样和每窗成功更新计数控制诊断，保持模型、Decoder 和 Loss 不变。按 GT 质量与运动预先冻结 1→2→4 窗对照，分别控制每窗曝光和总预算，记录跨窗梯度及条件消融，再决定是否修改 Condition Encoder。本轮不实施或启动这一后续实验。

| 类别 | 判断 | 证据与适用范围 |
| --- | --- | --- |
| Confirmed Bug | 新审计导出器错误依赖 Pillow；原模型未确认 Bug | 30904 失败日志、修复单测及 30905 完成；不解释历史近静态 |
| Supported Hypothesis | 原多窗口训练样本内拟合不足，条件利用很弱 | 24 个已见窗口改善不足 0.4%；单窗学会较大运动后条件消融仍极小 |
| Supported Hypothesis | 局部几何/渲染竞争存在 | Tactile 子模块负 cosine；E2 的 CD/渲染指标折衷；不是整网根因证明 |
| Inconclusive | 曝光不足、跨窗梯度竞争、条件映射/可辨识性或时间记忆何者主导 | 尚无等曝光的多窗干预；不能将 50 vs 400 次曝光差直接解释为因果 |
| Inconclusive | 独立重建质量与原始硬件同步的剩余影响 | 无材料 ID、无独立 robot 硬件原始时间轴；本轮不能穷尽排除 |
| Rejected Hypothesis（限定范围） | D0 无法表达本窗口局部运动 | Oracle 与两网络都显著优于 Copy；仅覆盖 ep2/start73 H5 |
| Rejected Hypothesis（限定范围） | Full Loss、全链路 detach 或永久零初始化使 S0 必然静态 | E2 完整链 400 步能动，真实梯度/更新非零；不等于所有样本均无问题 |
| Rejected Hypothesis（限定范围） | 退化仅由 H15 外推或诊断窗口未见引起 | 原 H5 的 24 个实际训练窗口仍近静态 |

### 复现入口与证据路径

完整命令、配置、Job、checkpoint 路径及哈希见结果目录的 `execution_manifest.json`。以下为已执行入口的可读形式；`$PY` 为服务器现有 `.venv/bin/python`，`$DATA` 为 `/data1/userdata/tcweng/projects/touch4d/datasets/touch4d-v1`，`$OUT` 为本节服务器结果根目录。GPU 命令只能在新的独立 Slurm 分配及对应固定 SHA Worktree 中运行；若需复现必须另选唯一输出目录，禁止重用已完成目录。

```bash
# CPU 单测，两端均执行
python -B -m unittest discover -s scripts/tests -t .

# 6651b791818a65859f3edfed30228ec6bd43b05c：30905、30907、30908
"$PY" -B -m scripts.experiments.debug_audit.runner diagnose --root "$DATA" --output "$OUT/diagnosis-v2" --expected-sha 6651b791818a65859f3edfed30228ec6bd43b05c
"$PY" -B -m scripts.experiments.debug_audit.runner fit --mode E1 --root "$DATA" --output "$OUT/E1" --expected-sha 6651b791818a65859f3edfed30228ec6bd43b05c
"$PY" -B -m scripts.experiments.debug_audit.runner fit --mode E2 --root "$DATA" --output "$OUT/E2" --expected-sha 6651b791818a65859f3edfed30228ec6bd43b05c

# 7ad9ed7b755b88431c249f806dd97b6c2ee79ff8：30906
"$PY" -B -m scripts.experiments.debug_audit.checkpoint_seen --root "$DATA" --manifest /data1/userdata/tcweng/projects/touch4d/audit/p3-p4-spatial-dynamics-v1/windows.json --output "$OUT/checkpoint_seen.json" --expected-sha 7ad9ed7b755b88431c249f806dd97b6c2ee79ff8
```

- HTML：`D:\Amax\Projects\touch4d-experiments\p3-p4-debug-audit-v1\report.html`；离线数据与图片同目录，显示本次 H5 五个独立预测状态。完整产物的 2160 个 PNG 均可解码，24 份 `geometry.npz` 各含完整五步；报告实际引用 2070 个 PNG、23 条记录，配对 CSV 有 115 行。E1/E2 第 400 次均含五个不同的预测状态。
- 浏览器边界：CUA 打开本地 `file://` 被浏览器安全策略拒绝，未完成真实浏览器交互或截图验收；静态资源与数值完整性检查不能替代浏览器验收。最终静态审查状态记录于 `acceptance.json`。
- 汇总：同目录 `results.json`、`gradient_audit.json`、`oracle_results.json`、`training_comparison.csv`、`conclusion.json`、`acceptance.json`、`execution_manifest.json`。原始分层记录保留于 `diagnosis-v2/`、`E1/`、`E2/`；CPU 数据与采样证据为 `data_audit.json`、`sampling_audit.json`。
- 新 checkpoint：服务器 `$OUT/E1/last_400.pt`、`$OUT/E2/last_400.pt` 及各自 `checkpoint_{0,100,200,400}.pt`；Oracle 在 `$OUT/diagnosis-v2/oracle-{cd,full}/`。
- 历史 checkpoint：`/data1/userdata/tcweng/experiments/touch4d/p3-p4-spatial-dynamics-v1/runs/p3p4-{S0,S2-C1}-D0-F0-h5-u1200-s42-3556608/{best,last}.pt`，读取前后哈希见 `diagnosis-v2/historical_hashes_{before,after}.json`；原窗口名单 SHA256 为 `12c50f8a259aed3e8840cfda3bcb943f7911580a0fce5559d24900fe9fe9ae80`。

本轮工程与有界实验验收完成，**科学根因仍是带边界的定位结论，并未唯一证实**。未新增 Attention、Decoder、物理 Loss、H15 正式训练或额外多 episode 训练；不进入 S5，不合并 main。
