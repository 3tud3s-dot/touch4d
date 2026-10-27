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

进行中。所有新产物与失败记录写入独立结果目录；最终集中给出 Confirmed Bug / Supported Hypothesis / Inconclusive / Rejected Hypothesis 证据表，以及 Q1–Q6 回答。具体执行 SHA、Job ID、checkpoint 哈希、完整命令和数值将在验收后补入本节，不重写原实验记录。
