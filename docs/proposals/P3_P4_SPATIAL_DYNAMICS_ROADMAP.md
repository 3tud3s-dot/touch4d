# P3–P4 Spatial Dynamics：预注册实验协议

## 1. 目标与证据边界

冻结日期：2026-10-27。正式基线 `main@64b1320d3718e7d05d590ce854b8aeb05f14236f`；选择性移植 A0/T3 的来源为 `9142704df41875895afe61a1a119bbe83e2f7c65` 的 `scripts/experiments/condition_encoder/model.py`。新分支 `codex/p3-p4-spatial-dynamics-v1`，不合并旧研究分支，不改正式 P2 默认行为，不创建 PR。

目标是定位工具空间交互、条件后传播、形变 Decoder、AR 递推的瓶颈。工程正确、训练可优化、验证动力学收益分别判断。负面结果同样构成完整交付；不得因结果不好更换窗口、无限加预算或改变损失。

历史 200-update 单 seed 验证 CD：复制初态 2.007290 mm，B0 2.013685，A1 2.013275，T1 2.014359，T2 2.011243，T3 2.011885。条件响应很小，均未胜过复制初态。A0/T3 是待验证的研究选择，不是已获证实的赢家。

## 2. 代码事实与固定接口

P2 每步用上一预测中心及初态 11D 属性组成 14D GS，逐点 MLP → LocalGSBlock → 两层 Structural Anchor Attention → 两层条件 Cross-Attention → 锚点位移头 → 初态固定 RBF。LocalGS 的 k=16、FPS 128、锚点池化 k=16、宽度128、4头。初态图索引固定，相对 XYZ 随预测更新。

Anchors 路径删除自由 Scale/Rotation Head，继承初态尺度、四元数、opacity、SH DC；当前视觉铺展不能归因于正在学习的 Scale Head。D2 是本轮唯一改变四元数的候选，仍不学习尺度。

统一张量：GS `[B,N,14]`；GS 特征 `[B,N,128]`；锚点 `[B,A,128]`、坐标 `[B,A,3]`、有效性 `[B,A]`；动作 `[B,H,2,13]`；A0 Controller `[B,H,60,7]`；触觉 `[B,H,4,12,32]`；条件 `[B,H,66,128]`；预测中心 `[B,H,N,3]`。未来 GT 是独立点集，只用于监督，无跨帧行号对应。

### A0/T3 移植边界

仅移植累计7D、同指10mm Gaussian Mean、Voronoi Local Max、Coverage、Age、残差融合及 fallback-only；不包含 A1/T1/T2 模式。60个工具 token、2个状态和4个备用槽位保持；有效零压力保留。局部坐标使用既有 geometry/rebuild 网格，空间资格不等于接触真值。

操作手元数据由已人工确认的 episode 规则写入 manifest：ep0–4 仅右手，ep5–9 双手。现有重构 metadata 的物理 slot 不能单独证明参与状态。整手关闭 action/controller/tactile/fallback，静止的双手仍保留。

条件预测为在线观测：第 k 步 memory 为 `[null,Z_0,...,Z_(k-1)]`，长度 `66k+1`。只用最新可用工具坐标构造空间项；动作是同帧状态，非提前知道的未来命令。所有 AR 状态保留梯度，不 teacher forcing，不喂未来 GS。

## 3. Step 3 候选

| ID | 变化模块、假设 | 不变的对照 |
|---|---|---|
| S0 | P2 逐GS内容 attention + A0/T3 | 全部候选共同基线 |
| S1 | 逐GS Query 加最新工具相对 XYZ 的逐头 bias，检验显式作用位置 | S0 的 Encoder/Decoder/条件/容量主干 |
| S2-C0 | 先池化128锚点，再做内容 attention；检验交互层级 | S0 的 Decoder，不能声称 attention 与池化交换等价 |
| S2-C1 | S2-C0 加最新工具→当前锚点 XYZ bias | S2-C0 |
| S3-Pre | S2-C1 条件融合前增加一层结构传播 | 与 Post 完全相同参数量 |
| S3-Post | 同一结构层改放融合后，检验传播顺序 | S3-Pre |
| S4-C0 | 96全局+32工具几何相关动态锚点 | S2-C1，固定总预算128 |
| S4-C1 | 相同候选池与选择算法，额外触觉相关性 | S4-C0 |

### 空间交互

`logit = QK^T/sqrt(d) + g*b(tool_xyz-query_xyz)`。`b` 为3→128→4 MLP，末层零初始化；只为合法 Controller token 2:62、最新条件时刻启用，state/fallback/null 无伪造位置。合法性与 token mask 同时满足。历史 token 保留内容和时间，不获得最新位置偏置。S1 初态与 S0、S2-C1 与 C0 的零 bias 数值等价须实测。

额外结构传播复用 `StructuralAnchorLayer` 的全有效锚点 attention（完整空间图，身份固定，坐标更新）；额外残差投影零初始化。它不是材料拓扑，自接触与远距错误连接仍是风险。Pre/Post 共同初始化、同一层数，不能把深度收益误认成顺序收益。

### 动态锚点

固定初态 FPS 前96为全局；其余合法 GS 为共同候选池。最新合法工具距离核带宽固定为场景归一化坐标0.1（001约15.7mm），不搜索。C0 relevance 为几何核最大值；C1 用 `alpha=coverage*(mean+max)/(1+mean+max)` 乘同一核，均为已缩放连续触觉，不是物理力。C1 有效零或无有效观测时退回全局 FPS；C0 无合法工具时退回全局。TopK 稳定排序、排除全局重复、点数不足补无效槽；离散索引无梯度，采样后的特征/坐标可微。

动态锚点以当前坐标的几何核将特征回写128固定解码锚点，再用 D0，固定总查询数与后续模块。报告选中身份切换率、局部配额、最近覆盖距离、局部集中程度；不隐瞒选择不连续性。比较 C0/C1 时上述机制完全一致。

复杂度：S0/S1 的 Cross-Attention 为 `O(sum_k N*(66k+1)*D)`；S2–S4 为 `O(sum_k A*(66k+1)*D)`；额外结构层 `O(H*A²*D)`；S4 增加 `O(H*N*60)` 距离及回写 `O(H*A²)`，不扩展锚点数。实测参数量、显存与用时另记。

## 4. Step 4 候选（训练前冻结）

**D0/F0**：`X_k=X_0+W_0 d_k`，d为本步预测累计锚点位移，不重复累加。W0由初态确定。

**D2**：头输出累计锚点平移t与世界旋转向量θ（后者零初始化），`R=Exp(θ)`；`X_ik=Σa W_ia[R_ak(X_i0-a_a0)+a_a0+t_ak]`。每个 Gaussian 的轴旋转使用锚点旋转四元数半球对齐（参考单位四元数）后按W加权、归一化，再左乘初态WXYZ，零范数回退单位旋转；记录旋转角与有限性。中心混合不是SO3，不能将混合矩阵当Gaussian旋转。尺度、外观继承初态，更新后的合法四元数反馈 Encoder。恒定纯平移须退化为 D0；不引入仿射/尺度头。

**R1**：初态 k=16 空间邻接排除 self/padding，冻结；采用 `mean smooth_l1(relu(abs(d_k-d_0)-0.1*d_0),0,beta=0.01)`（距离均为既定归一化单位），对有效边、H等权。权重候选集合仅 `{0.01}`，不搜索。10%邻距软容差是弱正则设计值，不是数据质量阈值或材料参数；审计初态边长度及可能的自接触，解释锁死运动风险。

**F1**：`d_total,k=d_total,k-1+delta_k; X_k=X_0+W_0*d_total,k`，delta为一步锚点增量，完整AR梯度。仅与同一主干 D0/F0 比较，首轮不叠加 D2。

**D1（条件触发）**：S4+D0 额外零初始化局部平移头；`X=X0+W0*d+g*Wc*dc`。最新有效触觉相关性定义g；32局部锚点定义Wc；无有效正响应则g=0自然退化。保持S4公共参数、固定监督，与同S4+D0单变量比较。

兼容矩阵：Stage1全部D0/F0/无R1；Stage3主干分别 D0、D2、D0+R1、D0+F1；D1仅S4；条件触发D2+R1。禁止全排列、P1a.5、物理loss、额外GS属性头。

## 5. 数据、监督与预算

Train=1,2,4,5,7,8；Val=3,6；Test=0,9不参与选择。本轮默认不触发Test：只有H15两seed验证一致支持并锁定架构才执行一次性Test，使用每ep4个合法分位窗口，报告其有界覆盖，不能冒称全测试集。

H5固定24训练起点：ep1 23/69/116/162；ep2 35/107/179/251；ep4 44/134/224/314；ep5 37/113/188/264；ep7 27/82/137/192；ep8 27/82/137/192。Val ep3 35/105/175/245；ep6 25/85/145/205。来自上一轮 `windows.json`，以原文件 SHA256溯源。种子random.Random(seed)重复shuffle24，首次200必须复现旧顺序，扩展至400/1200。

H15另行CPU核对同一起点完整16帧及三相机有效监督后冻结；若不合法，按不依赖预测的合法起点时间分位替代并记录，不能截短H。目的性可视化窗口 ep1/85、ep2/73、ep3/65/115/150、ep6/110，先审计H15合法性；不用于架构选择。

所有组从零、batch1、seed42（复验seed43）、AdamW lr=weight_decay=1e-4、clip_grad=1。CD1+RGB1+Alpha0.1+Scale0.01；RGB/alpha为目标GS同三个相机渲染，在既定HDF5有效深度区域监督，非原始RGB。无深度loss。逐成功optimizer step计数，不跳过无监督/非有限batch：报错停止该组。R1组唯一额外loss如上。

筛选400 updates，在200/400集中评估；保留checkpoint_200、last_400及验证CD最低best（不只展示best）。充分性复验1200，从零，评估400/800/1200；保留全部评估点，最终比较同预算last，best仅诊断稳定性。所有选模均episode等权。

## 6. 冻结决策树与停止边界

1. Stage0：CPU全测试、旧P2接口不变、真实CUDA每种结构最少2次更新及恢复/梯度检查通过，才开始Stage1。CUDA smoke不作科学结论。
2. Stage1八组必须完成400。主筛选门槛为相对S0 CD至少1%改善且ep3/ep6均不恶化、RGB/Alpha各不恶化超过5%、局部邻距变化及bbox异常不扩大超过10%；数值是预注册实际意义筛选参考，不是统计显著性或物理质量阈值。200/400方向矛盾标不稳定。再查copy与条件响应，不将更静态自动判成功。
3. 满足筛选的候选按400平均CD排序；1%以内近似并列优先简单结构（S1、S2-C0、S2-C1、S3-Pre、S3-Post、S4-C0、S4-C1）。报告对应配对差异：S1/S2、C0/C1、Pre/Post，不只全对S0。
4. 全无收益：明确没有赢家，保留S0负面对照，预选简单空间候选S2-C1进行Decoder和充分性诊断，禁止加attention层数。主干称为diagnostic而非adopted。
5. Stage3在该主干完成400的 D2、D0+R1、D0+F1；D0复用Stage1同seed同预算，S0始终保留。若S4-C1较C0在两个episode的工具近区CD均改善≥1%，且几何/渲染质量门槛通过，则额外S4-C1+D1；若C0本身满足上述相对S2-C1近区收益而C1未满足，使用S4-C0+D1。否则D1记NOT_TRIGGERED，不凭主观图像追增实验。
6. D2若两个episode质心位移向量误差均比D0降低≥1%，但邻距变化较D0增大>10%，追加一次D2+R1；否则NOT_TRIGGERED。R1权重不扩展。
7. Stage4始终进行有界充分性诊断：S0和一个候选各1200，seed42/43，共4组。候选取Stage3满足同一门槛的最佳；若没有任何候选满足，固定S2-C1+D0（不得选微小最优并称胜者）。这是对训练充分性的诊断，不是反复调参。
8. H15训练仅当1200两seed候选均优于S0≥1%、均不差于copy、两episode无相反退化且质量门槛通过时触发：S0/该候选分别H15/1200/seed42/43，共4组。未触发须报告“长期动力学未获验收”，不冒充H15训练成功。
9. 无论H15训练是否触发，最终必须完成H15目的性可视化。未触发时，仅进行明确标记的**H5权重H15外推诊断**：共享结构加载H5，已观测condition及prediction时间embedding以末项保持延拓至H15（无新增训练、无插值调参）；前5帧必须数值相同，h6–15属于未训练长度外推，不纳入H15学习能力结论。这是为满足长时失败分析与HTML交付的接口补充，不能与正式H15训练混淆。报告界面醒目标注。
10. 正式H15通过与否均报告，不以追加训练求改善。未触发Test、D1、D2R1明确登记依据。每个基础设施错误最多一次相同配置重试；数学/实现错误先修测试、新SHA、重做受影响对照，不能悄改已产生指标。

预算上限：Stage1 8×400；Stage3 3×400，条件分支最多2×400；Stage4 4×1200；H15条件分支4×1200；smoke、可视化和一次性Test为只读评估。所有组独立目录。没有无限搜索或自动完整epochs。

## 7. 评估与解释

每窗口/ep/horizon报告unsquared双向CD毫米、RGB/Alpha MAE、joint、copy；ep等权汇总，h1/5/10/15及全部h。GT集合变化使用初态→目标CD，不按行推断轨迹。

每重要checkpoint执行real、freeze_action、zero_tactile、copy。冻结动作同时冻结state/controller及其派生工具几何，保留合法性交集；触觉置零保留观测mask/age。ΔCD=perturbed−real，另报输出差异；只响应不代表有益。

额外诊断：质心位移向量误差、中心去平移CD、最新工具近区/远区（以0.1归一化距离作固定诊断分区，非接触标签）对称最近距离；预测位移RMS、bbox各轴跨度比；初态邻距变化、锚点位移邻域梯度；三相机alpha阈值0.5占用面积（仅诊断，非3D体积），相对GT与初态分别报告。GT局部区域用独立点集的同一工具距离划分，不假设Gaussian对应。

采样GT可见性、mask和重建缺失限制明确保留；模型有运动不代表方向正确。H越大GT集合差异也可能变大，不能把绝对CD增长全部称为AR累积。

## 8. 工程、恢复与历史保护

实验类放 `scripts/experiments/spatial_dynamics/`，复用PositionPredictor/P2结构/renderer/objective/processor，不修改正式模块。新checkpoint独立version、所有模式/horizon/seed/manifest/源码SHA/optimizer/RNG显式记录，配置不符拒绝恢复；按成功step恢复，无重复更新。旧checkpoint读前后SHA256一致。

CPU：A0/T3 masks/零压力/NaN/fallback、S0等价、零bias等价、动态选择去重与fallback、D2合法旋转/平移、R1平移不变、F1语义、未来条件/GT不泄漏、跨步梯度、可变点数、checkpoint拒绝冲突。旧全测试不得少测或放宽容差。

GPU：Slurm单卡5090，每组独立输出、最多4并发。新SHA固定独立服务器worktree。复用现有venv/data；专属扩展缓存先单作业预热，后续同SHA只读复用，避免并发构建。记录allocated/reserved峰值、梯度、用时。H15 OOM只允许验证等价的activation checkpointing，绝不detach/截短H；无法安全运行记BLOCKED。

调度采用array %4、阶段依赖和sbatch --wait；不循环tail/squeue。最终读取sacct及状态文件。用户原工作区、旧checkpoint、audit、数据均不写；不修改main、不删除分支、不覆盖结果。

## 9. 交付与证据目录

开发worktree：`D:/Amax/Projects/touch4d-worktrees/p3-p4-spatial-dynamics-v1`。服务器代码 `/data1/userdata/tcweng/projects/touch4d-worktrees/p3-p4-<SHA>/`，运行期间固定SHA，不改动main或历史worktree。

2026-10-27 非破坏性运行隔离补充：尚未提交的后续作业统一使用项目外输出根 `/data1/userdata/tcweng/experiments/touch4d/p3-p4-spatial-dynamics-v1/`，其中 `runs/p3p4-<ID>-s<seed>-<SHA>/` 保存checkpoint/config/metrics，`logs/` 保存Slurm输出，`reports/` 保存汇总与可视化。目录创建时拒绝覆盖；不同作业不得共写同一run。最终本地报告位于 `D:/Amax/Projects/touch4d-experiments/p3-p4-spatial-dynamics-v1/`。

已经提交的CUDA验收Job 30856不迁移、不重启，继续绑定 `p3-p4-3556608`（完整SHA `3556608016e718c926c4a951faed49a0e116b247`）。其唯一输出 `/data1/userdata/tcweng/projects/touch4d/audit/p3-p4-spatial-dynamics-v1/cuda-3556608/` 及 `cuda-30856.out/.err` 安全保留。已有manifest/CPU证据仍在该独立audit根及本地 `D:/Amax/Projects/touch4d/audit/p3-p4-spatial-dynamics-v1/`，不移动、不覆盖。上述仅修正执行路径，不改变任何研究假设、预算、数据窗口或验收标准。

保存manifest、config、数据hash、init配对、commands、SlurmID、resources、updates、每评估点metrics、checkpoint及hash、status/error。总表CSV/JSON追溯每window/horizon；离线HTML包含GT/pred/copy、h1/5/10/15、多相机、几何/运动/铺展、条件消融和训练曲线，不依赖CDN。

## 10. Results Log（只追加实际状态，不改以上预注册规则）

- PREPARED：远程main与已知SHA一致；原工作区未提交文件保留；当前无用户Slurm作业；新分支从main建立。
- 尚无本轮训练结果。Stage0–4均PENDING；条件分支尚未判定。
- CPU：本地236项，210通过/26跳过/0失败；服务器236项，234通过/2个CUDA opt-in跳过/0失败。H5/H15各24训练、8验证窗口及6个H15目的案例通过CPU数据契约核查。CUDA验收Job 30856已提交，当前绑定代码与输出保持不变；后续输出改用第9节的项目外命名空间。

## 11. 最终研究结论

待实验完成。分别回答空间bias、anchor交互层级、传播顺序、动态分配、D0/D1/D2、R1、F1、条件收益及copy优势；标明支持/无足够证据/反对/未测试。短预算固定子集不能证明完整数据泛化；两seed不等于统计显著性。
