# Condition Identifiability V1

## 冻结协议（预测前）

基于 GitHub 最新 `main@64b1320d3718e7d05d590ce854b8aeb05f14236f`，分支 `codex/p2-condition-identifiability-v1`。仅移植 `400b4c3` 中已验收的 A0/T3 `conditions.py`；`ToolPredictor` 继承主线 `PositionPredictor.forward`，GS Encoder、Cross-Attention、D0、Loss 未修改，没有移植 S1–S4、D1/D2。旧 checkpoint 不加载、不覆盖。

本地源码 `D:\Amax\Projects\touch4d-worktrees\p2-condition-identifiability-v1`；结果 `D:\Amax\Projects\touch4d-experiments\p2-condition-identifiability-v1`。服务器结果 `/data1/userdata/tcweng/experiments/touch4d/p2-condition-identifiability-v1/`，执行源码固定独立 Worktree。用户交互作业 30903 不进入、不复用、不干预；最多额外两个单卡作业。

两个实验同一配置：P2 14D/128宽、2层4头结构注意力、Local k16、128 anchors、pool k16、固定各向同性RBF倍率1、A0累计7D、T3 10mm局部Mean/Max、fallback-only、按episode整手Mask。H5、batch1、seed42、AdamW lr1e-4/wd1e-4、clip1，Full Loss = CD + RGB + .1 Alpha + .01 Scale。每个实验一个共享模型，两个样本严格交替，400次成功更新（每例200），同一初始权重哈希；0/100/200/400评估，0/400导出图像。不调参、不延长、不按结果换窗。GPU smoke先每个实验4次更新并检查恢复，再重新初始化正式实验。

### 合成

取 Train ep2/start73 初始GS与相机，仅作为模板。两个输入具有完全相同GS、时间编码、零值有效触觉及Masks；只改变Action/Controller位置。条件j以初态工具位姿为基础，沿世界Z偏移 ±2(j+1)mm；目标h为初态GS平移 ±2h mm，h=1..5，属性不变。h1已有不同条件，避免制造首步不可辨识目标。相邻GT仅用于监督，不进入输入。

合成全图RGB/Alpha由变换后的GS渲染，所有相机像素作为有效合成监督；不套用原始真实物体局部遮挡mask，否则会错误屏蔽移动后的合成对象。Depth为零占位且权重0，不能解释为合成深度真值。该实验只验证可学习性，不证明真实物理模型。

### 真实双窗口 CPU 预选

六个训练episode各12个确定性均匀起点，H5。候选须同一个episode内帧完整、操作手Controller有效，点数/初态比在0.8–1.25，以排除明显数量突变；这只是预选工程筛查，不证明GT完全无缺失。配对要求跨episode且同单双手模式，避免Hand Mask成为不同轨迹的捷径。

计算因果j=1..4累计Controller位移的配对差RMS，取配对差上四分位候选；按初态世界坐标256点集合CD排序，前8对再用全部初态点精算，选择最低完整CD的一对。顺序、窗口及原始来源哈希在训练前冻结；目标变化在选定后审计，不按模型输出选择。报告初态CD、动作差、逐h目标差及点数变化，不把“相对最近”宣称为严格相同。若初态或目标差仍存在明显混杂，结论明确受限。

### 验收

正常条件、Freeze Action、Zero Tactile、Swap Action、Swap Tactile、Swap All、Copy使用同一checkpoint和目标评价。记录逐h CD/RGB/Alpha、世界位移RMS/方向、去平移形状误差、条件扰动输出差与ΔCD；合成Swap Action同时对交换后的正确目标评价。实际更新按模块分别记录参数Norm、raw梯度、clip梯度及AdamW更新；不把非零梯度等同有用条件。

区分：单窗记忆、同初态下条件分离、条件敏感性、条件带来的误差改善、跨窗口泛化。真实初态不严格相同，因此即使拟合成功，也不能完全排除GS初态识别窗口。负面结果保留，本轮只完成以上两个实验。

## 实际结果

待 CPU 审计、CUDA smoke 与两组有界实验完成后填写。原始数值与失败记录保留在独立结果目录。
