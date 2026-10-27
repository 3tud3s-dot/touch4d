# Tool-to-Anchor B V1：H15 运动拟合非退化实验

## 目标与版本

本轮只验收 **Non-regression of Motion Fitting Capacity**，不是条件辨识、跨episode泛化或SoMA Stage 1。正式P2 `main@64b1320d3718e7d05d590ce854b8aeb05f14236f` 已与GitHub核对一致。独立分支 `codex/p2-tool-anchor-b-h15-v1`；不合并main，不导入S3/S4、D1/D2/F1。

A0/T3与Hand Mask逐字移植冻结 `265d28a:scripts/experiments/condition_identifiability/conditions.py`（源于已验收 `9142704`，在P3–P4 `3556608` 中同样使用）。只读参考 `MODEL_ITERATIONS.md` v3/P2。正式 `scripts/training/`、Dataset、processor、loss均不修改。Base继承原P2 forward；B仅覆盖独立实验类的交互路径，复用原GS/结构编码、AnchorPool与D0函数。

本地源码：`D:\Amax\Projects\touch4d-worktrees\p2-tool-anchor-b-h15-v1`。本地结果：`D:\Amax\Projects\touch4d-experiments\p2-tool-anchor-b-h15-v1`。服务器结果：`/data1/userdata/tcweng/experiments/touch4d/p2-tool-anchor-b-h15-v1/`；执行worktree使用提交SHA命名并冻结。用户30903及历史worktree不干预。GPU始终额外一张，Base→Candidate顺序执行。

## 唯一架构改动

保留14D→128D GS、Local k16、128 FPS Anchor、2层4头Structural Attention、A0 7D、T3同指10mm Mean/LocalMax/Coverage/Age/Mask、整手Mask、Fallback、共享D0累计位移、原Full Loss。

设当前结构化GS特征 `S[B,N,128]`，原初态固定场为 `W0[B,N,A]`、固定pool/FPS索引与带宽 `sigma0[B]`，`A<=128`。每步使用上一预测GS坐标（首步初态）取当前Anchor中心 `p[B,A,3]`，不重选FPS、不重算W0。

1. `Q=AnchorPool(S)+e_pred[k]`，形状 `[B,A,128]`。合法历史 `M=[null,Z0..Zk]`，形状 `[B,1+66(k+1),128]`；`G`沿用原两层4头CrossBlock，含原残差/FFN，只把Query由GS改为Anchor。
2. 最新时刻60个有效T3 Controller token `z[B,60,128]` 与坐标 `c[B,60,3]`。相对坐标 `r=p-c`；空间log先验 `-.5*||r/(2 sigma0)||²`，带宽倍数2在训练前固定，不搜索。使用同一场景归一化单位。
3. 每头分数 `q·k/sqrt(32)+MLP_bias(r)+log_prior`，工具无效位置为负无穷；拼入固定零logit、零Value的null后共同softmax，得到 `[B,A,4,61]`。不能剔除null再重新归一化工具权重。无效Anchor也仅指向null。远距离自然退让，不加接触幅值阈值。
4. 普通多头Value聚合为 `[B,A,128]`；同一权重按头平均聚合A0累计位移 `c_k-c_0` 为 `[B,A,3]`。初态或当前无效则该点位移为零，但当前有效空间token仍保留。h1累计位移本来为零，不读下一时刻补偿。
5. `T=MLP([Value,Motion])`，MLP为131→128→128、GELU、无bias，保证全null的零观测不经bias生成虚假局部响应。Q/K/V使用独立Linear，r使用3→128→4 bias网络。无额外位置Head、无物理门控。
6. `U=S+W0*MLP(G+T)`，MLP为128→128→128。随后仍为原 `AnchorPool(U)→position_head→d_k`，`Xhat_k=X0+W0*d_k`；位移相对同一初态累计，不能叠加到上一帧。其它GS属性继承初态。完整15步反馈图不detach、不teacher-force。

所有额外参数在共有模块之后初始化；共有参数逐项相同，零position_head使初始输出均为Copy。两模型的中间特征不要求相同。空间参数无隐藏超参数搜索。

## 与S0/S2-C1的区别

- S0：逐GS Query读取全部合法历史，直接Pool后D0。
- 历史 `3556608` S2-C1：Anchor Query，在历史memory的最近60个位置上加相对XYZ bias，输出Anchor直接进入Head；没有显式Gaussian邻近先验与局部null竞争、没有加权工具累计运动、没有条件响应W0传播回GS再Pool。
- B：历史分支仍保留所有合法条件；独立最新工具空间分支包含上述三个差异。未移植历史复杂结构，未将静止或零触觉操作手自动删除。

## 预注册实验与验收

只使用训练窗 ep1初态85→86–100、ep2初态73→74–88。H15、连续Δ=1，真实从零训练。数据manifest在训练前记录16帧、15条件时刻、3相机、时间戳、valid mask、初态带宽与来源SHA256；不根据预测结果换窗。

Base与B：seed42、batch1、AdamW lr/wd均1e-4、clip1、400成功更新、两窗交替各200；原loss `CD+RGB+0.1Alpha+0.01Scale`，depth0。两模型共有初始化与监督一致。每20步（含0）评估两窗全部H15；best按两窗/15时刻等权平均CD严格最小选择，相等保留较早；另保留last_400。不追加预算/seed、不恢复旧H5权重。

CPU先测试旧行为与新Shape、null、Mask、位置单位、固定场、D0累计、当前Anchor、因果性、梯度与checkpoint冲突。真实CUDA smoke每模型3更新，再单独用h15输出反向检查15次Head都有梯度；smoke结束重新seed42开始正式实验，不继承smoke权重。数值不finite即停止并保留证据。

主要成功条件：B best平均CD不高于配对Base best，并且h15几何、RGB/Alpha及可视化无明显退化。best与last均公开；若best达标、last明显退化则仅有表达能力证据，稳定性未通过。次要质量依据完整指标和逐帧图像判断，不事后编造数值容差。通过也只称“候选工程Base”，不宣称Action-conditioned dynamics或泛化。

保存全部每20步指标、成功update日志、参数/原始与裁剪后梯度/实际更新、峰值显存与时间、best/last哈希、只读Freeze Action/Zero Tactile、两窗GT/Copy/Base/B的H15 RGB/Alpha与三维点集。GT为独立重建，无跨帧ID对应。归档程序可视化独立15帧，禁止重复h5冒充h15。

## 实际结果

待必要测试、单卡配对训练及集中验收完成后填入。本节只记录真实结果，不追加新研究。
