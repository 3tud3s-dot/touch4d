# P3-v0: Two-Frame Temporal Dynamics + Template Refinement

冻结来源：用户本轮指令；基准 `fdd4013`，分支 `codex/p3-two-frame-h10`。不合并 main，不覆盖 P2 或旧审计。

## 目标与设计

独立验证 H1（两帧历史融合）和 H2（受限尺度/opacity 更新）。当前帧模板点数固定，历史 GS 可具有不同点数和排列，不匹配跨帧 ID。

共享 P2 Encoder：14D GS、LocalGSBlock、FPS128、2层128维4头结构 attention。真实 t-1/t 独立选锚点；预测帧沿用 t 的模板、FPS/邻域和各向同性 RBF。Temporal cross-attention 用当前结构锚点作 Q、上一帧作 K/V，并加入相对 XYZ，独立 padding mask，零初始化残差投影后通过 t 的 RBF 插回当前 GS。

P3 用一套共享 One-Step Dynamics 递推10步：`x_next=x_current+W_t*anchor_delta`。两帧状态滚动，预测属性反馈 Encoder，保留全部梯度，不 teacher forcing。各版本统一使用因果条件前缀；第 k 步目标 t+k，只读截至 t+k-1 的动作/触觉。未来触觉为在线到达的条件，不称开放环预测。

沿用原条件时间编码与预测步时间编码，但 Encoder、Temporal Fusion、条件 Cross-Attention 和位移 Head 在十步间共享参数。仅缓存已经编码的上一帧状态，不 detach；不为各步创建独立 Dynamics。每次成功 optimizer update 原子记录 last.pt，恢复从已持久化的下一步开始；每20次评估、保留 step checkpoint 与最低CD。恢复中补齐中断的评估时不重复 optimizer step。

Refinement 独立开关：锚点级零初始化 Head，插值后相对最新真实初态预测 `log_scale_t+log(1.3)*tanh(delta_scale)`、`opacity_logit_t+0.5*tanh(delta_opacity)`；不逐步累加属性。Quaternion/SH DC 继承 t，尺度和 opacity 反馈下一步。不引入增删点、协方差传输、材料先验或新 loss；不能保证恢复模板中完全不存在的区域。

## 冻结实验

| 模型 | 架构 | 历史 | 属性 |
|---|---|---|---|
| M0 | 原 P2 H10，累计位移 | 1 | 固定 |
| M1 | P3 共享 One-Step AR | 1 | 固定 |
| M2 | M1 + Temporal Fusion | 2 | 固定 |
| M3 | M2 + Template Refiner | 2 | 有界 scale/opacity |

| Scene/episode | history | targets |
|---|---|---|
| 001-rope/1 | 84,85 | 86–95 |
| 001-rope/2 | 72,73 | 74–83 |
| 051-cube/0 | 108,109 | 110–119 |
| 051-cube/0 | 131,132 | 133–142 |

每场景每模型400 updates，两个窗口交替各200；seed42、batch1、AdamW lr=1e-4、weight_decay=1e-4、梯度裁剪1、每20次评估。共8组，不按结果提前删组。损失保持 `CD+RGB+0.1*Alpha+0.01*ScaleRegularization`，相机与有效区沿用已验收配置。最低CD和last_400分开报告；经验改善约5%仅作值得关注幅度，不是统计显著性或物理容差。

数据保持 `frame_ids=t:t+10`，另加 `history_frame_ids=[t-1,t]`。历史独立 padding，但两帧用同一 scene 坐标/尺度归一化；padding与opacity都不作为遮挡真值。

## 执行、验收与交付

按 Stage0审计冻结→Stage1数据→Stage2模型→Stage3真实CUDA及8组训练→Stage4审计HTML→Stage5文档/Git执行。测试覆盖旧H5/H15不变、零初始化M1=M2=M3、可变点数/排列、padding、因果性、全十步梯度、属性界限、安全恢复与旧checkpoint保护。OOM时只采用保梯度工程优化，不缩H或detach。共享权重初始化一致，额外分支追加，全部配置写入独立P3 checkpoint。

指标：h1/3/5/10及均值CD、RGB/Alpha，世界/去质心CD，rope两端集合误差/邻域位移/铺展，cube平移RMS/质心误差/局部形态，M3尺度及opacity分布、训练曲线、最佳/末次。GT无固定ID；端区集合不是材料点轨迹。另做132/133共同目标134–142模板审计，不将未来133用于预测133。

报告：`audit/p3-two-frame-h10-v0/report.html`，显示两帧真实历史和10帧独立预测，窗口/模型/检查点切换、GT/预测/复制初态、几何/椭球/轨迹、三相机RGB/Alpha和完整曲线。离线资源同目录，附 results.json、acceptance.json、run_manifest.json、template_audit.json。区分工程通过、拟合能力与假设支持；不因结果不佳追加训练/调参。本轮不启动全量训练。

实际版本、命令与结果完成后记入现有 MODEL_ROADMAP.md；审计证据保留在上述独立目录。
