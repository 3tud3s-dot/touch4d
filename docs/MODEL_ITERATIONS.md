# 模型迭代记录

撰写协议：版本按 v0、v1、v2 递增，注明 Git ID；仅记录输入结构与维度、网络交互形式、输出、loss、核心实验结果和简要结论。优先使用公式与表格，省略实现细节，未验证的设想不作结论。

## v0：初始 GS 的条件位移预测

- Git ID：`33bf1f20719bc21cff0306515b8948246d15419f`（简称 `33bf1f2`）。
- 设批大小为 $B$，初始高斯数为 $N$（批内补齐后的点数），特征维度为 $128$。

### 1. 输入编码为 token

初始 GS 的坐标和属性共 14 维，通过共享的逐点 MLP：

14 维由中心坐标 3 + 对数尺度 3 + 旋转四元数 4 + 不透明度 logit 1 + SH DC 颜色系数 3 组成。SH 为 Spherical Harmonics（球谐函数）；DC 沿用 Direct Current 的缩写，在此指不随观察方向变化的零阶常量分量。

$$
\mathbf{G}_0\in\mathbb{R}^{B\times N\times14}
\xrightarrow{\mathrm{MLP}}
\mathbf{H}_0\in\mathbb{R}^{B\times N\times128}
$$

| 输入 | 原始维度 | 编码后的 token 维度 |
|---|---|---|
| 两只夹爪的位姿与开口 | $B\times5\times2\times13$ | $B\times5\times2\times128$ |
| 60 个控制点坐标 | $B\times5\times60\times3$ | $B\times5\times60\times128$ |
| 四路触觉及有效掩码、空间掩码、时间差 | $B\times5\times4\times1153$（每指展平后） | $B\times5\times4\times128$ |

每指触觉热图为 $12\times32=384$ 个像素，因此 $1153=384$ 个触觉值 $+384$ 个有效掩码值 $+384$ 个空间掩码值 $+1$ 个时间差；掩码转为数值后与触觉一起编码。

三类条件 token 沿 token 维拼接，并加入身份和时间编码：

$$
\mathbf{Z}
=\operatorname{Concat}(\mathbf{A},\mathbf{C},\mathbf{T})
+\mathbf{E}_{\mathrm{id}}+\mathbf{E}_{\mathrm{time}}
\in\mathbb{R}^{B\times5\times66\times128}
$$

每个时刻共 $2+60+4=66$ 个条件 token。

### 2. GS 与条件做 Cross-Attention

预测第 $k$ 帧（$k=1,\ldots,5$）时，以初始 GS 特征加预测时间编码作为 Query，以前 $k$ 个条件时刻作为 Key、Value 的来源：

$$
\mathbf{Q}_k=\mathbf{H}_0+\mathbf{e}^{\mathrm{pred}}_k,
\qquad
\mathbf{M}_k=\operatorname{Flatten}(\mathbf{Z}_{0:k})
\in\mathbb{R}^{B\times66k\times128}
$$

$$
\mathbf{F}_k
=\operatorname{CrossAttentionBlocks}
(\mathbf{Q}_k,\mathbf{M}_k,\mathbf{M}_k)
\in\mathbb{R}^{B\times N\times128}
$$

采用 **2 层、4 头 attention**，包含残差连接和前馈网络。有效性掩码排除无效条件；实现中另加入一个始终有效的 null token，使 memory 长度为 $66k+1$。预测 $t+k$ 只读取截至 $t+k-1$ 的条件。

### 3. 输出相对初态的位移

每个 GS 特征经过位移头，输出三维位移：

$$
\Delta\widehat{\mathbf{X}}_k
=\operatorname{Linear}_{128\rightarrow3}
\bigl(\operatorname{LayerNorm}(\mathbf{F}_k)\bigr)
$$

$$
\boxed{\widehat{\mathbf{X}}_k=\mathbf{X}_0+\Delta\widehat{\mathbf{X}}_k}
\qquad
\boxed{\widehat{\mathbf{X}}\in\mathbb{R}^{B\times5\times N\times3}}
$$

位移头零初始化；五帧均相对同一初态预测，不递推累加。坐标与位移处于归一化坐标系；每个样本的有效点数保持为其初始 GS 点数，补齐点由掩码排除。尺度、旋转、不透明度和颜色继承初始 GS。

### 4. Loss

在归一化坐标系中，对预测点集 $P$ 和目标点集 $Q$ 计算双向 Chamfer（使用非平方欧氏距离）：

$$
\operatorname{CD}(P,Q)=\frac12\left[
\frac1{|P|}\sum_{\mathbf p\in P}\min_{\mathbf q\in Q}\|\mathbf p-\mathbf q\|_2
+\frac1{|Q|}\sum_{\mathbf q\in Q}\min_{\mathbf p\in P}\|\mathbf q-\mathbf p\|_2
\right]
$$

五个预测时刻和 batch 等权平均：

$$
\boxed{\mathcal L=\frac1{5B}\sum_{b=1}^{B}\sum_{k=1}^{5}
\operatorname{CD}\left(\widehat{\mathbf X}_{b,k},\mathbf X^{\mathrm{GT}}_{b,k}\right)}
$$

仅使用掩码内的有效点，允许预测与目标点数不同，不要求逐点对应。当前仅用 Chamfer 训练；深度误差仅用于报告，不参与反向传播。

### 5. 实验结果

001-rope 验证集共 104 个窗口，五帧平均、episode 等权统计，Chamfer 已换算为毫米，越小越好：

| 方法 | 训练轮次 | Chamfer（mm） | 相对复制初态的误差变化 |
|---|---:|---:|---:|
| 复制初始 GS | — | 2.010 | 基准 |
| v0 best | 3 | 2.018 | +0.38% |
| v0 last | 7 | 2.147 | +6.84% |

误差变化按未舍入数值计算，正值表示误差增大，负值表示改善。

### 6. 结论

当前验证结果未优于复制初态，后期训练出现退化。条件消融显示动作状态有帮助，但对短时动作变化不敏感，部分触觉响应反而增大误差；下一步优先改进动作变化表达与同指触觉融合。该判断基于 001，尚不能推广至 051。

## v1：显式控制点位移与自回归预测

Git ID：`893b65d82309cacc4ae603743d2eb2e57939919e`（简称 `893b65d`）。

### 1. 输入编码为 token

夹爪状态、触觉和 GS 属性的编码沿用 v0。控制点输入从 3 维扩展为 7 维：

$$
\mathbf{u}_{j,i}
=\left[
\underbrace{\mathbf{C}_{j,i}}_{3},
\underbrace{\mathbf{C}_{j,i}-\mathbf{C}_{0,i}}_{3},
\underbrace{m^\Delta_{j,i}}_{1}
\right]\in\mathbb{R}^{7}
$$

其中位移始终相对窗口初态；仅当初态和当前控制点均有效时，$m^\Delta_{j,i}=1$，否则位移填零。

$$
\mathbf{U}\in\mathbb{R}^{B\times5\times60\times7}
\xrightarrow{\mathrm{MLP}}
\mathbf{C}^{\mathrm{feat}}\in\mathbb{R}^{B\times5\times60\times128}
$$

拼接后的条件仍为：

$$
\mathbf{Z}\in\mathbb{R}^{B\times5\times66\times128}
$$

### 2. 自回归状态与条件做 Cross-Attention

每一步重新编码上一帧预测坐标和初始 GS 属性：

$$
\widehat{\mathbf{X}}_0=\mathbf{X}_0
$$

$$
\mathbf{H}_{k-1}
=\phi_{\mathrm{GS}}\left(
\left[\widehat{\mathbf{X}}_{k-1},\mathbf{A}^{\mathrm{GS}}_0\right]
\right)
\in\mathbb{R}^{B\times N\times128}
$$

其中初始 GS 属性 $\mathbf{A}^{\mathrm{GS}}_0$ 为坐标之外的 11 维。

$$
\mathbf{Q}_k=\mathbf{H}_{k-1}+\mathbf{e}^{\mathrm{pred}}_k
$$

$$
\mathbf{F}_k
=\operatorname{CrossAttentionBlocks}
(\mathbf{Q}_k,\mathbf{M}_k,\mathbf{M}_k)
\in\mathbb{R}^{B\times N\times128}
$$

条件 memory $\mathbf{M}_k$、2 层 4 头 attention、null token 和掩码均沿用 v0。预测 $t+k$ 仍只读取截至 $t+k-1$ 的条件。

### 3. 输出单步位移并递推

位移头输出相对上一帧预测位置的单步位移：

$$
\delta\widehat{\mathbf{X}}_k
=\operatorname{Linear}_{128\rightarrow3}
\bigl(\operatorname{LayerNorm}(\mathbf{F}_k)\bigr)
$$

$$
\boxed{\widehat{\mathbf{X}}_k
=\widehat{\mathbf{X}}_{k-1}+\delta\widehat{\mathbf{X}}_k}
\qquad
\boxed{\widehat{\mathbf{X}}\in\mathbb{R}^{B\times5\times N\times3}}
$$

训练时使用自身预测递推，不以真实中间帧替换；五步之间不截断梯度。零初始化位移头、初始点数、继承属性及归一化规则沿用 v0。

注意：控制点输入位移相对初态，GS 输出位移相对上一帧预测。

## v2：GS 几何属性自回归与渲染监督

Git ID：`340f87335de2ab8c7074f98da9c91eb7de4dc247`（简称 `340f873`）。

### 1. 输入与网络交互

7 维控制点、动作和触觉编码沿用 v1；条件 token 为 $B\times5\times66\times128$。每一步将上一帧预测的位置、对数尺度、WXYZ 四元数与初始不透明度、SH DC 拼成 14 维，经过共享逐点 MLP 编码为 $B\times N\times128$，再通过 2 层、4 头 Cross-Attention 读取条件。

预测 $t+k$ 只读取截至 $t+k-1$ 的条件，保留有效性与补齐掩码、null token。GS 之间没有额外的邻域聚合或 Self-Attention。

### 2. 三个增量头与五步递推

设 $\boldsymbol\ell=\log\mathbf s$，三个零初始化输出头分别预测位置增量、对数尺度增量和世界坐标系旋转向量：

$$
\widehat{\mathbf X}_k=\widehat{\mathbf X}_{k-1}+\delta\mathbf X_k,
\qquad
\widehat{\boldsymbol\ell}_k=\widehat{\boldsymbol\ell}_{k-1}+\delta\boldsymbol\ell_k
$$

$$
\widehat{\mathbf q}_k=\operatorname{normalize}
\left(q(\delta\boldsymbol\theta_k)\otimes\widehat{\mathbf q}_{k-1}\right)
$$

旋转增量以 Hamilton 四元数左乘，仅改变高斯轴方向，不绕世界原点旋转中心。位置处于归一化坐标系，渲染前恢复世界坐标与尺度。五步使用自身预测递推，不截断梯度。

| 输出属性 | 维度 | 处理方式 |
|---|---|---|
| 位置 | $B\times5\times N\times3$ | 单步位移累加 |
| 对数尺度 | $B\times5\times N\times3$ | 单步增量累加 |
| WXYZ 四元数 | $B\times5\times N\times4$ | 增量左乘并归一化 |
| 不透明度 logit、SH DC | $B\times5\times N\times1$、$B\times5\times N\times3$ | 继承初态 |

每个样本保持初始有效点数；目标点数可变，不要求跨帧或预测与目标逐点对应。当前渲染仅使用 SH DC。

### 3. Loss 与 best 指标

$$
\mathcal L=\mathcal L_{\mathrm{RGB}}+0.1\mathcal L_{\alpha}
+0.01\operatorname{mean}\left[(\widehat{\boldsymbol\ell}_k-\boldsymbol\ell_0)^2\right]
$$

RGB、alpha 均为 L1，目标来自真实目标 GS 的渲染，而非原始 RGB 图像。预测与目标使用相同的三个固定标定相机、黑色背景和 SH0 渲染，仅在目标 HDF5 深度有效区域计算损失；区域外不受 RGB／alpha 监督。有效像素、可用视角、时刻和样本逐级平均；全无效样本排除，全无效 batch 跳过更新。

best 指标为 $\mathcal L_{\mathrm{RGB}}+0.1\mathcal L_{\alpha}$，不含尺度正则。Chamfer 与深度误差仅用于诊断，不参与反向传播；当前没有旋转或邻域形变正则。

### 4. 固定窗口实验

001 使用 ep1 起点 85、ep2 起点 73；051 使用 ep0 起点 109、132。逐 scene 独立训练，每个 scene 共 400 次更新，两个窗口交替、各更新 200 次。下表为第 400 次的平均渲染指标相对变化，负值表示改善：

| Scene | 相对复制初态 | 相对历史仅位置模型 |
|---|---:|---:|
| 001-rope | −54.1% | +28.9% |
| 051-cube | −67.4% | −5.3% |

历史仅位置对照采用独立五帧预测、3 维控制点和 Chamfer 训练，因此不是只改变尺度／旋转的严格消融。结果见 `audit/gs-geometry-overfit-v1/report.html`。

001 第 380 次渲染指标为 0.05431，第 400 次反弹至 0.06344；历史仅位置模型为 0.04922。ep2 第 1／5 步 Chamfer 分别为 1.711／2.921 mm，历史仅位置模型为 1.724／1.840 mm。当前结果出现后续帧位置误差增长和边缘毛糙。

001 两个窗口第五步的最大轴尺度倍率 P95 为 1.55／1.65，三轴尺度乘积倍率中位数为 1.77／1.90。仅在推理时恢复初始尺度，平均渲染指标反而恶化约 8.8%；这不等同于固定尺度重新训练的结果。

### 5. 简要结论

当前版本优于复制初态，但 001 未优于历史仅位置模型，且 400 次训练仍有震荡。尺度扩张、未监督区域的外溢和自回归反馈是值得检查的因素，尚不能确定各自因果贡献，也不能认定增加训练次数即可解决。后续优先研究可复用的 GS 编码／解码与形变先验；新架构尚未确定。


## v3：P2 结构化 GS 与共享锚点条件位移预测

- 正式主线 Git ID：`64b1320d3718e7d05d590ce854b8aeb05f14236f`（简称 `64b1320`）。本节汇总当前正式 P2，中间架构实验保留在 Roadmap 与历史报告中。
- 预测长度为 $H$，支持 H5/H10/H15；批大小为 $B$，初态补齐点数为 $N$，特征维度为 128。锚点数 $A\le128$，本轮真实数据中 $A=128$。

### 1. GS 编码与结构建模

GS 输入仍为 14 维：中心坐标 3 + 对数尺度 3 + WXYZ 四元数 4 + 不透明度 logit 1 + SH DC 颜色 3。每个预测步重新编码当前状态，第一个预测步使用真实初态，后续使用上一帧预测的位置；其余 11 维属性继承初态：

$$
\widehat{\mathbf X}_0=\mathbf X_0,\qquad
\mathbf G_{k-1}=\operatorname{Concat}
(\widehat{\mathbf X}_{k-1},\mathbf{Attr}_0)
\in\mathbb R^{B\times N\times14}
$$

$$
\mathbf G_{k-1}\xrightarrow{\mathrm{MLP}}
\mathbf F_{k-1}\in\mathbb R^{B\times N\times128}
$$

随后经过两级结构建模：

| 模块 | 网络交互 | 输出维度 |
|---|---|---|
| LocalGSBlock | 每个 GS 与初态选定的 16 个近邻交互，使用当前预测位置的相对 XYZ | $B\times N\times128$ |
| Structural Anchor Attention | 池化到 FPS 锚点，经 2 层、4 头相对位置 Attention，再以残差插值回 GS | $B\times N\times128$ |

FPS 锚点身份、局部邻域及 RBF 插值权重由初态确定，整个 rollout 固定；Attention 使用的相对位置随预测状态更新。结构模块的残差输出投影零初始化，得到结构增强特征 $\mathbf S_{k-1}\in\mathbb R^{B\times N\times128}$。

### 2. 动作与触觉编码为条件 token

| 输入 | 编码前维度 | 编码后维度 |
|---|---|---|
| 两只夹爪的位姿与开口 | $B\times H\times2\times13$ | $B\times H\times2\times128$ |
| 60 个控制点的 7 维特征 | $B\times H\times60\times7$ | $B\times H\times60\times128$ |
| 四路触觉及掩码、时间差 | $B\times H\times4\times1153$ | $B\times H\times4\times128$ |

夹爪状态 13 维为位置 3 + 旋转矩阵 9 + 原始开口 1，表示同帧观测状态，不是未来控制命令。控制点沿用 v1 的 7 维表示：

$$
\mathbf c_j=
[\mathbf q_j,\;\mathbf q_j-\mathbf q_0,\;m_j\land m_0]
\in\mathbb R^7
$$

初态无效时只禁用差分，当前有效坐标仍保留。每指触觉仍展平为 $1153=384$ 幅值 $+384$ 有效掩码 $+384$ 空间掩码 $+1$ 样本年龄；有效零响应保留，每指一个 token，当前模型未使用 `tactile_xyz`。

$$
\mathbf Z=\operatorname{Concat}(\mathbf A,\mathbf C,\mathbf T)
+\mathbf E_{\mathrm{id}}+\mathbf E_{\mathrm{time}}
\in\mathbb R^{B\times H\times66\times128}
$$

每时刻仍为 $2+60+4=66$ 个条件 token。

### 3. GS 与条件做 Cross-Attention

预测第 $k$ 帧时，以上一预测状态的结构特征加预测时间编码作为 Query，读取前 $k$ 个条件时刻，并加入 null token：

$$
\mathbf Q_k=\mathbf S_{k-1}+\mathbf e_k^{\mathrm{pred}},\qquad
\mathbf M_k=\operatorname{Concat}
(\mathbf z_{\mathrm{null}},\operatorname{Flatten}(\mathbf Z_{0:k}))
\in\mathbb R^{B\times(66k+1)\times128}
$$

$$
\mathbf U_k=\operatorname{CrossAttentionBlocks}
(\mathbf Q_k,\mathbf M_k,\mathbf M_k)
\in\mathbb R^{B\times N\times128}
$$

采用 2 层、4 头 Attention，包含残差、LayerNorm 和前馈网络。掩码排除无效条件，null token 保证条件全部无效时仍可计算。当前跨模态 Attention 尚无显式 GS–动作／触觉相对位置偏置。

预测 $t+k$ 只读取截至 $t+k-1$ 的条件。H15 是动作／触觉逐步到达的在线条件预测，不是初态已知未来触觉的开放环预测。

### 4. 共享锚点 Decoder 与自回归反馈

融合后不再由每个 GS 独立输出位移，而是利用每个锚点初态固定的 16 点邻域池化，再经共享位移头预测：

$$
\mathbf V_k=\operatorname{AnchorPool}(\mathbf U_k)
\in\mathbb R^{B\times A\times128},\qquad
\mathbf d_k=\operatorname{Linear}_{128\rightarrow3}
(\operatorname{LayerNorm}(\mathbf V_k))
\in\mathbb R^{B\times A\times3}
$$

初态确定各向同性 RBF 权重，带宽 $\sigma_0$ 由初态锚点间距确定，当前带宽倍率为 1：

$$
w_{ia}=\operatorname{softmax}_a
\left(-\frac{\|\mathbf x_{i,0}-\mathbf p_{a,0}\|_2^2}{2\sigma_0^2}\right)
$$

$$
\boxed{\widehat{\mathbf x}_{i,k}
=\mathbf x_{i,0}+\sum_a w_{ia}\mathbf d_{a,k}}
\qquad
\boxed{\widehat{\mathbf X}\in\mathbb R^{B\times H\times N\times3}}
$$

每步输出相对初态的累计位移，不把累计量重复加到上一预测。上一帧预测通过重新编码影响下一步，跨步梯度不截断：

$$
\widehat{\mathbf X}_{k-1}
\rightarrow\text{结构编码与条件融合}
\rightarrow\mathbf d_k
\rightarrow\mathbf X_0+\mathbf W\mathbf d_k
$$

位移头零初始化，初始输出复制初态；相同锚点位移对应一致整体平移。点数保持初态点数，补齐点排除；尺度、旋转、不透明度和颜色继承初态。未来 GT GS 仅用于监督，不进入编码器或 Decoder Query。

### 5. Loss 与 best 指标

$$
\boxed{\mathcal L=\mathcal L_{\mathrm{CD}}+\mathcal L_{\mathrm{RGB}}
+0.1\mathcal L_{\alpha}+0.01\mathcal L_{\mathrm{Scale}}}
$$

Chamfer 沿用 v0 的归一化坐标、双向非平方欧氏距离，按 $H$ 个预测时刻与 batch 等权平均，允许目标点数变化，不要求逐点对应。RGB／alpha 采用同相机下预测 GS 与真实目标 GS 渲染的 L1，在目标 HDF5 深度有效区域内逐级平均，区域外不受该渲染损失监督。目标不是原始 RGB 图像。

当前 P2 继承初态尺度，因此尺度正则为零。深度 loss 权重为 0，只保留诊断。完整训练 `best.pt` 按验证联合指标 $\mathcal L_{\mathrm{CD}}+\mathcal L_{\mathrm{RGB}}+0.1\mathcal L_\alpha$ 选择，不含尺度正则，**不是按最低 Chamfer 选择**。

### 6. 001-rope 完整数据训练结果

2026-10-26，H5／H15 从零训练，计划8轮，分别在完成7轮／3轮后主动停止。均使用完整训练／验证划分、batch=1、seed=42、AdamW lr与weight decay均为$10^{-4}$，无单手预训练阶段。以下为已保存验证记录，预测时刻平均、episode等权；两种H不是等预算对照。

| 实验／权重 | 轮次 | CD（mm） | RGB MAE | Alpha MAE | 联合验证指标 |
|---|---:|---:|---:|---:|---:|
| H5 复制初态 | — | 2.010 | 0.06405 | 0.06299 | 0.08315 |
| H5 best | 6 | 2.023 | 0.06301 | 0.05992 | 0.08189 |
| H5 last | 7 | 2.263 | 0.07046 | 0.06611 | 0.09148 |
| H15 复制初态 | — | 3.530 | — | — | — |
| H15 best／last | 3 | 3.960 | 0.11338 | 0.11473 | 0.15007 |

best按联合指标选取；“—”表示不适用或未摘录。权重均通过CPU读取与有限值检查，哈希未变，未重新推理。源码、Job及输出路径见 [H5记录](../audit/p2-001-h5-e8-submission-v1/submission.json)（`b977178`／30695）与 [H15记录](../audit/p2-001-h15-e8-submission-v1/submission.json)（`ca84a6e`／30666）。

### 7. 简要结论

P2在固定窗口中改善了运动连贯性，但本轮完整数据训练的验证CD未优于复制初态。H5 best的渲染略有改善，第7轮退化；H15联合指标下降而CD增大，说明渲染收益未同步转化为几何收益。

目前不能据此认定模型静态或动作／触觉编码失效。先保留best／last，做冻结条件消融与逐步误差诊断，再选择独立架构改动，暂不恢复全量训练。
