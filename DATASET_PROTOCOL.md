# 数据重构协议

本协议定义触觉、controller points 和 GS 的重构数据字段、处理规则及时间与溯源元数据。

## 目录与文件格式

以下为计划目录；脚本在本地编辑并由 Git 跟踪，重构数据仅保存在服务器 `datasets/touch4d-v1/`，不纳入 Git。

```text
touch4d/
├── scripts/rebuild_dataset/
│   ├── rebuild.py          # 数据重构入口
│   └── validate.py         # 数据一致性检查
└── datasets/touch4d-v1/
    └── 001-rope/
        ├── episode_0/
        │   ├── tactile.npz
        │   ├── controller.npz
        │   ├── timestamps.npz
        │   ├── metadata.json
        │   └── gs/
        │       ├── frame_000000.npz
        │       ├── frame_000001.npz
        │       └── ...
        └── ... episode_9/
```

| 文件 | 保存内容 |
|---|---|
| `tactile.npz` | `tactile`、`valid_mask` |
| `controller.npz` | `controller_points`、`controller_valid_mask`、统一夹爪维度但保留原始数值和 slot 顺序的 `T_worlds`、`openings`，以及开口裁剪标记 `opening_clipped[F,2]`（bool，按左手、右手排列；缺失手为 False） |
| `timestamps.npz` | `observation_timestamp_us`、四路完整 raw 触觉时间戳、`tactile_raw_index`、`tactile_age_us` |
| `metadata.json` | 场景与 episode 标识、通道顺序、左右手 slot 映射、处理参数、版本、源文件路径及 SHA-256（含 median 来源） |
| `gs/frame_XXXXXX.npz` | 对应原始帧号的 GS 字段、`frame_id`、`timestamp_us`；各帧点数可不同 |

数值数组使用 NPZ，说明与溯源信息使用 UTF-8 JSON。四路 `raw_tactile_timestamp_us` 在 NPZ 中分别保存为 `raw_tactile_timestamp_us_0` 至 `_3`，各为独立 int64 数组，按约定通道顺序读取，不使用 object 数组或 pickle。GS 文件名使用原始帧号，至少六位、左侧补零。

原始数据继续保留在服务器现有目录，通过元数据引用，不重复复制到重构目录；`audit/` 用于审计脚本与结果，与正式重构数据分开。

## 数据字段与处理规则

以 GS 帧为主轴匹配 controller points；每路触觉取 GS 时刻之前最近的 raw 样本，扣除对应基线，并附带时间差和有效掩码。

触觉值：`tactile = maximum(raw - median, 0)`，保留连续正值，负值截零，不设接触幅值阈值；截零不改变有效掩码。

重构后的触觉数据字段：

- `tactile[F, 4, 12, 32]`：float32，连续非负触觉值，无效位置填 0。
- `valid_mask[F, 4, 12, 32]`：bool，对应位置的有效掩码，用于区分有效零响应与无效填零。

`F` 为当前 episode 的 GS 帧数；两者按同一 GS 帧序列对齐，四路顺序固定为 `tactilel_left`、`tactilel_right`、`tactiler_left`、`tactiler_right`。时间戳、原始索引和基线来源作为元数据保留，并记录在预处理审计文件中。

掩码规则：每路独立取不晚于 GS 时刻的最近 raw 样本，时间差在 0–75 ms 内有效（75 ms 暂定）。像素须位于已确认区域且 raw、median 均为有限数；按零起始编号屏蔽第 31 列，裁剪第 12–15 行，低响应不作为坏像素依据。最终掩码为时间有效与像素有效的交集，无效位置填 0 并保留掩码。

控制点：`controller_points[F, 2, 30, 3]`，float32，按同一 GS 帧序列对齐；手维顺序为 `[左手, 右手]`，每手 30 点，每点为三维坐标。

控制点掩码：`controller_valid_mask[F, 2, 30]`，bool，缺失手的坐标填 0、掩码为 False。当前 `001-rope` 已发布位姿全部通过审计：episode 0–4 左手全 False、右手全 True，episode 5–9 两手全 True。开口超范围按既定范围裁剪，不因此置为无效；触觉掩码不影响控制点掩码。生成数据时仍保留位姿、开口及输出坐标的有限性检查，并以绝对容差 `1e-6` 检查旋转正交误差、行列式相对 +1 的偏差及齐次矩阵末行误差。

刚体检查具体为 `max(abs(RᵀR-I)) <= 1e-6`、`abs(det(R)-1) <= 1e-6`、`max(abs(T[3,:]-[0,0,0,1])) <= 1e-6`，相对容差为 0。手部绑定未确认、位姿或开口检查失败时，该手当前帧全部无效；生成坐标非有限的点无效。所有无效控制点均填 0 并保留 False 掩码。

两类掩码独立：第 31 列仅屏蔽触觉响应，其控制点几何仍保留；缺失夹爪位姿也不自动否定该手的二维触觉。

时间边界：raw 时间戳须非递减，目标观测时间戳须严格递增，不满足则报错。重复 raw 时间戳取最后一条，无过去样本则该路无效。controller 使用 GS 时刻的位姿，触觉可滞后最多 75 ms（暂定），数组对齐不代表同时采样；完整序列不额外加 `metadata.start_frame` 偏移。

| 触觉通道 | 对应控制点 |
|---|---|
| `tactile[:,0,:,:]` 左手左指 | `controller_points[:,0,0:15,:]` |
| `tactile[:,1,:,:]` 左手右指 | `controller_points[:,0,15:30,:]` |
| `tactile[:,2,:,:]` 右手左指 | `controller_points[:,1,0:15,:]` |
| `tactile[:,3,:,:]` 右手右指 | `controller_points[:,1,15:30,:]` |

## 时间与溯源元数据

当前三类数据涉及 5 条时间轴：4 条独立 raw 触觉时间轴，以及 GS 与 controller 共用的 1 条观测时间轴。以观测时间为目标，各路触觉独立选取最近过去样本；controller 和 GS 没有额外独立的时间轴。

以下字段作为重构数据的时间与溯源元数据保留；时间单位为 Unix 微秒，时间差单位为微秒。

| 字段 | 形状 / 类型 | 含义 |
|---|---|---|
| `observation_timestamp_us` | `[F]`，int64 | 共同观测时间轴，与 `gs_frames[f].timestamp_us` 逐项一致 |
| `raw_tactile_timestamp_us` | 4 个独立的 `[T_s]` int64 数组 | 各路完整原始时间戳，顺序与触觉四路一致，各路长度可不同 |
| `tactile_raw_index` | `[F,4]`，int64 | 每个观测帧选中的各路 raw 样本索引 |
| `tactile_age_us` | `[F,4]`，int64 | 观测时间减去选中 raw 样本时间 |

索引非负时，选中时间为 `raw_tactile_timestamp_us[s][tactile_raw_index[f,s]]`。无过去样本时索引与 age 均为 `-1`；过期样本保留真实索引和 age，但触觉掩码无效。完整 raw 时间戳保留原始顺序及重复项，并与保留的原始触觉数据逐条对应，用于重做对齐、调整门限和追溯来源。

## GS 数据结构

每个 episode 存为长度为 `F` 的逐帧列表 `gs_frames`；第 `f` 帧含 `N_f` 个 Gaussian，点数可变，不假定固定 `[F,N,...]`。

| 字段 | 单帧形状 | 类型 | 含义 |
|---|---|---|---|
| `xyz` | `[N_f,3]` | float32 | 世界坐标，单位米 |
| `log_scales` | `[N_f,3]` | float32 | 原始对数尺度，exp 后为实际尺度 |
| `quaternions` | `[N_f,4]` | float32 | WXYZ 顺序，读取时归一化 |
| `opacity_logits` | `[N_f,1]` | float32 | 原始不透明度参数，sigmoid 后为不透明度 |
| `sh_dc` | `[N_f,3]` | float32 | `f_dc_0..2` |
| `sh_rest` | `[N_f,45]` | float32 | 按原始编号保存 `f_rest_0..44` |
| `frame_id` | 标量 | int64 | 原始 GS 帧号 |
| `timestamp_us` | 标量 | int64 | 对应观测时间戳，Unix 微秒 |

同帧内各字段的相同下标属于同一个 Gaussian；跨帧没有可直接使用的固定对应关系，不定义稳定 `gaussian_id`，不能将相同下标当作同一点。`gs_frames[f]` 与 `tactile[f]`、`controller_points[f]` 对齐，位置与控制点使用同一世界坐标系。法线字段暂不纳入上述重构字段，仍保留在原始 PLY 中；原始 PLY 保留。

读取 GS 时检查上述浮点字段均为有限数、四元数范数非零；异常则报错，不静默删除或修补。SH 暂按原始字段编号存储，通道与基函数的具体排列后续确认。

## 控制点的源数据

```text
T_worlds[F, R, 4, 4]
[ R00 R01 R02  tx ]
[ R10 R11 R12  ty ]
[ R20 R21 R22  tz ]
[  0   0   0   1 ]

openings[F, R]  # 开口距离，单位米
```

- `T_worlds[F, R, 4, 4]`：每帧、每个已发布夹爪的根坐标到世界坐标的齐次变换；左上 `3×3` 为旋转矩阵，最后一列前三项为平移（米），最后一行为 `[0,0,0,1]`。
- `openings[F, R]`：对应帧、对应夹爪的开口距离，单位米。

以上是加载后的统一结构；原始单手文件可能省略夹爪维，分别为 `T_worlds[F,4,4]`、`openings[F]`，加载时增加 `R=1` 维，数值不变。

这里 `R` 是已发布夹爪数量，原始 slot 不等于固定左右手顺序。`001-rope` episode 0–4 为 `R=1`、`[右手]`；episode 5–9 为 `R=2`、`[右手,左手]`，其它场景需单独确认。

生成关系：`openings` 决定夹指开合位置，将固定的 30 个夹指局部点变换为 `p_root`；再通过对应帧和 slot 的 `T_worlds` 计算 `p_world = T_worlds[:3,:3] @ p_root + T_worlds[:3,3]`（此处省略帧和 slot 下标）。最后按 `[左手,右手]` 整理为 `controller_points[F,2,30,3]`；缺失手须明确标记无位姿，不得复用另一只手的轨迹。

## 坐标系与转换

1. **夹指局部坐标**：每个夹指自己的坐标系，随夹指开合运动。
2. **夹爪根坐标**：整个夹爪的共同坐标系，用来描述两个夹指的位置。
3. **世界坐标**：场景的共同坐标系，用来统一描述夹爪与 GS。

以下两个公式均使用夹指局部坐标；`controller_points` 存储世界坐标，单位米。

令 `s=+1` 表示左夹指、`s=-1` 表示右夹指（不是左右手），`r,c` 为未翻转的原始网格行列编号：

```text
触觉像素：p_finger = [s*0.007, -0.056-0.002*c, 0.0235-0.002*r]
控制点：  p_finger = [s*0.007, -0.056-0.002*c, (0.025/12)*(11.5-r)]
```

触觉使用 `r=0..11, c=0..31`；每个夹指的控制点使用 `r=[0,6,11], c=[0,8,16,23,31]`，按行优先排列。两者行间距略有不同，不能视为严格逐点重合。

两者均按以下关系转换（坐标向量按列向量计算）：

```text
q = 0.038 - 0.033*(clip(opening, 0.04, 0.112)-0.04)/0.072
b_left  = [-0.04246242, 0.0835, 0.0097]
b_right = [ 0.04246242, 0.0835, 0.0107]
p_root  = b_side + [s*q, 0, 0] + diag(1,-1,-1) @ p_finger
p_world = R_world_from_root @ p_root + t_world_from_root
```

`opening` 单位为米；`R,t` 来自对应 GS 帧、对应 robot slot 的发布位姿 `T_worlds`，不重复添加 EEF 偏移。`diag(1,-1,-1)` 表示绕 X 轴旋转 180°，将局部 Y/Z 方向反转。因此行号增大时夹指局部 Z 减小、夹爪根 Z 增大，世界 Z 则取决于夹爪姿态；列号增大始终对应靠近指尖。不能将局部或根坐标的 Z 方向直接称为世界或画面的“上方”。
