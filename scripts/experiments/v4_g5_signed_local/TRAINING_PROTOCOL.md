# G5-v1 Direct / Propagate 正式训练协议

这是工程入口，不是训练结果。冻结模型基准为
`51ab5c455929128c66941d3d3f75c1d4421c9507`。三个 G5 模型文件、生产
V4/G4、Dataset、Renderer、Loss 和固定 RBF 均不修改。G4 Best
CD=2.88409 mm、Joint=0.119000 仅为上下文参考。

## 协议与配对

两种配置均为 001-rope、H15/Δ1、8 Epoch、seed42、batch1、AdamW
lr=1e-4、weight_decay=1e-4、Norm1 clipping、CD+RGB+0.1Alpha。
复用 G4 `perform_epoch` 和生产 `train_epoch/evaluate_loader`，不是另写 Loss
或训练框架。独立观察器调用原 G5 `return_trace=True` 并原样返回预测；
诊断与 Loss/Best 无关，CPU 测试比较裸模型和观察器的预测及梯度。

冻结 G0 config/manifest 的 SHA256 由既有 G4 protocol 校验。
1514 Train：ep1/2/4/5/7/8、stride1；100 Val：ep3/6、stride5；
不读取 Test，不过滤人工接触区间，不限制 batches/windows。
原始三相机、1280×720 gsplat 与原始 Mask/归一化不变。

逐 Epoch 使用原 DataLoader generator seed42+epoch。预检查克隆真实
loader 的 generator 和 sampler 设置，通过只读索引代理实际遍历八轮顺序，
包含 DataLoader iterator 消耗 base-seed 的行为；不读取全部 GS，也不消耗
原训练 generator。运行时再由 G4 CheckedStream 逐窗口核对。

相同 seed42 下逐项比较 G4 共享参数与 Direct Mapping；局部最后投影全零。
CPU 合成 Batch 上验证完整 H15 全属性与从头 G4 精确一致。该检查不是
新的真实 GPU 验收。正式模型只能从头初始化；不接受 G4 Best、Smoke 权重
或任意 warm-start 参数。恢复只接受自己的正式完整 Epoch 格式。

## Checkpoint 与恢复

Direct/Propagate 使用不同正式 version，均与 G0/G4/G5 Smoke 不兼容。
identity 包含 variant、Source SHA、规范化 Run 绝对路径、完整训练/模型/
Processor 配置、Manifest、八轮顺序及 Hash、共享初始化证据和环境。
保存完整模型/Optimizer、Python/NumPy/CPU/CUDA RNG、每轮真实窗口顺序、
逐窗口更新标志、访问数、成功更新数、验证指标与 Best。

- `initial.pt` 与 `epoch_01.pt`…`epoch_08.pt` 原子发布且不可覆盖。
- `last.pt` 指向最新完整提交；`best.pt` 指向完整 Val100 的 episode 等权
  Joint 最低、并列时最早的一轮。未做完整验证前不制造 initial Best。
- 验证要求 ep3/ep6 样本数及 render_samples 与冻结 Manifest 完全相符，
  并重算 episode 等权 Joint，拒绝 sample 加权或部分验证。
- Linux `flock` 防止同 Run 并发写入。每轮前后检查干净的冻结源码。
- 中断后只读取完整不可变 Epoch 链，恢复模型、Optimizer 和所有 RNG。
  未提交 Epoch 的访问/更新不累计，整轮从原 seed/order 重跑；原日志与
  `.tmp` 证据不删除。过期同 Run 别名可恢复，外来格式/Run/较新别名拒绝。
- 跨 variant、source、Run、配置、Processor、Manifest、环境均拒绝续训。
  若 `initial.pt` 尚未耐久保存即中断，不自动覆盖重建，需使用新的 Run。
- CPU replay/RNG/Optimizer 精确恢复有测试。**不承诺 CUDA 逐位复现**。
  数据 Manifest 固定窗口与来源，不声称对所有 Dataset 文件做字节扫描。

## 诊断与只读评估

训练/验证每轮保存 G4 全局/每手贡献、每手 Direct/Propagated Seed、
Far 几何 Softmax 最大值/Entropy/输入尺度、Self/Entropy、传播差异，
以及局部 RBF 后 GS 均值和去均值 RMS。诊断是有效观测的 batch 均值与
最大值，不参与选模。被屏蔽手不参与统计。

`evaluate.py` 只接受完成8轮的 Direct/Propagate Last 及其选出的 Best，
同协议重算四个 G5、G4 Best、G0 Best 和 Copy 的完整 Val100。
保存 ep3/start135、ep6/start110、ep7/start105 的 H1–H15 签名 XYZ、
Near/Middle/Far、局部误差、16NN 比率、各手/G4/局部分支诊断及 JSON/CSV。
局部误差是区域预测点到完整 GT 点集的单向最近距离，不能称作材料运动
误差；全体 CD 仍是原始双向非平方距离。空间图不等于材料拓扑，16NN
比率不是物理应变。保留 Far≈0.999 与近 Uniform 的诊断，不修改公式。

## 仅展示：未来批准后启动

先将批准的最终 SHA 同步到干净服务器 Worktree；运行期间禁止更新源码。
两个 Run 输出目录必须尚不存在；父目录为各模型独立 bucket。
以下是服务器 Bash 命令模板，**本轮不执行**：

```bash
SOURCE=/data1/userdata/tcweng/projects/touch4d-worktrees/v4-g5-signed-local-v1
SHARED=/data1/userdata/tcweng/projects/touch4d
REF=/data1/userdata/tcweng/experiments/touch4d/v4-001-h15-e8/run-30995
SHA=REPLACE_WITH_APPROVED_FORMAL_COMMIT
DIRECT=/data1/userdata/tcweng/experiments/touch4d/v4-g5-direct-v1-training/run-001
PROP=/data1/userdata/tcweng/experiments/touch4d/v4-g5-propagate-v1-training/run-001

# CPU-only readiness; reports must be new and separate from training Runs.
CUDA_VISIBLE_DEVICES='' "$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_g5_signed_local.train \
 --variant direct --root "$SHARED/datasets/touch4d-v1" --depth-root "$SHARED/datasets/touch4d-depth-v1" \
 --g0-reference "$REF" --output "$DIRECT" --expected-sha "$SHA" --preflight-only \
 --preflight-report /ABSOLUTE/NEW/AUDIT/preflight_direct.json

# ONLY AFTER EXPLICIT APPROVAL; one GPU per job. Check queue/quota first.
sbatch --job-name=g5-direct-h15-e8 --export=ALL,SOURCE="$SOURCE",EXPECTED_SHA="$SHA",VARIANT=direct,RUN_OUTPUT="$DIRECT",G0_REFERENCE="$REF" \
 --output=/data1/userdata/tcweng/experiments/touch4d/v4-g5-direct-v1-training/direct-%j.log \
 "$SOURCE/scripts/experiments/v4_g5_signed_local/train.sbatch"
sbatch --job-name=g5-propagate-h15-e8 --export=ALL,SOURCE="$SOURCE",EXPECTED_SHA="$SHA",VARIANT=propagate,RUN_OUTPUT="$PROP",G0_REFERENCE="$REF" \
 --output=/data1/userdata/tcweng/experiments/touch4d/v4-g5-propagate-v1-training/propagate-%j.log \
 "$SOURCE/scripts/experiments/v4_g5_signed_local/train.sbatch"
```

同一 Run 续训仅增加 `RESUME=1`，仍必须是原 SHA/配置；不能移到另一 Run。
CPU 预检查不得与 `--resume` 同用。预检查不创建训练目录、不构造 Optimizer。
filesystem 保守保留4GiB配对空间，另人工核查个人 quota；不清理任何历史文件。
没有自动提交、自动重试、自动续训或超参数搜索。
