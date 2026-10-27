# 001-rope 人工两刀标注

完整保留目录后，直接双击 `index.html`。采用相对图片路径和按需加载的普通 JS 数据文件，不使用 fetch/CDN，支持离线 file://。

若浏览器限制本地资源，可在本目录执行 `python -m http.server 8765 --bind 127.0.0.1`，打开 `http://127.0.0.1:8765/index.html`。仅监听本机；无需安装前端框架。

1. 切换 ep0–ep9，浏览真实三相机 RGB、触觉、开口/运动、GS 几何与距离。左右键逐帧、Shift+左右键跳10帧。
2. 用“当前帧设为第一/第二刀”或整数输入设置草稿，再拖动时间轴上的两条线微调。自动灰虚线和响应跳转只作参考，不会自动填入人工边界。
3. 满足 `0 <= start < end <= total_frames` 后点击“确认本 Episode 两刀”。修改两刀后需要重新确认。不确定时保持 UNREVIEWED，可写备注。
4. 半开区间：接触前 `[0,start)`，操作 `[start,end)`，接触后 `[end,N)`。结束帧是接触后第一帧，允许 N。
5. H10/Δ1 完整操作窗口数为 `max(0,end-start-10)`，因每个窗口需要初态和10目标共11帧。这里仅是人工所选区间的容量，不自动筛选 Dataset。
6. 时长显示优先使用 `timestamp[end]-timestamp[start]`；若 end=N，不编造额外时间戳，只显示首末观测跨度并明确末边界未知。
7. 浏览器自动保存草稿/确认/备注；建议同时下载 JSON。换浏览器、换目录或清理浏览器存储后，导入 JSON 恢复。CSV/文本只列已确认边界，其他为 UNREVIEWED；JSON 额外保留草稿但不将其当作确认标签。

你只需要为每个 Episode 选定两条分界线。全部完成后，点击‘复制标注结果’，把生成的文本发送给我即可。

## 证据与限制

- ep0–4 只显示右手，ep5–9 显示左右手。物理 slot 按 metadata 解码，不能将 slot0 当左手。
- 触觉为原始 median 扣除后截零响应，不是标定力。有效零与缺测不同；Age 超限或一指零响应不代表物理脱离。
- 复用 episode1 审计的数据读取、触觉统计和背景候选方法。候选只是定位辅助，不是正式接触 GT。
- 三相机真实去畸变 RGB，时间戳逐项核对；控制点及 GS 世界坐标。几何图为独立正交示意，不假装投影到 RGB。GS center 不是物理表面，逐帧重建无材料点身份。
- 全部图像按原始帧号导出；GS仅作轻量抽样显示，最近距离计算使用完整GS。
- 本工具只输出一个总体操作区间。多次接触/分离可用备注说明，不适合两刀表达时保留 UNREVIEWED。
- `audit_manifest.json` 记录总帧数、来源、SHA和缺失信息；各 `episodes/ep*/data.json` 和 `observations.npz` 保存证据，`source_integrity.json` 保存源文件读取前后哈希。
- 浏览器测试使用 `?qa=1` 独立存储键，不会将测试边界带入正式标注。

## 生成

在冻结代码 worktree 中运行：

```text
python -B -m scripts.experiments.contact_manual.collect --datasets <datasets> --ep1-audit <contact-audit-ep1-v1> --output <新的空目录>
```

CPU-only；不加载模型、不使用GPU、不修改原始数据。输出目录必须为空，旧报告保持不动。
