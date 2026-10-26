# touch4d 协作规则

## 工作目录与同步

- 本地项目为 `D:\Amax\Projects\touch4d`，对应 AMAX `/data1/userdata/tcweng/projects/touch4d`。两端同一相对路径表示对应文件；操作前确认实际 worktree、分支、HEAD 与项目根目录。
- 独立 worktree 放在项目之外，例如本地 `D:\Amax\Projects\touch4d-worktrees\`、服务器 `/data1/userdata/tcweng/projects/touch4d-worktrees/`。不得把不同分支源码混放或创建嵌套仓库。
- 代码在本地修改、测试、提交一次；经授权推送 GitHub，并推送服务器的独立同步引用，再由对应服务器 worktree 执行 `git merge --ff-only`。无需服务器再次提交。
- 同步前检查两端 `git status`、分支、HEAD、远程与 worktree。发现意外未提交修改或分叉立即停止对应同步；不覆盖、不删除、不强制 stash。可在独立 worktree 继续不受影响的工作。
- 禁止 `git reset --hard`、`git clean -fd`、强制推送和改写共享历史。模型、数据集、环境及重型日志只留服务器并由 Git 排除。
- 本文件纳入版本管理，取代以前“AGENTS.md 仅本地保留”的约定。

## 分支、采用与 PR

- `main` 只保存已经验证并正式采用的稳定主线；当前正式模型是 P2。P3-v0 是未采用的归档实验，不能混入正式 P2。
- 新主架构实验从最新 `main` 创建独立 `codex/` feature branch。实验期间允许在该分支直接修改 `scripts/training/`；不要为多个架构在同一工作区复制整套源码。
- 验证工具优先新增于 `scripts/experiments/` 并复用正式模块。一次只引入可归因的研究修改，保留可比较 baseline。
- 实验成功仍须完成测试、研究验收与代码 review，再通过 PR 请求合并 main；是否采用由项目负责人确认。不得因指标看似改善而自动合并。
- 实验失败或收益不足时不合并 main，保留分支、准确提交 SHA 和证据索引。PR 是合并请求，不是保存实验源码的文件夹。
- 开发流程为独立分支 → 实现与测试 → 授权后的 commit/push → PR → review → 负责人批准的合并。未明确授权时不提交、推送或合并；已有本轮授权无需重复询问。

## scripts 职责

- `training/`：正式模型、编码/解码、训练、推理、loss、渲染、评估及必要的轻量兼容入口。
- `experiments/`：Oracle、固定窗口、ablation、诊断和专用评估，复用 `training/`，不复制框架。
- `experiments/archive/`：历史版本与复现入口索引；完整源码由 Git commit/branch 保留。
- `tests/`：统一测试；`data_process/`、`dataloader/`、`rebuild_dataset/`：共享数据基础设施。
- 模块重命名或迁移须同时审计 import、CLI、mock/patch、测试发现及 checkpoint 路径。保持 `scripts.training.model.PositionPredictor` 与旧 checkpoint 的加载/恢复语义，不为目录整洁删除兼容模式。
- 默认从仓库根目录执行 `python -B -m unittest discover -s scripts/tests -t .`。环境缺失导致的 skip 须单独报告；不得删除测试、修改旧期望值或放宽容差掩盖回归。

## 实验与数据完整性

- 实验记录准确代码 SHA、运行命令、配置、数据划分、seed、窗口/帧号、指标、checkpoint 和证据路径。工程通过、固定窗口表达能力、泛化结论分别报告。
- 固定窗口过拟合不能直接证明泛化；不根据测试结果挑选训练数据或改变冻结的验收标准。
- 不把未来 GT GS 或当前预测时尚不可获得的触觉泄漏到预测状态。保留既定时间因果性、独立点集、padding 与有效性掩码协议。
- 未经许可不覆盖历史结果、不删除实验分支、不修改共享服务器已有 checkpoint。读旧 checkpoint 做回归时核对前后哈希；新实验使用独立目录。
- GPU 工作只通过集群调度；先检查现有作业与环境。普通目录重构不启动训练，不改变模型数学、预测、loss 或训练协议。

## 文件与范围

- 创建任何文件或文件夹前先告知用户完整路径；批量生成时说明目标目录及命名规则。
- 未获许可不写入 Markdown。获准更新指定文档时在该范围内整理，不扩写无关文档。
- 保留用户未提交和未跟踪文件；仅提交本轮授权内容。重构 scripts 时不移动 `docs/`、`audit/`、`runs/`、`datasets/` 或历史图像、日志、数据与 checkpoint。
