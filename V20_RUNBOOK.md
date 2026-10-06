# V20：DoRA、LoRA+ 与联合校准选优

这版执行已批准的两候选方案：在同一去重数据划分、官方 CLIP ViT-B/32、24 epoch、seed 2026 和真实 batch256×1 下比较 DoRA 与 LoRA+。V19 三路继续使用原源码及恢复任务；V20 的评估和交付均进入独立目录。目前尚无 V20 线上分数。 2026-10-06 12:22:06（北京时间）已在 A100 的 `v20_joint_20261006` 会话启动独立后台流水线；12:22:54 核验主进程 PID 3444202 存活，DoRA 与 LoRA+ 均处于 `waiting_gpu`，GPU 官方检查和真实 batch 短测尚未开始。最终 V20 专项 **50 项**和 V19 兼容回归 **65 项**均通过，见[验收记录](exports/v20_startup/implementation_verification.json)。

| 候选 | 分辨率 / 放大边 | rank / alpha | A、B 顶层学习率 | 额外设置 |
|---|---|---|---|---|
| dora_rank32 | 320 / 366 | 32 / 64 | 5e-5、5e-5 | 幅值 m 学习率 5e-5、零 weight decay |
| loraplus_rank32 | 320 / 366 | 32 / 64 | 5e-5、2e-4 | B/A 比例 4 |

两路均有 48 个 attention 和 24 个 MLP 适配模块，dropout=0，层学习率衰减 0.8，head 学习率 5e-4。DoRA 的方向范数在 FP32 中计算并停止梯度，偏置不随幅值缩放。旧参数的初始化顺序保持 V19 随机数消费顺序。训练损失、EMA、噪声修复、采样和增强沿用原方案。上述学习率属于本次实验选择，不能从论文结果推断它们一定在本赛道提分。

方法依据为 [DoRA 论文](https://arxiv.org/abs/2402.09353)中的幅值与方向分解，以及 [LoRA+ 论文](https://arxiv.org/abs/2402.12354)中 A/B 不同学习率的思路。比赛限制以[官方赛道规则](https://www.aicomp.cn/tracks/tracks-1/3714.html)为准：只使用官方模型和当期数据，最终为单模型推理。

每个候选和固定 rank32_control 都用同一 V20 联合协议重新评估：outer5 / inner4，从全部 24 个 epoch、两/四视图 TTA 和 alpha=0..1（步长0.05）中选择；最终冻结使用全 holdout 的 inner5。内层使用拼接后的完整 OOF 按类计算宏指标，outer 标签不参与选择，压力条件只作诊断。已有且身份匹配的中心/翻转训练缓存只读复用，仅补算 zoom/zoom_flip；其余缓存保存到新目录。

晋级需要同时满足：native 类宏准确率比固定对照提升至少 0.2 个百分点、尾类下降不超过 0.3 个百分点、整体准确率不下降。合格候选依次按宏准确率、尾类、较低 NLL 排名，完全相同再优先 LoRA+、DoRA、dropout、384。仅获胜者从官方初始权重全量 refit，训练到冻结 epoch，仍使用 24 epoch 调度器，复用冻结校准。没有候选合格则复用原 V18 全量模型及其提交包；历史线上 69.3195 为用户已报告结果，不能当作新验证结果。

后台入口为 [run_v20_pipeline.sh](scripts/run_v20_pipeline.sh)，调度器为 [v20_pipeline.py](v20_pipeline.py)。独立输出路径、服务器快照路径及进程信息记录在 [启动记录](exports/v20_startup/run_paths.json)；[源码清单](exports/v20_startup/source_manifest.json)与[源码包](exports/v20_startup/v20_source.tar.gz)用于核验实际运行版本。

GPU 池只使用物理 0、1、2、3，每次等待至少 71,680MiB 空闲显存，同时遵守 V19 和 V20 的同卡锁。正式训练前依次完成官方模型两步检查与真实 batch256 短测。typed CUDA OOM 记为 resource_infeasible；其他故障记为失败，不降低 batch 或自动重复探测。完整 24 epoch 的 V19 对照或候选未完成时继续等待，不能静默略过。源码、审计、探测和资源配置在启动训练前绑定哈希。

48 小时为总墙钟目标，计时包含准备、显存排队、训练、评估、V19 等待、refit 和预测。超时会显示警告并保留完整实验；不会缩短训练。断点恢复会核验 A/B/m 状态、EMA、参数分组、学习率、优化器步数及调度器；完成的联合评估和 selection 验证后复用，不重写。

运行时查看独立输出目录内 `pipeline/status.json`、各路 `*_preflight_status.json`、`budget.json` 和 `supervisor.log`；训练日志在候选目录，联合评估日志在 pipeline 目录。需要重启时使用封存快照内的 `launch_v20.sh`，保持原目录及配置，不重新封存运行中的源码。

交付包位于 `delivery/refit/`（晋级）或 `delivery/fallback_refit/`（回退），包括无表头、37444 行的 `pred_results.csv`、只含该 CSV 的 ZIP 和 `provenance.json`。交付校验包括文件名、四位类别、完整 calibration、来源与选择哈希、全量训练池和阶段。提交排行榜后才可判断实际提分。

2026-10-06 14:25:28（北京时间）复查：DoRA 在 13:26 取得 GPU3，13:29 已完成官方模型检查和真实 batch256 短测；LoRA+ 仍处于 `waiting_gpu`。两路预检结束后才冻结资源并训练，当前尚无 `train_v20.py` 正式训练进程或新分数。
