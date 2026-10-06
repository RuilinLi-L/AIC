# V15：V14 加全层 MLP LoRA

目标是在约 1–2 天内完成一个候选的验证、选优及全量重训。线上已确认最佳为 V13 expanded 两视图的 **65.9598**；目标为 **72**。线下宏准确率只是选择依据，不能视作线上成绩。

2026-10-01 10:02（服务器 CST）已启动 screen `738705.v15_pipeline`，GPU2训练，GPU6负责后续交付预测；V14保持原流程运行。代码快照为 `/home/mcxu/lrl/AIC/.runs/v15_20261001_092146`。145项回归测试通过；官方真实CLIP两步更新验证了72模块梯度、冻结底座和保存重载；真实图片短测约102.31张/秒、峰值23.73GiB。短测不含完整教师审计和验证，服务器为共享负载，不能直接当作整条流程耗时。

## 固定配方

V15 `expanded_mlp` 从官方 CLIP ViT-B/32 权重初始化。在 V14 的全部 12 层 Q/K/V/out LoRA 外，为每层 MLP 的 `fc1/fc2` 加入 LoRA，共 72 个模块。两类 LoRA 都使用 rank16、alpha32、dropout0；视觉 LoRA 参数共 2,654,208。最后层学习率 1e-4，向前逐层乘 0.8；分类头和 Adapter 学习率 5e-4。

沿用 320 输入、第二视图随机缩小/JPEG、24 轮 warmup/余弦日程、有效 batch256、EMA、动态标签修复、损失和重复因子采样。原型、近邻及表征锚定仍使用核对过身份的 224 冻结特征。当前实验只改变 MLP 适配范围。

官方规则要求当前阶段官方数据、官方 CLIP ViT-B/32 和单模型推理；测试图像仅用于预测。线上为整体 Top-1 准确率，测试类别均衡。规则见 [官方赛题页](https://www.aicomp.cn/tracks/tracks-1/3714.html)。每日可提交两次，阶段成绩取最高分，见 [复赛通知](https://www.aicomp.cn/notice/notice-1/5278.html)。72 是本轮目标，不是本文确认的官方固定晋级线。

## 运行与恢复

服务器 Python 为 `/data/mcxu/conda-envs/aic/bin/python`，数据和官方底座分别为 `/home/mcxu/lrl/AIC/data`、`/home/mcxu/lrl/AIC/clip-ViT-B-32`，输出根为 `/data/mcxu/AIC/outputs`。实际代码快照、GPU 和启动记录见本地 `exports/v15_startup/deployment.json`。

正式训练前运行 `benchmark_v15.py`，使用真实训练图像、2 步预热和 8 步测量。默认 8 workers、batch256、prefetch1、关闭 pinned memory；仅实际 OOM 时回退 batch128、累积2。短测不保存训练权重，正式训练重新初始化。

V15 的 `scripts/run_v15_pipeline.sh` 在独立 screen `v15_pipeline` 中运行。启动前必须检查 GPU 进程，并显式指定训练 GPU；如使用并行交付 GPU，应与训练 GPU 不同。启动时显式设置 `AIC_PROJECT_DIR` 为冻结快照，同时设置 `AIC_DATA_DIR`、`AIC_MODEL_DIR` 为上述原始数据路径。

使用 `screen -r v15_pipeline` 查看；按 Ctrl+A 后按 D 脱离。流水线及训练使用锁防止重复运行。重启使用同一快照、输出目录及参数，保留完成标记和选优来源；不要覆盖正在被 refit 使用的选择 JSON。恢复允许调整微批量/worker 等运行参数，但有效 batch 必须为256，科学配方必须一致。

## 自动流程与选优

1. 完成 V15 的 24 轮验证训练及同口径正式评估。
2. 等待已有 V14 流水线冻结选择；基线为其最终胜出的验证模型，可能是 V14，也可能是 V13 expanded。
3. 核对数据、行键、标签、折、精度和压力条件；比较原图宏准确率、整体准确率、尾类和三项压力条件，并做2000次配对 bootstrap。
4. 只有原图宏准确率提高至少0.2个百分点、原图尾类及各压力条件退步均不超过0.3个百分点，V15才进入全量重训。bootstrap仅供参考，不另设门槛。
5. V15胜出后冻结轮次、原24轮学习率日程、视图和类别偏置，从官方底座全量重训至选定轮次；重建原型和近邻，不在已并入训练的原验证集上重新校准。
6. 未通过时记录回退，复用已有 V14 流水线结果，不启动一次重复的回退训练。

验证模型和全量重训模型分别出包，并保留同一套冻结视图及偏置，便于在线判断全量重训的效果。V14胜者的一对产物优先交付，通过门槛的V15随后交付；按就绪顺序及每日额度上传，不混合模型预测。

## 日志与交付

- 状态：`/data/mcxu/AIC/outputs/v15_pipeline/status.json`
- 验证训练：`/data/mcxu/AIC/outputs/v15_expanded_mlp/train.log`
- 正式评估：`/data/mcxu/AIC/outputs/v15_expanded_mlp/strict_eval.json`
- 冻结选择：`/data/mcxu/AIC/outputs/v15_selection.json`
- 全量重训：`/data/mcxu/AIC/outputs/v15_refit/train.log`
- 两轮后的流程耗时估计：对应训练目录内的 `eta.json`
- 分别出包：`/data/mcxu/AIC/outputs/v15_delivery/`

每份提交 ZIP 仅包含无表头 `pred_results.csv`，共37444行，文件名无重复且类别为有效四位编号。旁边的来源记录包含实际配方、训练阶段、epoch、视图、alpha、精度及 checkpoint/CSV/ZIP 的 SHA256。线上分数在实际获得平台结果后才能填写。

前两轮完成后日志更新耗时估计，包含教师审计；后续评估及重训耗时另行估计。共享服务器负载可能变化，48小时是预算目标，不是完成时间保证。工程短测和合成数据测试不代表真实精度。

## 测试

在部署快照内执行：

```bash
CUDA_VISIBLE_DEVICES="" AIC_PIN_MEMORY=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/mcxu/conda-envs/aic/bin/python -m unittest discover -s tests -v
```

重点覆盖72模块更新、骨干冻结、严格权重重载、断点恢复、来源绑定、胜出/回退、重训隔离以及完整CSV/ZIP路径。测试及启动实测结果写入 `exports/v15_startup/`。
