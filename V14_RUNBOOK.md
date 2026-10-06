# V14：expanded 的图像质量增强与全量重训

线上基线为用户报告的 V13 expanded 两视图试跑包 **65.9598**；每天可评测两次。V14 的线上效果尚未确认。

2026-10-01 00:55（服务器 CST）已启动 screen `2688660.v14_pipeline`：GPU0训练，GPU2评估基线。代码快照为 `/home/mcxu/lrl/AIC/.runs/v14_20261001_005121`。全项目113项回归与额外4项shell流程测试通过；真实短测4/8 worker分别45.57/88.26张每秒，选8 worker、batch256、prefetch1、pin memory关闭，峰值显存19.19GiB。短测不包含完整教师审计和验证，不能直接推算整条流程时长。

## 配方

候选 `expanded_robust` 继承 V13 expanded：官方 CLIP ViT-B/32，320 输入，全部12层 Q/K/V/out LoRA，rank16/alpha32，24轮日程、有效 batch256，原学习率、EMA及动态标签修复不变。第二视图先独立以50%概率缩小最长边到随机320–640像素（不放大），再以50%概率进行质量65–95、subsampling2的JPEG编码，然后轻裁剪与翻转。第一视图、教师审计和验证沿用原流程。

继续复用已验证身份的224冻结特征。V14模型独立初始化于官方底座，吞吐测试使用V13权重只是测试速度；不将测试中更新过的权重写入正式训练。

## 自动流程

`scripts/run_v14_pipeline.sh` 在独立screen中依次完成：

1. V13 expanded的独立正式评估，与V14验证训练同时运行。
2. V14全部24轮完成后，进行同一留出集的五折选轮/校准及三项压力评估。
3. 原图宏准确率至少提高0.2个百分点，且原图尾类与每项压力宏准确率下降均不超过0.3个百分点，才选择V14；否则选expanded。
4. 冻结所选轮次、24轮学习率日程、视图和类别偏置，从官方底座对全部可用训练行重训。重建近邻/原型；不再验证或调参。
5. 预测并验证37444行CSV及ZIP。

不同阶段有完成标记；原checkpoint和选择记录绑定sha256。重启原pipeline会恢复最近完整轮次，并跳过已完成阶段。不要手工改动完成标记、来源checkpoint、选择记录或已有运行配置。

## 服务器启动

环境沿用 `/data/mcxu/conda-envs/aic/bin/python`，主项目 `/home/mcxu/lrl/AIC`，输出根 `/data/mcxu/AIC/outputs`。先检查GPU及活跃进程。

先运行 `benchmark_v14.py`，比较4/8 worker、预取1、关闭pin memory、batch256；只有实际OOM才回退128并累积2。报告路径为 `v14_benchmark.json`，pipeline会读取它。

```bash
cd /home/mcxu/lrl/AIC
CUDA_VISIBLE_DEVICES=0 AIC_PIN_MEMORY=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/mcxu/conda-envs/aic/bin/python -u benchmark_v14.py \
  --model-dir /home/mcxu/lrl/AIC/clip-ViT-B-32 --train-dir /home/mcxu/lrl/AIC/data/train \
  --data-manifest /data/mcxu/AIC/outputs/v7/dataset_manifest_v7.json \
  --checkpoint /data/mcxu/AIC/outputs/v13_expanded/best_model.pt \
  --feature-cache /data/mcxu/AIC/outputs/v13_expanded/cache/frozen.npy \
  --output /data/mcxu/AIC/outputs/v14_benchmark.json --device cuda --steps 8 --warmup-steps 2

# 0与2仅为示例；启动时以检查过的GPU为准。
AIC_GPU=0 AIC_BASELINE_GPU=2 screen -dmS v14_pipeline \
  bash scripts/run_v14_pipeline.sh
```

实际部署可用代码快照作为 `AIC_PROJECT_DIR`，同时显式设置原项目的 `AIC_DATA_DIR` 和 `AIC_MODEL_DIR`，避免后续代码修改影响长任务。

## 查看和产物

- screen：`screen -r v14_pipeline`；按 `Ctrl+A` 后按 `D` 脱离，训练继续。
- 运行状态：`/data/mcxu/AIC/outputs/v14_pipeline/status.json`。
- 验证训练：`/data/mcxu/AIC/outputs/v14_expanded_robust/train.log`。
- 基线评估：`/data/mcxu/AIC/outputs/v14_pipeline/baseline.log`。
- 全量重训：`/data/mcxu/AIC/outputs/v14_refit/train.log`。
- 选择记录：`/data/mcxu/AIC/outputs/v14_selection.json`。
- 最终提交：`/data/mcxu/AIC/outputs/v14_refit/pred_results.zip`。

前两轮后日志会给出当前训练阶段剩余时间估计，不包含未来评估和全量重训。服务器负载会影响实际耗时。全量重训后的效果只能通过线上评测确认。

## 检查

```bash
CUDA_VISIBLE_DEVICES="" AIC_PIN_MEMORY=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/mcxu/conda-envs/aic/bin/python -m unittest discover -s tests -v
```

测试覆盖320增强、复现和不放大、48个LoRA模块梯度与保存重载、修复标签损失隔离、断点恢复、运行配置兼容、24轮评估、来源绑定、候选/回退选择、两/四视图及完整CSV/ZIP路径。小数据测试只验证工程行为，不代表真实精度。
