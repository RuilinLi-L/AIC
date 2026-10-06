# V12：增强 LoRA + 单卡大批量

基于官方 CLIP ViT-B/32，320×320 输入、V9 soft_teacher、固定数据划分和 12 轮训练。最后 4 个视觉层的 Q/K/V/输出投影使用 rank 16、alpha 32 的 LoRA，共 16 个模块、393,216 个视觉可训练参数。学习率仍为 LoRA 1e-5、分类头 5e-5。训练和推理保持单模型。

默认训练 batch 256、梯度累积 1、16 workers、prefetch factor 2。训练、EMA、校准和预测在 A100 上使用 BF16 autocast，概率与统计使用 FP32；统计累加器使用 FP64。批量入队保持同类样本顺序与队列覆盖规则。

V12 文件单独新增，V11 可继续运行。以下 GPU 短测速、正式训练、校准和预测都由你选卡执行。`N` 请换成物理 GPU 编号；本轮交付没有启动这些任务。

## 1. 可选短测速

```bash
cd /home/mcxu/lrl/AIC
nvidia-smi
AIC_GPU=N AIC_V12_STAGE=benchmark bash scripts/run_v12_train.sh
```

依次比较 `64×4`、`128×2`、`256×1`，每组默认预热 3 次优化器更新、计时 10 次更新。报告为 `/data/mcxu/AIC/outputs/v12_benchmark.json`，包含原图数/秒、峰值显存和最快批量；吞吐没有将两路增强重复计为两张原图。

使用真实训练图像和训练原型，以及独立构造的第 5 轮历史状态和填满的特征队列，覆盖对比损失、soft teacher 和标签修复分支。每组重新初始化可训练参数、优化器、EMA 和历史数组；不读取或写入正式训练断点。测速反映该受控负载，不代表完整一轮耗时或模型质量；初始化、全量 EMA 检查和验证另计。

如需更稳定的测速，可增加 `AIC_BENCHMARK_STEPS=30`；可用 `AIC_WORKERS=8` 或 `16` 重测数据加载速度。首次加载缓存及计算原型在计时前进行，可能需要等待。

## 2. 正式训练与恢复

```bash
AIC_GPU=N bash scripts/run_v12_train.sh
```

默认 `256×1`。根据测速结果手动换配置，脚本自动计算累积次数：

```bash
AIC_GPU=N AIC_BATCH_SIZE=128 bash scripts/run_v12_train.sh
AIC_GPU=N AIC_BATCH_SIZE=64 bash scripts/run_v12_train.sh
```

也可以显式提供 `AIC_GRAD_ACCUM`，但乘积必须等于 256。`AIC_WORKERS`、`AIC_PREFETCH_FACTOR` 可调整；`AIC_EVAL_BATCH_SIZE` 默认 256，单独控制训练中的全量 EMA 检查和验证。worker 数为 0 时不传 prefetch 参数。

重复训练命令会从最近完整轮次的 `resume_latest.pt` 恢复，可在此时换卡、改微批量和数据加载参数。改变 LoRA、损失、精度或数据配置会被拒绝。微批量会影响对比损失、梯度投影和随机增强，不能期待不同 batch 的训练逐位一致。

启动默认要求至少 24 GiB 空闲显存，这只是检查门槛，不是峰值显存保证。需要时可显式设置 `AIC_MIN_FREE_MIB`。OOM 后先降低 batch；若发生在全量 EMA 或验证阶段，降低 `AIC_EVAL_BATCH_SIZE`。失败轮次从头重跑，已完成轮次保留。

每 50 批及阶段末尾显示进度、原图吞吐和显存；每轮分别记录 `train_seconds`、`audit_seconds`、`validation_seconds`。前两轮包含额外的全训练集 EMA 检查，通常更慢。

## 3. 校准、比较与提交 ZIP

训练完成后：

```bash
AIC_GPU=N bash scripts/run_v12_evaluate_and_predict.sh
```

复用原有外层五折、内层四折的轮次/TTA/类别偏置校准和压力条件报告。V9、完成后的 V11 用同一套 BF16 流程重评，分别保存到 `v12_reference_v9_soft_teacher`、`v12_reference_v11_320`，并复用与 checkpoint、数据、预处理及精度匹配的缓存。尚未完成的对照会明确提示待比较；之后可以只重跑评估：

```bash
AIC_GPU=N AIC_V12_STAGE=evaluate bash scripts/run_v12_evaluate_and_predict.sh
AIC_GPU=N AIC_V12_STAGE=predict bash scripts/run_v12_evaluate_and_predict.sh
```

评估或预测显存紧张时添加 `AIC_EVAL_BATCH_SIZE=128`。校准和预测必须使用相同计算精度。

产物均位于 `/data/mcxu/AIC/outputs/v12_lora_fast`：

- `metrics.json`、`resume_latest.pt`、逐轮 checkpoint 和验证 logits。
- `strict_eval.json`、`model.pt`：校准结果及对应的同一套单模型权重。
- `comparison_v9_soft_teacher.json`、`comparison_v11_320.json`：对照完成后生成，包含宏、整体及尾类指标。
- `pred_results.zip`：仅含无表头的 `pred_results.csv`，强制检查 37,444 行、文件名顺序、四位类别编号及 ZIP/CSV 内容一致。

测试图像只在最终预测阶段读取。线上是否超过 V9 的 56.6259，需要提交新 ZIP 后确认。

## 已完成验证

A100 上隐藏 CUDA 的 CPU 全回归 82 项通过。额外用本地官方 CLIP 权重检查了 320 输入的前向/反向：16 个 LoRA 模块均有非零且有限的梯度。检查还覆盖 CPU BF16、原有损失的数值一致性、队列覆盖、轮末恢复、精度缓存失效、预测入口及 37,444 行格式校验。

真实 GPU 大批量峰值显存、训练吞吐和线上分数尚未测量；真实模型及提交 ZIP 要等你运行实验后生成。
