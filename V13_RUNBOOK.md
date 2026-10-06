# V13：三组对照与最优方案全量重训

V12 线上基线：2026-09-30 10:26:46 提交，58.8292 分，排名 281。V13 尚未完成正式训练或线上评测；70 分以上是目标，不是已取得的结果。

## 配方

| 名称 | 训练 | 标签处理 | 视觉 LoRA |
| --- | --- | --- | --- |
| strength | 24 轮，LoRA 1e-4，head/Adapter 5e-4 | 保留 V12 流程，修复权重按 24 轮日程增长 | 最后 4 层 Q/K/V/out，rank 16，alpha 32 |
| dynamic | 同 strength | 周期质量更新与完整替换修复目标 | 同 strength |
| expanded | 同 dynamic | 同 dynamic | 全部 12 层；最后一层 1e-4，向前逐层乘 0.8 |

共同使用官方 CLIP ViT-B/32、320 输入、原有轻增强、5% warmup、余弦学习率、EMA、有效 batch 256 和现有去重/重复因子采样。不混合不同实验的模型或预测。

动态配方在第 1、2、4、6……轮末检查每一条训练样本。下一轮更新质量、可信集合和类别先验，并清空历史对比队列。前两轮沿用 V12 初始质量逻辑；第 3 轮起最低质量权重为 0.05。第 5 轮起，两次检查类别一致、当前两个视图一致、置信度至少 0.70，且原型支持或近邻支持度至少 0.50 的错标签候选可以被修复。每个原始类别最多选 floor(15%×样本数) 条，小类别上限至少为 1，不强制选满。

修复样本只使用停止梯度的 EMA 软标签监督，权重在第 5–8 轮由 0.125 增至 0.5；移除该样本的原标签 CE、GCE、旧标签对比队列贡献及重复 soft-teacher 项。表征锚定和视图一致性保留。每次检查重新决定修复集合，证据不足时可以撤回。跨类重复图仍受原候选标签集合约束。

## 环境与数据

服务器项目 `/home/mcxu/lrl/AIC`，Python `/data/mcxu/conda-envs/aic/bin/python`，输出 `/data/mcxu/AIC/outputs`。现有 750 类、148695 张训练图及 37444 张测试图属于本轮数据；只在预测入口读取测试目录。

依赖沿用现有 A100 环境；本轮验证版本记录在 `requirements_v13.txt`：Python 3.11.16、PyTorch 2.14.0、torchvision 0.29.0、Transformers 4.57.6、Pillow 12.3.0、NumPy 2.4.6、tqdm 4.70.1。运行脚本不安装或修改环境。完整重现先生成训练清单：

```bash
cd /home/mcxu/lrl/AIC
bash scripts/audit_v7_data.sh
```

已有 `/data/mcxu/AIC/outputs/v7/dataset_manifest_v7.json` 时无需重做。新实验独立生成带官方底座权重指纹的冻结特征缓存、带训练池指纹的近邻缓存。旧 V12 缓存和模型不被覆盖；首次准备缓存会额外耗时。冻结特征是逐图的无标签编码，可供全量重训复用；重训的近邻参考池和原型一定重新计算。

## 1. 启动三组训练

先执行 `nvidia-smi`，将 N 换成你选定的物理 GPU 编号。以下命令各在一个终端运行；若并行，请分别指定不同的可用显卡。

```bash
cd /home/mcxu/lrl/AIC
AIC_GPU=N bash scripts/run_v13_train.sh strength
AIC_GPU=N bash scripts/run_v13_train.sh dynamic
AIC_GPU=N bash scripts/run_v13_train.sh expanded
```

产物分别保存在 `v13_strength`、`v13_dynamic`、`v13_expanded`。不要让两个进程同时写同一个实验目录。脚本检查空闲显存至少 24 GiB，但这不保证实际峰值显存足够；expanded 的显存需求更高。

显存不足时减小微批量，保持有效 batch 256：

```bash
AIC_GPU=N AIC_BATCH_SIZE=128 AIC_EVAL_BATCH_SIZE=128 bash scripts/run_v13_train.sh expanded
```

也支持 `AIC_BATCH_SIZE=64`；累积次数自动计算。中断后重复命令，从最近完整轮次恢复。允许改变微批量、worker 和评估批量；不允许改变配方、数据、底座权重、训练阶段和学习率日程。切换微批量不保证逐位一致。

可通过 `AIC_WORKERS`（默认 16）、`AIC_PREFETCH_FACTOR`（默认 2）、`AIC_PROJECT_DIR`、`AIC_OUTPUT_ROOT`、`AIC_DATA_DIR`、`AIC_MODEL_DIR`、`AIC_PYTHON`、`AIC_MANIFEST` 覆盖环境路径。训练日志每 50 批输出进度、吞吐与显存；每轮有训练、教师检查和验证计时。正式 GPU 吞吐和峰值显存需由实际运行确认。

共享服务器若出现 `cuMemHostAlloc: Failed to allocate physical memory` 或 `pin memory thread` 错误，可关闭锁页内存并减少数据加载预取后重启该实验：

```bash
AIC_GPU=N AIC_PIN_MEMORY=0 AIC_WORKERS=4 AIC_PREFETCH_FACTOR=1 bash scripts/run_v13_train.sh strength
```

`AIC_PIN_MEMORY` 仅接受 `0` 或 `1`，默认 `1`。该开关只改变主机到显卡的数据传输方式，不改变模型与训练配置；降低 worker 数量也有助于减少并发内存需求。A 组首轮前中断时没有可恢复的轮次检查点，但已完成的冻结特征缓存可以复用。

## 2. 评估与冻结选择

三组 24 轮均完成后：

```bash
AIC_GPU=N bash scripts/run_v13_evaluate_and_predict.sh
```

执行同一留出集的选轮次、固定两/四视图和类别偏置校准，并报告原图、resize384、jpeg75、combined 条件。五折仅用于模型选择与校准，训练并不是五折独立训练，验证标签也不是人工干净标签。

脚本读取已有 `v12_lora_fast/strict_eval.json` 和 `model.pt` 作为基线。候选须同时满足：原图宏准确率提升至少 0.5 个百分点，原图尾类及每项压力条件宏准确率下降不超过 0.5 个百分点。合格候选按宏准确率、宏 NLL、参数量排序；完全同分再按配方名确定顺序。无合格候选时回退到 V12 的 12 轮配方。

输出 `/data/mcxu/AIC/outputs/v13_selection.json`，包括逐项比较、所选配方和轮次、日程、权重/数据标识、固定视图和类别偏置。只需重做选择时可用无 GPU 的命令：

```bash
AIC_V13_STAGE=select bash scripts/run_v13_evaluate_and_predict.sh
```

选择记录及其来源 checkpoint 必须保留，修改后不能继续原来的全量重训断点。不要重新评估后覆盖一个正在被重训使用的选择记录。

## 3. 第四次训练：全量重训

```bash
AIC_GPU=N AIC_V13_STAGE=refit bash scripts/run_v13_train.sh
```

自动读取选择记录，无需手工填选中的配方。从官方底座重新初始化，纳入全部可用训练清单行，包括原验证图像；重建原型和近邻，不读取此前微调后的参数初始化。训练停止在已选轮次，学习率仍遵循原来的 24 轮日程；V12 回退配方遵循 12 轮日程。

本阶段不执行验证或重新选择参数，直接保存最后一轮 EMA 单模型，并附上之前冻结的校准偏置。输出 `/data/mcxu/AIC/outputs/v13_refit/model.pt`。这次重训的效果只能通过真实线上评测确认，原验证集已经参与训练，不能继续用于报告无偏精度。

## 4. 预测与提交

```bash
AIC_GPU=N AIC_V13_STAGE=predict bash scripts/run_v13_evaluate_and_predict.sh
```

生成 `/data/mcxu/AIC/outputs/v13_refit/pred_results.zip`，ZIP 中仅有无表头 `pred_results.csv`。校验 37444 行、四位类别编号、文件名和顺序、无重复文件名以及 ZIP/CSV 一致性。预测必须使用与校准相同的计算精度；这里只使用一个 checkpoint 的多视图推理。

## 验证范围

全项目 CPU 回归 85 项通过，其中 V13 8 项行为测试覆盖：48 个 LoRA 模块的有限非零梯度、底座冻结、保存/加载结果一致、已修复样本不受原标签损失影响、修复教师停止梯度、修复条件与按类上限、连续两次检查、训练/验证隔离、断点恢复、选优门槛、缓存标识，以及全量重训到 CSV/ZIP 的短流程。

额外在真实官方 CLIP 权重上完成两次 CPU 参数更新及保存/重载检查：expanded 的 48 个模块梯度有限且非零，视觉可训练参数 1,179,648，head/Adapter 515,713，输出维度为 750；重载前后推理输出逐位一致。CUDA 在这些测试中被隐藏，未启动正式 GPU 训练。

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/mcxu/conda-envs/aic/bin/python -m unittest discover -s tests -v
```

测试中的合成数据指标仅验证工程行为；V13 的正式模型、提交文件和线上成绩要在上述 GPU 流程完成后产生。
