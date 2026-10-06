# AIC V10：图像质量泛化训练与严格折外评估

V10 只使用当前阶段官方训练图像和官方 CLIP ViT-B/32。它沿用 V9 选中版本的噪声策略、固定清单与划分、12 轮 LoRA 训练，只对第二个训练视图随机加入缩小和 JPEG 重编码。测试图像只在最终预测阶段读取。V10 不做全量重训；提交模型就是用完整验证集选择轮次后校准的那一套单模型权重。

## 运行条件

默认等待 V9 后续流程生成 `/data/mcxu/AIC/outputs/v9/selection.json`。V10 训练脚本会从该文件读取 `selected_run`，继承 `v9_coverage` 或 `v9_soft_teacher` 的设置；文件不存在时会直接报错。它只读复用 V9 coverage 的近邻缓存，且校验缓存中的数据签名。

如果要提前开始，可以显式固定一个**已经完成 12 轮**的 V9 版本，例如 `AIC_V10_BASE=v9_coverage`。此时不需要 `selection.json`，但 V10 只对照 coverage 策略；若最后 `soft_teacher` 获胜，这次训练不能算作“仅改变第二视图”的 V9 获胜版本对照。断点恢复时保持相同的 `AIC_V10_BASE`。最终比较与生成 ZIP 仍需等待 V9 的 `selection.json`。

在 A100 上先检查显卡，再把以下示例中的 `5` 改为当前空闲卡号。脚本要求显式指定物理卡号，启动时至少要有 12 GiB 空闲显存。V9 和 V10 训练不要同时运行。

```bash
cd /home/mcxu/lrl/AIC
nvidia-smi
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/mcxu/conda-envs/aic/bin/python -m unittest discover -s tests -v
AIC_GPU=5 bash scripts/run_v10_train.sh
```

提前固定 coverage 策略时，把训练命令改为：

```bash
AIC_V10_BASE=v9_coverage AIC_GPU=<空闲物理卡号> bash scripts/run_v10_train.sh
```

如果在 V9 训练期间执行这条命令，应使用不同的 GPU；这会改变原计划中 V9 与 V10 不同时占用训练资源的条件，并可能产生显存或计算资源竞争。

训练写入 `/data/mcxu/AIC/outputs/v10_robust_views`。中断后重复同一命令会从 `resume_latest.pt` 恢复。已有断点时，脚本会拒绝在同一目录从头训练，防止混用旧文件。训练命令不会读取测试图像，也不会生成提交 ZIP。

## 评估和生成 ZIP

训练完成后运行：

```bash
cd /home/mcxu/lrl/AIC
AIC_GPU=5 bash scripts/run_v10_evaluate_and_predict.sh
```

脚本先把 V9 获胜版本和 V10 放进同一套严格五折评估：每个外层保留折都排除在轮次选择、TTA 配置选择和类别偏置拟合之外。中心加翻转与等权四视图在内层折间选择；JPEG、缩小和组合压力条件只用于评分，不参与任何参数拟合。V9 的重评产物放在独立的 `v10_reference_<V9运行名>` 目录，不改写 V9 输出。

随后脚本将完整验证集参数写入 V10 的 `model.pt`，生成并核验 `pred_results.zip`（仅一个无表头、37,444 行的 `pred_results.csv`）。无论比较门槛是否通过，ZIP 都会生成。检查以下文件：

```bash
cat /data/mcxu/AIC/outputs/v10_robust_views/comparison.json
unzip -l /data/mcxu/AIC/outputs/v10_robust_views/pred_results.zip
```

若只想重跑评估或预测，可分别设置 `AIC_V10_STAGE=evaluate` 或 `AIC_V10_STAGE=predict`。`comparison.json` 中的推荐标记只是固定验证集与压力条件下的判断；线上提升必须以新 ZIP 的实际评测结果确认。
