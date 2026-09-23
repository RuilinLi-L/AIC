# AIC V7 运行手册

V7 不改动 V6 文件或 `/data/mcxu/AIC/outputs/robust_visual_v6`。三个实验使用固定 seed 2026、同一份 clean 验证划分，训练产物分别写入独立目录。

## 1. 审计数据（只需一次）

```bash
cd /home/mcxu/lrl/AIC
bash scripts/audit_v7_data.sh
```

清单应位于 `/data/mcxu/AIC/outputs/v7/dataset_manifest_v7.json`。脚本只读图片并写清单，不修改或重编码原图。

## 2. 前两个实验

建议在 `tmux` 中依次运行：

```bash
bash scripts/run_v7_train.sh v7_clean_drop drop shuffle
bash scripts/run_v7_train.sh v7_partial partial shuffle
```

脚本默认自动使用 `resume_latest.pt` 续训。显存不足时保持有效 batch 256：

```bash
AIC_BATCH_SIZE=32 AIC_GRAD_ACCUM=8 bash scripts/run_v7_train.sh v7_clean_drop drop shuffle
```

## 3. 选择实验 3 的冲突策略

```bash
/data/mcxu/conda-envs/aic/bin/python select_v7_base.py \
  --clean /data/mcxu/AIC/outputs/v7_clean_drop \
  --partial /data/mcxu/AIC/outputs/v7_partial \
  --output /data/mcxu/AIC/outputs/v7/base_selection.json
```

查看最后输出的 `V7_TAIL_CONFLICT_POLICY`。若为 `drop`：

```bash
bash scripts/run_v7_train.sh v7_tail drop repeat-factor
```

若为 `partial`：

```bash
bash scripts/run_v7_train.sh v7_tail partial repeat-factor
```

## 4. 四视图校准与生成提交

按 `validation_metrics.json` 的 `selected.cv_macro_accuracy` 选择最好的两个运行目录，对每个运行：

```bash
bash scripts/run_v7_calibrate_and_predict.sh RUN_NAME
```

每个目录会独立生成 `model.pt`、`calibration.json`、`pred_results.csv` 和 `pred_results.zip`。ZIP 会被强制验证为恰好包含一个 `pred_results.csv`，CSV 会被强制验证为 37,444 行且无表头。

## 5. 比较两个候选与 2,000 次配对 bootstrap

```bash
/data/mcxu/conda-envs/aic/bin/python compare_v7.py \
  /data/mcxu/AIC/outputs/v7_clean_drop \
  /data/mcxu/AIC/outputs/SECOND_RUN \
  --bootstrap 2000 \
  --seed 2026 \
  --output /data/mcxu/AIC/outputs/v7/final_comparison.json
```

主要看严格五折 `cv_macro_accuracy`，并同时核对普通 Macro、Macro NLL、Tail/Mid/Head 与 bootstrap 置信区间。
