# AIC 噪声标签细粒度识别：鲁棒 CLIP V4

当前线上最佳为 **59.4264**（V3：5% 留出 + HFlip TTA + 验证集类别偏置校准）。V4 在保留这套分类头的基础上，加入：

- CLIP 视觉编码器末端注意力层 LoRA，而不再只训练缓存特征上的 Adapter；
- 训练参数 EMA，最终保存为一套单模型权重；
- 基于训练集历史预测的低质量样本动态改标；
- LoRA 特征对原始冻结 CLIP 特征的余弦锚定，限制灾难性遗忘；
- 轻量图像增强、标签平滑、余弦学习率和 AMP；
- 同时记录全验证集与高质量验证子集，减少噪声验证标签对选 epoch 的干扰。

## V4 一键运行（Windows CMD）

先运行均衡版；它使用最后 4 层的 Q/V LoRA，风险和显存占用较低：

```bat
run_v4_train.cmd
```

训练完后自动完成原图推理、验证集 HFlip 检验、类别偏置校准、最终 HFlip 推理和 ZIP 压缩：

```bat
run_v4_submit.cmd robust_visual_v4_lora
```

最终文件：

```text
.\outputs\robust_visual_v4_lora_tta_calibrated\pred_results.zip
```

第二次可以跑激进版。它使用最后 6 层 Q/K/V/Out LoRA，并微调对应 LayerNorm 与视觉投影；动态改标也更积极：

```bat
run_v4_aggressive_train.cmd
run_v4_submit.cmd robust_visual_v4_aggressive
```

4070 Laptop 8GB 上均衡版默认 `batch-size=16, accumulation=2`。Windows 下训练脚本默认 `workers=0`，避免长时间运行后 DataLoader 子进程耗尽系统内存；图像预处理已改为等价的 float32 CLIP 变换。若显存不足，把 batch size 改为 8、accumulation 改为 4，有效 batch 不变。不要同时运行两个训练任务。若训练中断，提交脚本会自动使用逐轮保存的 `best_model.pt`。

所有路径都固定官方 OpenAI CLIP ViT-B/32。旧 `train.py` 仍提供冻结骨干的 V3；新 `train_lora.py` 在同一单模型中微调视觉 LoRA。训练只读取官方训练集，不使用测试集训练、额外图像、其他基础模型或模型集成。

## V3 修正

- 冻结 CLIP 特征只提取一次并缓存，后续 Adapter 实验不再每个 epoch 重跑主干；
- 视觉原型只使用训练划分，并通过类内迭代裁剪降低错标污染；
- 验证划分不再进入原型或筛样，逐 epoch 记录指标并保存最佳权重；
- AdamW 状态跨轮保留，不再每轮重新初始化；
- 默认使用连续质量权重下的 `soft_ce`，避免 500 类 MAE 对低置信样本几乎没有梯度；
- 默认固定 logit scale=30，避免旧 V2 中温度冲到上限 100；
- 强裁剪、硬 CE/MAE 切分和特征漂移均改为显式可选项。

## 安装

```bat
python.exe -m pip install -r requirements.txt
```

## 推荐训练（Windows CMD）

下面使用的是 CMD 的续行符 `^`。它必须是每行最后一个字符，后面不要加空格。

```bat
python.exe train.py ^
  --train-dir ".\初赛数据集\train" ^
  --model-dir ".\clip-ViT-B-32" ^
  --output-dir ".\outputs\robust_visual_v3_soft" ^
  --feature-cache ".\outputs\frozen_clip_train.npy" ^
  --device cuda ^
  --batch-size 64 ^
  --workers 4 ^
  --epochs 8 ^
  --rounds 3 ^
  --clean-fraction-schedule 0.90,0.85,0.80 ^
  --loss-mode soft_ce ^
  --augmentation none ^
  --prototype-keep-fraction 0.7 ^
  --prototype-iterations 2 ^
  --initial-logit-scale 30 ^
  --max-logit-scale 50 ^
  --low-confidence-multiplier 1.0 ^
  --distill-weight 0 ^
  --drift-weight 0
```

第一次会生成约 100 MB 的 `frozen_clip_train.npy`。以后使用相同 `--feature-cache` 的实验会按样本数和特征维度直接复用，不做耗时的逐文件哈希。

训练目录会产生：

- `model.pt`：不重复保存冻结 CLIP，约 1.5 MB；
- `metrics.json`：每个 epoch 的训练/验证指标与最终选中 epoch；
- `sample_quality.npy`：训练样本质量权重；
- `frozen_features.npy`：仅当未显式指定共享 feature cache 时生成。

不要从 `screen_reweight_v1` 或 `robust_visual_v2` 续训 V3。只有恢复同一 V3 配置时才使用 `--resume`。

## 可选消融：GCE 处理低置信样本

复用上面的特征缓存，因此这次训练会快很多：

```bat
python.exe train.py ^
  --train-dir ".\初赛数据集\train" ^
  --model-dir ".\clip-ViT-B-32" ^
  --output-dir ".\outputs\robust_visual_v3_gce" ^
  --feature-cache ".\outputs\frozen_clip_train.npy" ^
  --device cuda ^
  --batch-size 64 ^
  --workers 4 ^
  --epochs 8 ^
  --rounds 3 ^
  --clean-fraction-schedule 0.90,0.85,0.80 ^
  --loss-mode hard_gce ^
  --gce-q 0.7 ^
  --low-confidence-multiplier 0.5 ^
  --augmentation none ^
  --initial-logit-scale 30 ^
  --max-logit-scale 50 ^
  --distill-weight 0 ^
  --drift-weight 0
```

先比较两个目录 `metrics.json` 中的 `selected_val_noisy_accuracy`；不要仅凭训练准确率选模型。

## 验证集类别偏置校准

校准仅使用官方训练集内部的验证划分，并通过分层 5 折选择一个标量强度；它不会使用测试标签或重新训练 CLIP。已有测试 logits 时，可以直接生成校准 checkpoint 和新 CSV，无需重复图像推理：

```bat
python.exe calibrate.py ^
  --checkpoint ".\outputs\robust_visual_v3_soft\model.pt" ^
  --train-dir ".\初赛数据集\train" ^
  --feature-cache ".\outputs\frozen_clip_train.npy" ^
  --val-logits ".\outputs\robust_visual_v3_soft\val_logits.npy" ^
  --output-checkpoint ".\outputs\robust_visual_v3_calibrated\model.pt" ^
  --test-logits ".\outputs\robust_visual_v3_soft\test_logits.npy" ^
  --base-predictions ".\outputs\robust_visual_v3_soft\pred_results.csv" ^
  --output-predictions ".\outputs\robust_visual_v3_calibrated\pred_results.csv" ^
  --device cuda
```

## 预测

```bat
python.exe predict.py ^
  --checkpoint ".\outputs\robust_visual_v3_soft\model.pt" ^
  --model-dir ".\clip-ViT-B-32" ^
  --test-dir ".\初赛数据集\test" ^
  --output ".\outputs\robust_visual_v3_soft\pred_results.csv" ^
  --logits-output ".\outputs\robust_visual_v3_soft\test_logits.npy" ^
  --device cuda ^
  --batch-size 64 ^
  --workers 4
```

`--tta hflip` 可以启用单模型水平翻转 TTA，但应先用验证实验确认收益。默认不启用。

如果已经保存未校准的原图 logits，可以只计算翻转视图，避免重复原图推理：

```bat
python.exe predict.py ^
  --checkpoint ".\outputs\robust_visual_v3_tta_calibrated\model.pt" ^
  --model-dir ".\clip-ViT-B-32" ^
  --test-dir ".\初赛数据集\test" ^
  --output ".\outputs\robust_visual_v3_tta_calibrated\pred_results.csv" ^
  --logits-output ".\outputs\robust_visual_v3_tta_calibrated\test_logits.npy" ^
  --tta hflip ^
  --base-logits-input ".\outputs\robust_visual_v3_soft\test_logits.npy" ^
  --device cuda ^
  --batch-size 128 ^
  --workers 4
```

## 当前实测结果与推荐提交

所有数值只来自官方训练集内部的固定分层留出集；校准强度由留出集上的分层 5 折选择。

| 候选 | 原图准确率 | HFlip 平均准确率 | 校准后全留出准确率 | 校准 5 折准确率 |
|---|---:|---:|---:|---:|
| 10% 留出 | 70.6563% | 70.8994% | 71.5314% | 71.2105% |
| 5% 留出 | 70.6462% | **71.2569%** | 71.4736% | **71.3357%** |

当前首选是 5% 留出 + HFlip TTA + 验证集类别偏置校准。它保持了与 10% 留出近似的原图准确率，同时多使用约 5,200 张训练图，并取得更高的 TTA 5 折准确率。可直接提交：

```text
.\outputs\robust_visual_v3_val05_tta_calibrated\pred_results.zip
```

10% 留出候选使用更大的验证集，作为稳健备选保留：

```text
.\outputs\robust_visual_v3_tta_calibrated\pred_results.zip
```

复现 5% 留出训练时，在推荐训练命令中将输出目录改为 `robust_visual_v3_soft_val05_retry`，并增加：

```bat
  --val-ratio 0.05
```

随后验证 TTA 并生成带校准偏置的 checkpoint：

```bat
python.exe evaluate_tta.py ^
  --checkpoint ".\outputs\robust_visual_v3_soft_val05_retry\model.pt" ^
  --train-dir ".\初赛数据集\train" ^
  --model-dir ".\clip-ViT-B-32" ^
  --original-val-logits ".\outputs\robust_visual_v3_soft_val05_retry\val_logits.npy" ^
  --hflip-val-logits ".\outputs\robust_visual_v3_soft_val05_retry\val_hflip_logits.npy" ^
  --output-metrics ".\outputs\robust_visual_v3_soft_val05_retry\tta_metrics.json" ^
  --output-checkpoint ".\outputs\robust_visual_v3_val05_tta_calibrated\model.pt" ^
  --device cuda ^
  --batch-size 64 ^
  --workers 4
```

如果原图测试 logits 已存在，只需计算翻转视图并写出最终预测：

```bat
python.exe predict.py ^
  --checkpoint ".\outputs\robust_visual_v3_val05_tta_calibrated\model.pt" ^
  --model-dir ".\clip-ViT-B-32" ^
  --test-dir ".\初赛数据集\test" ^
  --output ".\outputs\robust_visual_v3_val05_tta_calibrated\pred_results.csv" ^
  --logits-output ".\outputs\robust_visual_v3_val05_tta_calibrated\test_logits.npy" ^
  --tta hflip ^
  --base-logits-input ".\outputs\robust_visual_v3_soft_val05_retry\test_logits.npy" ^
  --device cuda ^
  --batch-size 128 ^
  --workers 4
```

最终压缩命令：

```bat
powershell -NoProfile -Command "Compress-Archive -LiteralPath '.\outputs\robust_visual_v3_val05_tta_calibrated\pred_results.csv' -DestinationPath '.\outputs\robust_visual_v3_val05_tta_calibrated\pred_results.zip' -Force"
```

从 CMD 创建提交压缩包时，需要显式调用 PowerShell：

```bat
powershell -NoProfile -Command "Compress-Archive -LiteralPath '.\outputs\robust_visual_v3_soft\pred_results.csv' -DestinationPath '.\outputs\robust_visual_v3_soft\pred_results.zip' -Force"
```

压缩包内只保留无表头的 `pred_results.csv`，格式为：

```text
image_a.jpg, 0001
image_b.jpg, 0123
```

## V5：双视图一致性与自适应 TTA

V5 保留 V4 文件和输出，新增一套单模型实验入口。默认配置适合 8GB 显存：

```bat
run_v5_train.cmd
run_v5_submit.cmd robust_visual_v5
```

V5 在 5% 分层留出上选择 EMA 权重，并加入两个独立 light 视图的一致性损失。低质量样本只有在训练划分构建的视觉原型给出不同类别且 margin 达到阈值时，才会与时序教师共同参与伪标签修复。训练结束默认从验证最佳 EMA 权重开始进行 2 轮全量官方训练集短训；`best_model.pt` 保留留出集最佳模型，`model.pt` 是最终 full-refit 模型。使用 `--full-refit-epochs 0` 可关闭 full-refit。

提交脚本会强制用最终 checkpoint 重新生成验证原图/HFlip logits，并通过 5 折分层验证搜索原图权重（步长 0.05）和类别偏置强度，再生成：

```text
.\outputs\robust_visual_v5_tta_calibrated\pred_results.zip
```

`predict.py` 新增可选 `--original-weight`。不传时，HFlip TTA 优先读取校准 checkpoint 中的 `calibration.view_weights`；旧 V3/V4 checkpoint 没有该字段时仍使用原来的 0.5/0.5 平均。V5 仍只使用官方 CLIP ViT-B/32、官方训练集和单一模型推理流程。
