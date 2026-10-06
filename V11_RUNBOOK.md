# V11：V9 soft_teacher + 320 输入

完整训练一次，共 12 轮；从官方 CLIP ViT-B/32 初始化。两路训练视图都是 V9 轻增强，输出 320×320；验证和预测同样使用 320，缩放视图短边为 366。位置编码使用 Transformers 自带插值，原始权重形状不变。224 冻结特征和 V9 近邻缓存继续只读复用。

## 训练

把 `N` 替换为当前有足够显存的物理卡号。默认 batch 64、累积 4，启动前要求至少 24 GiB 空闲显存；这个检查不能保证运行中的显存始终充足。

```bash
cd /home/mcxu/lrl/AIC
nvidia-smi
AIC_GPU=N bash scripts/run_v11_train.sh
```

中断后重复命令，从最近完成的一轮恢复。显存不足时，改用 batch 32、累积 8（至少 16 GiB 空闲）：

```bash
AIC_GPU=N AIC_BATCH_SIZE=32 AIC_GRAD_ACCUM=8 bash scripts/run_v11_train.sh
```

允许在断点处切换这两种配置，保持有效 batch 256；切换微批次会影响数值和随机增强分组，不能期待与原配置逐位一致。

## 评估、预测和 ZIP

训练结束后：

```bash
AIC_GPU=N bash scripts/run_v11_evaluate_and_predict.sh
```

复用 V10 的外层五折、内层四折流程，比较 V9 soft_teacher 与 V11 的宏、整体及尾类准确率。原图选择轮次、视图和偏置，已有压力条件只作附加报告。V9 重评写入独立的 `v11_reference_v9_soft_teacher`，V11 校准参数与同一选中轮次的权重一起保存。

产物在 `/data/mcxu/AIC/outputs/v11_320`：`comparison.json`、`model.pt`、`pred_results.zip`。ZIP 只含无表头的 `pred_results.csv`，强制核对 37,444 行及图片名。最终是否超过 V9 的 56.6259，以线上评测为准。

可用 `AIC_V11_STAGE=evaluate` 或 `AIC_V11_STAGE=predict` 单独运行对应步骤；推理显存不足时加 `AIC_BATCH_SIZE=32`。
