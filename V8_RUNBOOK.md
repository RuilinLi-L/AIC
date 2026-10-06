# AIC V8 运行手册

V8 在 V7 `v7_tail` 的固定数据划分、partial 冲突策略、尾类重复采样和单 CLIP ViT-B/32 分类器上增加折外近邻可信度。V8 产物只写入 `/data/mcxu/AIC/outputs/v8_neighbors`；它直接校准并提交选中的验证阶段模型，不进行全量重训。

```bash
cd /home/mcxu/lrl/AIC
/data/mcxu/conda-envs/aic/bin/python -m unittest discover -s tests -v
nvidia-smi
AIC_GPU=3 bash scripts/run_v8_train.sh
AIC_GPU=3 bash scripts/run_v8_calibrate_and_predict.sh
```

运行前根据 `nvidia-smi` 将 `AIC_GPU=3` 改为空闲且显存充足的卡号。训练脚本自动从 `resume_latest.pt` 续训，冻结特征直接复用 V7 partial 缓存。训练会生成近邻缓存、逐轮断点、`best_model.pt` 和验证指标。第二个脚本生成 `model.pt`、折外校准结果、`pred_results.csv`、`pred_results.zip` 与相对 V7 的 `comparison.json`。

`comparison.json` 中的 `recommended_for_submission` 仅表示同划分折外验证通过预先指定的门槛；线上分数仍以赛事评测为准。无论该字段为何值，ZIP 都会生成。ZIP 内只有无表头、37,444 行的 `pred_results.csv`。
