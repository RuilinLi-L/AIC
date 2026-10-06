# V16：轻增强与后四层 LayerNorm 的对照实验

V16 保留 V15 全 12 层 attention/MLP LoRA，以两个候选检验可归因的改动。`mlp_light` 恢复 V13 的轻增强；`mlp_light_ln` 在此基础上开放视觉第 9–12 层（代码索引8–11）`layer_norm1/2` 的仿射参数，额外 12,288 参数，学习率 1e-5、weight decay 0。LN 收益尚未实证，不承诺线上涨分。两者均从官方 CLIP ViT-B/32 初始化，不续训 V15 权重。

2026-10-01 21:36:30（服务器CST）已启动 `3443294.v16_pipeline`：A在GPU1、B在GPU3，均batch256；原V14/V15流程继续运行。快照为 `/home/mcxu/lrl/AIC/.runs/v16_20261001_201019`。139个代码文件指纹已核对，186项完整CPU回归通过，最后恢复修复后的6项shell流程复测通过。两种配方均通过官方底座两步更新及精确保存重载检查；B的16个LN张量均更新，冻结参数不变，缺少任一LN拒载。

同批真实图像短测：A为132.51张/秒、B为118.55张/秒，峰值显存均约23.73GiB；不含完整审计与验证，不代表准确率或实际整轮吞吐。原始证据、启动环境和来源指纹位于本地 `exports/v16_startup/` 与服务器 `/data/mcxu/AIC/outputs/v16_startup/`。两轮后以正式日志更新耗时估计。

21:42:05已确认两候选均推进至第1/24轮、50/530批，状态running，无异常回溯；实际初段吞吐A为85.98张/秒、B为88.57张/秒，峰值显存23.68GiB。该初段速率反映当时共享负载，与短测不同；完整验证、选优和线上成绩仍待后续。

## 固定实验与选优

- 验证训练完整 24 轮；有效 batch256。每个配方先用真实数据做 2 步预热、8 步测量，默认微批量256，仅记录实际 CUDA OOM 后允许128×2。benchmark不提供正式训练权重。验证近邻缓存可复制已验证身份的V15缓存，两个候选各持独立副本，训练入口再次核验；本次复用记录为`v16_startup/neighbor_cache_reuse.json`，refit仍独立重建。
- 保留 V15 的数据划分、LoRA、教师审计、鲁棒损失、balanced softmax、重复因子采样和校准逻辑；不再叠加长尾修正。
- 等待 `v14_baseline_expanded` 与 `v15_expanded_mlp` 的正式 `strict_eval.json`、校准 `model.pt`，以二者较强者为基线。候选须原图 macro 至少高0.2个百分点、尾类退步不超过0.3个百分点；压力结果仅诊断，配对 bootstrap2000次仅供参考。
- 仅胜出的 V16 配方全量 refit。冻结选中 epoch、原24轮学习率日程、视图和偏置；refit不重新校准。回退时复用校准基线模型与已存在的匹配提交；没有验证包则只补预测。不存在的基线 refit 不重训、不无限等待。

## 启动与恢复

由服务器独立 `screen` 执行主脚本，屏幕脱离和本轮对话结束后仍自动完成后续流程。示例中的 GPU 必须由当时空闲情况决定，禁止停止既有实验腾显存。

```bash
screen -dmS v16_pipeline bash -lc 'exec env \
  AIC_PROJECT_DIR=/home/mcxu/lrl/AIC/.runs/v16_20261001_201019 \
  AIC_DATA_DIR=/home/mcxu/lrl/AIC/data \
  AIC_MODEL_DIR=/home/mcxu/lrl/AIC/clip-ViT-B-32 \
  AIC_PYTHON=/data/mcxu/conda-envs/aic/bin/python \
  AIC_OUTPUT_ROOT=/data/mcxu/AIC/outputs \
  AIC_GPU_A=1 AIC_GPU_B=3 \
  AIC_BENCHMARK_A=/data/mcxu/AIC/outputs/v16_startup/benchmark_mlp_light.json \
  AIC_BENCHMARK_B=/data/mcxu/AIC/outputs/v16_startup/benchmark_mlp_light_ln.json \
  bash /home/mcxu/lrl/AIC/.runs/v16_20261001_201019/scripts/run_v16_pipeline.sh'
```

示例不是启动确认；实际 GPU、快照、时间与 screen ID 由部署记录给出。`AIC_GPU_A` 必须显式指定；不设 `AIC_GPU_B` 时两候选依次使用同卡，两卡不同时并行。候选分别有日志、锁与阶段标记。默认训练与评估前至少要求32768MiB空闲显存，不足则排队；可显式设置 `AIC_MIN_FREE_MIB`、`AIC_EVAL_MIN_FREE_MIB`。排队和依赖等待各自默认最长48小时，超时明确失败，不自动缩短24轮。

主脚本支持 `--gpu-a/--gpu-b/--output-root/--project-dir/--python/--data-dir/--model-dir/--min-free-mib/--baseline-expanded/--baseline-v15/--candidate-a/--candidate-b/--selection`，也可使用同名对应的 `AIC_*` 环境变量。额外输出覆盖为 `AIC_PIPELINE_DIR`、`AIC_REFIT_DIR`、`AIC_DELIVERY_DIR`。已有两份 benchmark 可用 `AIC_BENCHMARK_A/B` 复用，内容仍严格验证。

同一参数重新执行即恢复：完成阶段跳过，训练读取 `resume_latest.pt`；已供 refit 使用的选择 JSON 不能改变。流水线、候选、训练、评估与交付分别加锁。原 V13/V14/V15 目录只读，不改旧文件、不杀旧任务。

## 状态、预算与交付

输出根默认 `/data/mcxu/AIC/outputs`：

- `v16_mlp_light/`、`v16_mlp_light_ln/`：训练、评估、benchmark及候选阶段标记。
- `v16_pipeline/status.json`：主状态；`mlp_light_status.json`、`mlp_light_ln_status.json`：独立候选状态；GPU等待记录包含卡号、空闲显存与排队时长。
- `v16_pipeline/budget.json`：首次启动时间，恢复不重置。48小时是含准备、排队、依赖等待的预算目标；若官方检查与benchmark先单独执行，启动时传入其准备开始时间的Unix秒值`AIC_PIPELINE_STARTED_AT`，使预算包含前置准备。`eta.json`在两轮后按实测更新，未来共享负载和依赖等待仍未知；预算超出只报告，不暗中短训。
- `v16_selection.json`：冻结选择；`v16_pipeline/paired_comparison.json`：完整对比。
- `v16_refit/`：仅V16胜者重训。
- `v16_delivery/validation/`、`v16_delivery/refit/`：选中验证模型和V16 refit分别出包；回退时可额外有`fallback_refit/`，缺失原因写入`fallback_refit_status.json`。

每包仅含无表头 `pred_results.csv`，默认37444行，严格核对测试文件名、四位类别与ZIP内容。`provenance.json`记录模型、CSV、ZIP与选择文件SHA256，以及真实配方、epoch、视图、alpha、精度；`online_score`保持空值直至平台实测。

## 验证

CPU回归涵盖真正shell的单卡串行/双卡并行、失败与恢复、重复运行锁、基线等待、只训练胜者、回退预测、模型与提交绑定、显存队列和预算计时。正式启动前另跑官方底座更新/重载检查及真实图像benchmark。测试通过与吞吐不等于精度已提升。
