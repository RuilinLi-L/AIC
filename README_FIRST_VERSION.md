# AIC 赛题第一版鲁棒 CLIP 微调

本版本严格固定 OpenAI CLIP ViT-B/32，训练时只更新轻量残差 Adapter 和分类参数；当前这版加入了 EMA 软筛样与 effective-number 类别重加权，不读取官方测试集，不使用额外图像数据，也不使用模型集成。

## 目录约定

训练集需要按类别文件夹组织：

```text
train/
├── 0001/
│   ├── image_001.jpg
│   └── image_002.jpg
└── 0002/
    └── image_003.jpg
```

当前目录中的 `clip-ViT-B-32` 是 Sentence-Transformers 保存格式，脚本会自动从 `clip-ViT-B-32/0_CLIPModel` 加载 `CLIPModel` 与 `CLIPProcessor`。如果使用标准 Hugging Face 目录（根目录直接包含 `config.json`），也可以直接传入该目录。

## 安装

```powershell
python -m pip install -r requirements.txt
```

建议在 Linux/WSL 或带 CUDA 的 Python 环境中执行大规模训练。Windows PowerShell 和 Linux 的脚本参数相同。

## 训练

```powershell
python train.py `
  --train-dir <初赛训练集路径> `
  --model-dir .\clip-ViT-B-32 `
  --output-dir .\outputs\screen_reweight_v1 `
  --rounds 3 `
  --clean-fraction-schedule 0.90,0.85,0.80 `
  --score-momentum 0.7 `
  --weight-floor 0.25 `
  --weight-cap 2.0 `
  --class-weight-beta 0.9999
```

继续训练已有 checkpoint（类别目录必须保持一致）：

```powershell
python train.py `
  --train-dir <初赛训练集路径> `
  --model-dir .\clip-ViT-B-32 `
  --resume .\outputs\screen_reweight_v1\model.pt `
  --output-dir .\outputs\screen_reweight_v1_continue
```

训练过程会：

1. 从训练集内部建立确定性分层验证划分；
2. 使用 CLIP 图文相似度得到初始样本质量；
3. 在每个类别内部按轮次计划筛选高可信样本，并保留软权重；
4. 训练残差 Adapter 与余弦分类参数；
5. 使用可信度加权 CE/MAE、effective-number 类别重加权和 CLIP 蒸馏；
6. 在下一轮训练前用 EMA 重新筛样。

验证集仍来自含噪训练集，因此输出的 `noisy_validation_label_accuracy` 仅用于诊断，不代表官方测试准确率。

## 预测

```powershell
python predict.py `
  --checkpoint .\outputs\screen_reweight_v1\model.pt `
  --model-dir .\clip-ViT-B-32 `
  --test-dir <初赛测试集路径> `
  --output .\pred_results.csv
```

输出没有表头，格式为：

```text
image_a.jpg, 0001
image_b.jpg, 0123
```

如果测试目录有多级子目录且存在同名文件，可以使用 `--filename-mode relative`。如果类别文件夹不是数字名称，使用 `--class-map class_map.json` 指定提交编号，例如：

```json
{
  "sparrow": 1,
  "oak": 2
}
```

## 规则与数据安全

- `train.py` 只接收训练目录，不会打开测试目录；
- 不联网下载模型，`local_files_only=True`；
- Pillow 解码失败的图片会记录警告并使用黑图占位，便于批处理不中断；
- 最终推理只使用一个 CLIP 模型和一个分类流程；
- 请将生成的 `pred_results.csv` 按赛事要求压缩提交。
