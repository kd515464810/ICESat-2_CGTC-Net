# ICESat-2 DualTask 数据预处理与数据路径说明

## 1) 原始数据存放路径
将 CSV 放到：

```bash
data/raw/
```

每个 CSV 至少包含列：
- `alongtrack_distance`
- `elevation`
- `classed_pc_flag`
- `source_type`

---

## 2) 数据预处理命令
手动运行：

```bash
python data_utils/prepare_icesat_dualtask.py \
  --raw_dir data/raw \
  --out_dir data/processed \
  --bin_size_m 0.5 \
  --seq_length 128 \
  --stride 64 \
  --max_points_per_bin 32
```

输出路径：
- `data/processed/train.pkl`
- `data/processed/val.pkl`
- `data/processed/test.pkl`
- `data/processed/manifest.json`

---

## 3) 训练时自动触发预处理（可选）
如果你还没有 `train.pkl/val.pkl`，可以让训练脚本自动调用预处理：

```bash
python train_icesat_dualtask.py \
  --prepare_if_missing \
  --raw_dir data/raw \
  --processed_dir data/processed \
  --train_pkl data/processed/train.pkl \
  --val_pkl data/processed/val.pkl
```

---

## 4) 测试路径
```bash
python test_icesat_dualtask.py \
  --test_pkl data/processed/test.pkl \
  --checkpoint checkpoints/cgtc_net.pt \
  --out_dir outputs/test
```

测试输出：
- `outputs/test/predictions.csv`
- `outputs/test/sample_*.png`


---

## 5) 一键环境检查（建议先跑）
```bash
bash scripts/install_icesat_deps.sh
python scripts/smoke_test_runtime.py
```

如果 `smoke_test_runtime.py` 输出 `forward/backward ok`，说明核心依赖和模型主流程可运行。

## 6) 是否还需要额外库？
- **必须**：`requirements-icesat.txt` 中列出的依赖（含 `torch`，但 torch 需与你的 CUDA/CPU 匹配）。
- **可选**：`mamba-ssm` 和 `causal-conv1d`（仅当你要 `--seq_backbone mamba` 时需要）。
- 如果只先跑通完整流程，使用 `--seq_backbone tcn` 即可，不强依赖 mamba。
