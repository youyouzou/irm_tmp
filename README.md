# WiFi DG SparseIRM

用于 WiFi CSI 领域泛化（DG）的训练框架，支持以下方法：

- ERM
- IRM
- REx
- SparseIRM
- SparseREx

当前数据集：

- `wifi_data/5300`
- `wifi_data/mmfi`

## 数据与 Baseline 策略

- 5300：使用默认 `wifi_csi_default` backbone。
- MMFI：`ERM/IRM/REx` 使用 `mmfi_resnet` 专用 backbone（与 5300 分离）。
- MMFI 输入重排：`(297, 3, 114, 10) -> (30, 297, 114)`。
- MMFI 非有限值处理：`linear_interp`（时间维线性插值 + 边界延拓 + 全无效序列置零）。

## 选模与 Early Stopping

`new_best` 与 best checkpoint 的判定规则：

1. `val_acc`（越高越好）
2. `worst_source_env_acc`（越高越好）
3. `val_loss`（越低越好）

Early stopping 默认启用：

- `early_stopping_enabled: true`
- `early_stopping_patience: 15`
- `early_stopping_min_epochs: 10`

## 训练命令

MMFI ERM：

```bash
python train_wifi_erm.py --config configs/wifi_dg_sparseirm/mmfi_erm_dense.yaml --seed 0
```

`mmfi_erm_dense.yaml` 已内置稳健参数（`epochs=200, batch_size=12, lr=3e-4, weight_decay=1e-4, dropout=0.1, early_stopping_min_epochs=60, early_stopping_patience=40`）。

MMFI IRM / REx：

```bash
python train_wifi_sparseirm.py --config configs/wifi_dg_sparseirm/mmfi_irm_dense.yaml --seed 0
python train_wifi_sparseirm.py --config configs/wifi_dg_sparseirm/mmfi_rex_dense.yaml --seed 0
```

5300 ERM：

```bash
python train_wifi_erm.py --config configs/wifi_dg_sparseirm/erm_dense.yaml --seed 0
```

## 输出目录（固定分层）

每次训练统一输出到：

```text
outputs/{dataset_name}/{method_name}/seed{n}/
```

目录结构：

```text
seed{n}/
  config.yaml
  checkpoints/
    checkpoint_best.pth
  logs/
    train_log.csv
  plots/
    curves_acc.png
    curves_loss.png
    confusion_matrix_test_normalized.png
  metrics/
    run_metrics.json
```

其中 `metrics/run_metrics.json` 为唯一统计汇总文件，包含：

- `dataset_info`
- `best_val_record`
- `val_best_metrics`
- `test_metrics`
- `final_summary`
- `sparsity`（仅稀疏方法存在）

## Benchmark

按方法批量运行并汇总：

```bash
python run_wifi_dg_sparseirm_benchmark.py --config-dir configs/wifi_dg_sparseirm --methods erm_dense irm_dense rex_dense --seeds 0 1 2
```

方法级与数据集级汇总分别输出在：

- `outputs/{dataset_name}/{method_name}/summary.json`
- `outputs/{dataset_name}/{method_name}/summary.csv`
- `outputs/{dataset_name}/benchmark_summary.csv`
