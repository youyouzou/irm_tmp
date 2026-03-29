import argparse
import csv
from pathlib import Path

from sparse_ops.net_utils import ensure_dir, write_json
from wifi_dg.config import load_config
from wifi_dg.experiment import run_experiment, write_method_summary
from wifi_dg.metrics import summarize_numeric_metrics


DEFAULT_METHODS = [
    "erm_dense",
    "irm_dense",
    "rex_dense",
    "erm_sparse",
    "sparseirm",
    "sparserex",
]

SUMMARY_KEYS = [
    "val_worst_source_env_acc",
    "val_overall_acc",
    "test_macro_f1",
    "test_balanced_acc",
    "test_overall_acc",
    "global_sparsity_rate",
    "effective_params",
    "approx_effective_flops",
]


def _aggregate_runs(run_records):
    compact_records = [{key: record[key] for key in SUMMARY_KEYS} for record in run_records]
    summary = summarize_numeric_metrics(compact_records)
    flattened = {}
    for key, stats in summary.items():
        flattened[key] = f"{stats['mean']:.4f} +- {stats['std']:.4f}"
    return summary, flattened


def main():
    parser = argparse.ArgumentParser(description="Run the WiFi DG SparseIRM benchmark.")
    parser.add_argument("--config-dir", default="configs/wifi_dg_sparseirm")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    benchmark_rows = []
    benchmark_dataset_dir = None
    for method in args.methods:
        config_path = Path(args.config_dir) / f"{method}.yaml"
        method_config = load_config(config_path)
        if benchmark_dataset_dir is None:
            benchmark_dataset_dir = Path(method_config["output_root"]) / str(method_config["dataset_name"])
        method_records = []
        for seed in args.seeds:
            overrides = {"seed": seed}
            method_records.append(run_experiment(config_path, overrides=overrides))

        method_dir = Path(method_records[0]["output_dir"]).parent
        summary, flattened = _aggregate_runs(method_records)
        write_method_summary(method_dir, method_records)
        write_json({"runs": method_records, "aggregate": summary}, method_dir / "summary.json")
        aggregate_row = {"method_name": method_config["method_name"], **flattened}
        with (method_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate_row.keys()))
            writer.writeheader()
            writer.writerow(aggregate_row)

        benchmark_rows.append(aggregate_row)

    if benchmark_dataset_dir is None:
        benchmark_dataset_dir = Path("outputs")
    ensure_dir(benchmark_dataset_dir)
    benchmark_path = benchmark_dataset_dir / "benchmark_summary.csv"
    if benchmark_rows:
        with benchmark_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(benchmark_rows[0].keys()))
            writer.writeheader()
            writer.writerows(benchmark_rows)


if __name__ == "__main__":
    main()
