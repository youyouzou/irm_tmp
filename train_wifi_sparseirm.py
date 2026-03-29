import argparse

from wifi_dg.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Train a WiFi DG experiment in SparseIRM mode.")
    parser.add_argument(
        "--config",
        default="configs/wifi_dg_sparseirm/sparseirm.yaml",
        help="Path to a config file.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--penalty-type", dest="penalty_type", choices=["irm", "rex"], default=None)
    parser.add_argument("--backbone-type", dest="backbone_type", choices=["dense", "sparse"], default=None)
    args = parser.parse_args()

    overrides = {
        "seed": args.seed,
        "penalty_type": args.penalty_type,
        "backbone_type": args.backbone_type,
    }
    run_experiment(args.config, overrides=overrides)


if __name__ == "__main__":
    main()
