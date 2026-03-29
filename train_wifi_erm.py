import argparse

from wifi_dg.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Train a WiFi DG experiment in ERM mode.")
    parser.add_argument(
        "--config",
        default="configs/wifi_dg_sparseirm/erm_dense.yaml",
        help="Path to a config file.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone-type", dest="backbone_type", choices=["dense", "sparse"], default=None)
    args = parser.parse_args()

    overrides = {
        "seed": args.seed,
        "penalty_type": "none",
        "backbone_type": args.backbone_type,
    }
    run_experiment(args.config, overrides=overrides)


if __name__ == "__main__":
    main()
