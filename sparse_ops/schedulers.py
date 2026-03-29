import math


def get_lr_scheduler(optimizer, config):
    scheduler_name = config.get("lr_scheduler", "cosine")
    epochs = int(config["epochs"])
    warmup_epochs = int(config.get("warmup_epochs", 0))

    if scheduler_name == "cosine":
        def _schedule(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            cosine_total = max(1, epochs - warmup_epochs)
            cosine_epoch = max(0, epoch - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * cosine_epoch / cosine_total))

        return _schedule
    if scheduler_name == "constant":
        return lambda epoch: 1.0
    raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")
