import csv
from pathlib import Path
from typing import Dict

from sparse_ops.net_utils import ensure_dir


class CsvLogger:
    def __init__(self, path: Path):
        self.path = path
        self.fieldnames = None
        if self.path.exists():
            self.path.unlink()

    def write_row(self, row: Dict[str, object]) -> None:
        ensure_dir(self.path.parent)
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        file_exists = self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
