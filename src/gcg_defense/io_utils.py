"""Logging, CSV output and small run-directory helpers."""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence


def setup_logging(run_dir: Optional[str] = None, level: int = logging.INFO) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(run_dir, "log.txt")))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


class CSVLog:
    """Append-only CSV with a fixed header, flushed after every row."""

    def __init__(self, path: str, fields: Sequence[str]) -> None:
        self.path = path
        self.fields = list(fields)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        self._fh = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fields)
        if not exists:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: Dict[str, Any]) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.fields})
        self._fh.flush()

    def write_all(self, rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            self.write(row)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "CSVLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


#: long-format metric rows, one file per run
METRIC_FIELDS = (
    "method",
    "method_params",
    "seed",
    "model",
    "split",
    "condition",
    "metric",
    "value",
)


def metrics_log(run_dir: str, name: str = "metrics.csv") -> CSVLog:
    return CSVLog(os.path.join(run_dir, name), METRIC_FIELDS)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


def read_json(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


class Timer:
    """Wall-clock timer with an exponential moving average of step times."""

    def __init__(self, smoothing: float = 0.1) -> None:
        self.start = time.time()
        self.last = self.start
        self.ema: Optional[float] = None
        self.smoothing = smoothing

    def tick(self) -> float:
        now = time.time()
        delta = now - self.last
        self.last = now
        self.ema = delta if self.ema is None else (
            self.smoothing * delta + (1 - self.smoothing) * self.ema
        )
        return delta

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    def eta(self, steps_left: int) -> float:
        rate = self.ema if self.ema is not None else 0.0
        return rate * steps_left
