"""Determinism seeding and structured console/file logging utilities."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Optional, TypeVar
import numpy as np
import torch

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None

T = TypeVar("T")


def progress_bar(iterable: Iterable[T], desc: str = "") -> Iterable[T]:
    """Wrap an iterable with `tqdm` if installed, otherwise yield items directly."""
    if _tqdm is not None:
        return _tqdm(iterable, desc=desc)
    return iterable


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible generation & training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(
    name: str = "refcompose",
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a rank-aware logger writing to stdout and an optional log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        logger.setLevel(logging.WARNING)

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file is not None and rank == 0:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
