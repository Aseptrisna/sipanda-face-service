"""Centralized logging: console + file handler, timestamped, idempotent setup."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

LOG_DIR = "logs"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str, log_dir: str = LOG_DIR, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{datetime.now(timezone.utc):%Y%m%d}.log"
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
