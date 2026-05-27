from __future__ import annotations

import sys

from runtime.config import (
    ConfigError,
    ServingSettings,
    TrainingSettings,
    get_app_mode,
)
from runtime.serve_mode import run_server
from runtime.train_mode import run_training


def main() -> None:
    try:
        mode = get_app_mode()
        if mode == "train":
            run_training(TrainingSettings.from_env())
            return

        run_server(ServingSettings.from_env())
    except ConfigError as exc:
        print(f"CONFIG_ERROR {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
