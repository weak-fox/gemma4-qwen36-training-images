from __future__ import annotations

import sys
from pathlib import Path

from runtime.config import (
    ConfigError,
    DEFAULT_BAKED_MODEL_PATH,
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

        if not Path(DEFAULT_BAKED_MODEL_PATH).exists():
            raise ConfigError(
                "APP_MODE=serve requires the runtime-with-model image "
                f"with a baked model at {DEFAULT_BAKED_MODEL_PATH}"
            )
        run_server(ServingSettings.from_env())
    except ConfigError as exc:
        print(f"CONFIG_ERROR {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
