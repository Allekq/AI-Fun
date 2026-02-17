import os
from pathlib import Path


def get_project_root() -> Path:
    env_root = os.environ.get("AI_FUN_ROOT")
    if env_root:
        return Path(env_root)
    return Path.cwd()
