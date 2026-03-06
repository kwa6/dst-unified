import argparse
import os
import platform
from pathlib import Path

import yaml
from rich import print


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    print("[bold green]OK:[/bold green] loaded config")
    print(cfg)

    print("\n[bold]Environment[/bold]")
    print({"python": platform.python_version(), "platform": platform.platform()})
    print({"cwd": str(Path.cwd()), "venv": os.environ.get("VIRTUAL_ENV")})


if __name__ == "__main__":
    main()