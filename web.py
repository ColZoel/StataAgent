"""Launch the StataAgent web UI.

Usage:
    python web.py                     # config.yaml, http://127.0.0.1:8765
    python web.py --port 8000
    python web.py --config my.yaml --no-browser
"""
import argparse

from hf_agent import AgentConfig
from webui.server import launch


def main() -> None:
    parser = argparse.ArgumentParser(description="StataAgent web UI")
    parser.add_argument("--config", default="config.yaml", metavar="PATH",
                        help="YAML config file to load (default: config.yaml)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open a browser tab automatically")
    args = parser.parse_args()

    cfg = AgentConfig.from_yaml(args.config)
    launch(cfg, host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
