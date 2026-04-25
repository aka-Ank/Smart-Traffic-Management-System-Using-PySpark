from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_analytics.config import PipelineConfig
from traffic_analytics.pipeline import optimize_route, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic analytics on PySpark datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the analytics pipeline.")
    _add_common_args(run_parser)

    route_parser = subparsers.add_parser("optimize-route", help="Find a lower-cost route.")
    _add_common_args(route_parser)
    route_parser.add_argument("--source", required=True, help="Source sensor id")
    route_parser.add_argument("--target", required=True, help="Target sensor id")

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset", default="METR-LA")
    parser.add_argument("--output-root", default=Path("output"), type=Path)


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        output_root=args.output_root,
    )

    if args.command == "run":
        result = run_pipeline(config)
    else:
        result = optimize_route(config, args.source, args.target)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
