#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from cowarch.label_studio import (
    build_label_studio_tasks,
    import_label_studio_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bridge the CowArch manifest and Label Studio JSON."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export-tasks", help="create Label Studio task JSON")
    export.add_argument("--manifest", default="data/manifest.csv")
    export.add_argument("--output", required=True)
    export.add_argument("--phase", choices=("posture", "geometry"), required=True)
    export.add_argument("--data-root", default="data")
    export.add_argument("--split", default=None)
    export.add_argument("--include-completed", action="store_true")
    export.add_argument(
        "--approved-sources",
        default=None,
        help="optional CSV whose rows must all have license_status=approved",
    )

    merge = commands.add_parser("import-results", help="merge a Label Studio export")
    merge.add_argument("--manifest", default="data/manifest.csv")
    merge.add_argument("--export", required=True)
    merge.add_argument("--output-manifest", required=True)
    merge.add_argument("--phase", choices=("posture", "geometry"), required=True)
    merge.add_argument("--reviewer", default=None)
    merge.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "export-tasks":
        summary = build_label_studio_tasks(
            args.manifest,
            args.output,
            phase=args.phase,
            data_root=args.data_root,
            split=args.split,
            include_completed=args.include_completed,
            approved_sources_path=args.approved_sources,
        )
    else:
        summary = import_label_studio_results(
            args.manifest,
            args.export,
            args.output_manifest,
            phase=args.phase,
            reviewer=args.reviewer,
            allow_overwrite=args.allow_overwrite,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
