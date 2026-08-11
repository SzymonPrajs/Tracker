from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline import PacketReader, acquire, load_config, plan_acquisition, validate_packet
from data_pipeline.errors import PipelineError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data_pipeline", description="Build and validate bounded Tracker data packets"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="check that the library and adapters load")
    acquire_parser = commands.add_parser("acquire", help="acquire one source into one packet")
    acquire_parser.add_argument("--config", type=Path, required=True)
    acquire_parser.add_argument("--dry-run", action="store_true")
    validate_parser = commands.add_parser("validate", help="independently validate a packet")
    validate_parser.add_argument("packet", type=Path)
    inspect_parser = commands.add_parser("inspect", help="summarize a validated packet")
    inspect_parser.add_argument("packet", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "check":
            output: dict[str, object] = {
                "status": "passed",
                "library": "data_pipeline",
                "adapters": ["manifest", "open_images"],
                "raw_retention_option": False,
            }
        elif args.command == "acquire":
            config = load_config(args.config)
            output = plan_acquisition(config) if args.dry_run else acquire(config).as_dict()
        elif args.command == "validate":
            output = validate_packet(args.packet)
        else:
            reader = PacketReader(args.packet)
            records = list(reader.records())
            output = {
                "status": "passed",
                "packet": str(reader.packet_root),
                "images": len(records),
                "instances": sum(len(record.instances) for record in records),
                "semantics": sorted(
                    {instance.semantic.value for record in records for instance in record.instances}
                ),
                "source_splits": sorted({record.source_split for record in records}),
            }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except PipelineError as error:
        print(json.dumps({"status": "failed", "error": error.as_dict()}, indent=2, sort_keys=True))
        return 2
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": {
                        "code": "invalid_input",
                        "message": f"{type(error).__name__}: {error}",
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
