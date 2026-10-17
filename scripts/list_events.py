"""Inspect SQLite events; times must include a timezone offset or Z."""

import argparse
import json
import sys
from datetime import datetime

from multicamera_ai.events import SQLiteEventStore


def parse_timestamp(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None or value.utcoffset() is None:
        raise argparse.ArgumentTypeError("Timestamp must include timezone or Z")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="storage/events.db")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--start", type=parse_timestamp)
    parser.add_argument("--end", type=parse_timestamp)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    rows = SQLiteEventStore(args.db).list_events(
        camera_id=args.camera, start=args.start, end=args.end, limit=args.limit
    )
    sys.stdout.write(
        json.dumps([row.model_dump(mode="json") for row in rows], ensure_ascii=False, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
