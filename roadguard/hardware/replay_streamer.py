"""
RoadGuard Agent: Replay Event Streamer.
Ingests replay.ndjson from the NextEra kit to simulate live telemetry streaming.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional
import json
import time

from roadguard.config import DEFAULT_DATA_DIR


class ReplayStreamer:
    """
    Streams chronological inspection events from replay.ndjson.
    """

    def __init__(self, replay_file: Optional[Path] = None):
        self.replay_file = Path(replay_file or (DEFAULT_DATA_DIR / "replay.ndjson"))
        if not self.replay_file.exists():
            raise FileNotFoundError(f"Replay stream not found: {self.replay_file}")

    def get_event_generator(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Yields JSON events from replay.ndjson."""
        count = 0
        with open(self.replay_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
                    count += 1
                    if limit and count >= limit:
                        break

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Loads all events into a list."""
        return list(self.get_event_generator())
