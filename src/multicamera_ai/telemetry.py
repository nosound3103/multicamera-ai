"""Machine-readable timestamp strip for measuring RTSP decode latency."""

import zlib
from datetime import UTC, datetime

import numpy as np


def stamp_frame(frame, captured_at: datetime, sequence: int) -> None:
    # 48-bit Unix milliseconds + 16-bit sequence + 16-bit CRC, 8 pixels/cell.
    payload = int(captured_at.timestamp() * 1000).to_bytes(6, "big")
    payload += (sequence % 65536).to_bytes(2, "big")
    data = payload + (zlib.crc32(payload) & 65535).to_bytes(2, "big")
    for index, bit in enumerate(np.unpackbits(np.frombuffer(data, dtype=np.uint8))):
        frame[0:16, index * 8 : (index + 1) * 8] = 255 if bit else 0


def read_stamp(frame) -> tuple[datetime, int] | None:
    if frame.shape[1] < 640 or frame.shape[0] < 16:
        return None
    bits = [frame[4:12, i * 8 + 2 : i * 8 + 6].mean() > 127 for i in range(80)]
    data = np.packbits(bits).tobytes()
    if (zlib.crc32(data[:8]) & 65535) != int.from_bytes(data[8:], "big"):
        return None
    stamp = datetime.fromtimestamp(int.from_bytes(data[:6], "big") / 1000, UTC)
    return stamp, int.from_bytes(data[6:8], "big")
