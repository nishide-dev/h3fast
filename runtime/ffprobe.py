#!/usr/bin/env python3
"""Minimal ffprobe-compatible JSON adapter for the pinned SGLang H3 runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import av

if TYPE_CHECKING:
    from fractions import Fraction


def _seconds(duration: int | None, time_base: Fraction | None) -> str | None:
    if duration is None or time_base is None:
        return None
    return f"{float(duration * time_base):.6f}"


def _stream_fields(stream: av.stream.Stream) -> dict[str, object]:
    codec = stream.codec_context
    fields: dict[str, object] = {
        "codec_type": stream.type,
        "codec_name": codec.name,
        "duration": _seconds(stream.duration, stream.time_base),
    }
    if stream.type == "video":
        pixel_format = codec.format.name if codec.format is not None else None
        fields.update(
            {
                "pix_fmt": pixel_format,
                "width": codec.width,
                "height": codec.height,
                "avg_frame_rate": str(stream.average_rate or "0/0"),
                "nb_frames": str(stream.frames) if stream.frames else "N/A",
            }
        )
    elif stream.type == "audio":
        fields.update(
            {
                "sample_rate": str(codec.sample_rate or 0),
                "channels": codec.channels,
            }
        )
    return {key: value for key, value in fields.items() if value is not None}


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("ffprobe adapter requires an input path\n")
        return 2
    path = Path(sys.argv[-1])
    try:
        with av.open(path) as container:
            payload = {
                "streams": [_stream_fields(stream) for stream in container.streams],
                "format": {
                    "format_name": container.format.name,
                    "duration": (
                        f"{container.duration / av.time_base:.6f}"
                        if container.duration is not None
                        else None
                    ),
                },
            }
    except (OSError, ValueError, av.error.FFmpegError) as error:
        sys.stderr.write(f"could not probe {path}: {error}\n")
        return 1
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
