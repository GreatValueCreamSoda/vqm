#!/usr/bin/env python3
"""Remove JPEG ICC APP2 segments while preserving compressed scan bytes."""

import argparse
from pathlib import Path


def _segment_end(data: bytes, offset: int) -> int:
    """Validate a length-prefixed JPEG marker and return its end offset.

    :param data: Complete JPEG file bytes.
    :param offset: Marker offset before the first scan.
    :returns: Offset immediately after the marker segment.
    :raises ValueError: If the marker or its length is invalid.
    """
    if offset + 4 > len(data) or data[offset] != 0xFF:
        raise ValueError("Expected a JPEG header marker")
    length = int.from_bytes(data[offset + 2:offset + 4], "big")
    end = offset + 2 + length
    if length < 2 or end > len(data):
        raise ValueError("Invalid JPEG marker length")
    return end


def strip_icc(data: bytes) -> tuple[bytes, int]:
    """Remove ICC APP2 segments before the first JPEG scan.

    The CID22 probe uses ordinary JPEG marker headers. All bytes from the
    first start-of-scan marker onward are copied verbatim.

    :param data: Complete JPEG file bytes.
    :returns: Modified JPEG bytes and the number of removed bytes.
    :raises ValueError: If the JPEG header is unsupported or truncated.
    """
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("Expected JPEG start-of-image marker")
    result = bytearray(data[:2])
    offset = 2
    removed = 0
    while offset < len(data):
        if data[offset:offset + 2] == b"\xff\xda":
            result.extend(data[offset:])
            return bytes(result), removed
        end = _segment_end(data, offset)
        segment = data[offset:end]
        if segment.startswith(b"\xff\xe2"
                              ) and segment[4:16] == b"ICC_PROFILE\x00":
            removed += len(segment)
        else:
            result.extend(segment)
        offset = end
    raise ValueError("JPEG has no start-of-scan marker")


def main() -> None:
    """Write one ICC-free JPEG copy and report the removed byte count.

    :raises ValueError: If the source and destination paths are identical.
    :raises FileExistsError: If the destination already exists.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        raise ValueError("Source and destination must differ")
    data, removed = strip_icc(args.source.read_bytes())
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("xb") as output:
        output.write(data)
    print(f"removed_bytes={removed}")


if __name__ == "__main__":
    main()
