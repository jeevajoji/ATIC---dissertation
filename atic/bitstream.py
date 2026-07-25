"""Fixed-size, versioned ATIC entropy-bitstream container.

The previous implementation compressed rounded tensors with NumPy.  Such an
``npz`` payload is not a learned codec bitstream.  This module packages the
actual byte strings emitted by CompressAI's entropy models.

Version 1 stores one image per file.  Its 128-byte header is followed by the
hyperlatent (``z``) rANS string and then the main-latent (``y``) rANS string.
The ordering mirrors decoding: ``z`` must be decoded before the probability
model and gain map needed to decode ``y`` can be reconstructed.
"""

from __future__ import annotations

import hashlib
import math
import operator
import os
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union


MAGIC = b"ATIC\r\n\x1a\n"
FORMAT_MAJOR = 1
FORMAT_MINOR = 0
HEADER_BYTES = 128

COLORSPACE_RGB = 1
ENTROPY_CODER_RANS = 1

# magic; version; header size; flags; eight spatial dimensions; y channels;
# bit depth; colourspace; coder; batch; quality id; z/y lengths; model SHA-256;
# z/y/header CRC-32; reserved.
_HEADER = struct.Struct(">8sHHII8IHBBBBHQQ32sIII8s")
_HEADER_CRC_OFFSET = 116
_KNOWN_FLAGS = 0x0F
_MAX_DIMENSION = 1 << 20
_MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_ATIC_FILE_BYTES = HEADER_BYTES + _MAX_PAYLOAD_BYTES

PathLike = Union[str, os.PathLike[str]]


class ATICBitstreamError(ValueError):
    """Raised when an ATIC file is corrupt, unsupported, or inconsistent."""


@dataclass(frozen=True)
class ATICBitstream:
    """Validated contents and metadata of one ATIC version-1 file."""

    flags: int
    original_width: int
    original_height: int
    coded_width: int
    coded_height: int
    y_width: int
    y_height: int
    z_width: int
    z_height: int
    y_channels: int
    bit_depth: int
    colorspace: int
    entropy_coder: int
    quality_id: int
    model_hash: bytes
    z_string: bytes
    y_string: bytes
    num_bytes: int

    @property
    def model_hash_hex(self) -> str:
        return self.model_hash.hex()

    @property
    def batch_size(self) -> int:
        return 1


def normalise_model_hash(value: Union[str, bytes, bytearray, memoryview]) -> bytes:
    """Return a 32-byte model identifier.

    A raw 32-byte digest or a 64-character hexadecimal SHA-256 is preserved.
    Other identifiers are deterministically SHA-256 hashed.  The codec CLI
    passes the exact checkpoint-file SHA-256 here.
    """

    if isinstance(value, str):
        if len(value) == 64:
            try:
                decoded = bytes.fromhex(value)
            except ValueError:
                decoded = b""
            if len(decoded) == 32:
                return decoded
        return hashlib.sha256(value.encode("utf-8")).digest()

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("model hash must be text or bytes-like")
    if memoryview(value).nbytes != 32:
        raise ValueError("A binary model hash must contain exactly 32 bytes")
    decoded = bytes(value)
    return decoded


def _integer(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not boolean")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a checkpoint's SHA-256 without loading it into memory."""

    chunk_size = _integer("chunk_size", chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_dimension(name: str, value: int) -> int:
    value = _integer(name, value)
    if value <= 0 or value > _MAX_DIMENSION:
        raise ValueError(f"{name} must be between 1 and {_MAX_DIMENSION}")
    return value


def _header_crc(header: bytes) -> int:
    if len(header) != HEADER_BYTES:
        raise ValueError("Internal ATIC header-size error")
    mutable = bytearray(header)
    struct.pack_into(">I", mutable, _HEADER_CRC_OFFSET, 0)
    return zlib.crc32(mutable) & 0xFFFFFFFF


def pack_atic(
    *,
    z_string: Union[bytes, bytearray, memoryview],
    y_string: Union[bytes, bytearray, memoryview],
    original_width: int,
    original_height: int,
    coded_width: int,
    coded_height: int,
    y_width: int,
    y_height: int,
    z_width: int,
    z_height: int,
    y_channels: int,
    model_hash: Union[str, bytes, bytearray, memoryview],
    flags: int = 0,
    quality_id: int = 0,
    bit_depth: int = 8,
    colorspace: int = COLORSPACE_RGB,
    entropy_coder: int = ENTROPY_CODER_RANS,
) -> bytes:
    """Create one validated ATIC version-1 byte container."""

    if not isinstance(z_string, (bytes, bytearray, memoryview)):
        raise TypeError("z_string must be bytes-like")
    if not isinstance(y_string, (bytes, bytearray, memoryview)):
        raise TypeError("y_string must be bytes-like")
    z_nbytes = memoryview(z_string).nbytes
    y_nbytes = memoryview(y_string).nbytes
    if z_nbytes + y_nbytes > _MAX_PAYLOAD_BYTES:
        raise ValueError("ATIC payload exceeds the version-1 safety limit")
    z_value = bytes(z_string)
    y_value = bytes(y_string)
    if not z_value or not y_value:
        raise ValueError("ATIC entropy strings cannot be empty")

    dimensions = (
        _validate_dimension("original_width", original_width),
        _validate_dimension("original_height", original_height),
        _validate_dimension("coded_width", coded_width),
        _validate_dimension("coded_height", coded_height),
        _validate_dimension("y_width", y_width),
        _validate_dimension("y_height", y_height),
        _validate_dimension("z_width", z_width),
        _validate_dimension("z_height", z_height),
    )
    if dimensions[0] > dimensions[2] or dimensions[1] > dimensions[3]:
        raise ValueError("Original dimensions cannot exceed coded dimensions")
    y_channels = _integer("y_channels", y_channels)
    if y_channels <= 0 or y_channels > 0xFFFF:
        raise ValueError("y_channels must fit in an unsigned 16-bit integer")

    flags = _integer("flags", flags)
    quality_id = _integer("quality_id", quality_id)
    bit_depth = _integer("bit_depth", bit_depth)
    colorspace = _integer("colorspace", colorspace)
    entropy_coder = _integer("entropy_coder", entropy_coder)
    if not 0 <= flags <= _KNOWN_FLAGS:
        raise ValueError("flags contain bits not defined by ATIC version 1")
    if not 0 <= quality_id <= 0xFFFF:
        raise ValueError("quality_id must fit in an unsigned 16-bit integer")
    if not 1 <= bit_depth <= 0xFF:
        raise ValueError("bit_depth must fit in an unsigned 8-bit integer")
    if not 1 <= colorspace <= 0xFF:
        raise ValueError("colorspace must fit in an unsigned 8-bit integer")
    if not 1 <= entropy_coder <= 0xFF:
        raise ValueError("entropy_coder must fit in an unsigned 8-bit integer")

    model_digest = normalise_model_hash(model_hash)
    z_crc = zlib.crc32(z_value) & 0xFFFFFFFF
    y_crc = zlib.crc32(y_value) & 0xFFFFFFFF
    header = _HEADER.pack(
        MAGIC,
        FORMAT_MAJOR,
        FORMAT_MINOR,
        HEADER_BYTES,
        flags,
        *dimensions,
        y_channels,
        bit_depth,
        colorspace,
        entropy_coder,
        1,  # version 1 stores exactly one image per file
        quality_id,
        len(z_value),
        len(y_value),
        model_digest,
        z_crc,
        y_crc,
        0,
        b"\0" * 8,
    )
    header_crc = _header_crc(header)
    header = bytearray(header)
    struct.pack_into(">I", header, _HEADER_CRC_OFFSET, header_crc)
    return bytes(header) + z_value + y_value


def unpack_atic(data: Union[bytes, bytearray, memoryview]) -> ATICBitstream:
    """Validate and unpack one ATIC version-1 byte container."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("ATIC data must be bytes-like")
    if memoryview(data).nbytes > MAX_ATIC_FILE_BYTES:
        raise ATICBitstreamError("ATIC file exceeds the version-1 safety limit")
    raw = bytes(data)
    if len(raw) < HEADER_BYTES:
        raise ATICBitstreamError("ATIC file is shorter than its fixed header")

    header = raw[:HEADER_BYTES]
    (
        magic,
        major,
        minor,
        header_bytes,
        flags,
        original_width,
        original_height,
        coded_width,
        coded_height,
        y_width,
        y_height,
        z_width,
        z_height,
        y_channels,
        bit_depth,
        colorspace,
        entropy_coder,
        batch_size,
        quality_id,
        z_length,
        y_length,
        model_hash,
        z_crc,
        y_crc,
        header_crc,
        reserved,
    ) = _HEADER.unpack(header)

    if magic != MAGIC:
        raise ATICBitstreamError("Invalid ATIC magic bytes")
    if major != FORMAT_MAJOR or minor != FORMAT_MINOR:
        raise ATICBitstreamError(
            f"Unsupported ATIC version {major}.{minor}; "
            f"expected {FORMAT_MAJOR}.{FORMAT_MINOR}"
        )
    if header_bytes != HEADER_BYTES:
        raise ATICBitstreamError(f"Unsupported ATIC header size: {header_bytes}")
    if batch_size != 1:
        raise ATICBitstreamError("ATIC version 1 supports exactly one image")
    if flags & ~_KNOWN_FLAGS:
        raise ATICBitstreamError("ATIC flags contain unsupported reserved bits")
    if reserved != b"\0" * 8:
        raise ATICBitstreamError("ATIC reserved header bytes must be zero")
    if _header_crc(header) != header_crc:
        raise ATICBitstreamError("ATIC header checksum validation failed")

    for name, value in (
        ("original_width", original_width),
        ("original_height", original_height),
        ("coded_width", coded_width),
        ("coded_height", coded_height),
        ("y_width", y_width),
        ("y_height", y_height),
        ("z_width", z_width),
        ("z_height", z_height),
    ):
        try:
            _validate_dimension(name, value)
        except ValueError as exc:
            raise ATICBitstreamError(str(exc)) from exc
    if y_channels == 0:
        raise ATICBitstreamError("ATIC y channel count cannot be zero")
    if original_width > coded_width or original_height > coded_height:
        raise ATICBitstreamError("ATIC original dimensions exceed coded dimensions")
    if bit_depth == 0 or colorspace == 0 or entropy_coder == 0:
        raise ATICBitstreamError("ATIC coding metadata contains a zero identifier")
    if z_length == 0 or y_length == 0:
        raise ATICBitstreamError("ATIC entropy payloads cannot be empty")
    if z_length + y_length > _MAX_PAYLOAD_BYTES:
        raise ATICBitstreamError("ATIC payload exceeds the safety limit")

    expected_size = HEADER_BYTES + z_length + y_length
    if len(raw) != expected_size:
        raise ATICBitstreamError(
            f"ATIC size mismatch: header declares {expected_size} bytes, "
            f"file contains {len(raw)}"
        )

    z_start = HEADER_BYTES
    y_start = z_start + z_length
    z_string = raw[z_start:y_start]
    y_string = raw[y_start:]
    if (zlib.crc32(z_string) & 0xFFFFFFFF) != z_crc:
        raise ATICBitstreamError("ATIC z payload checksum validation failed")
    if (zlib.crc32(y_string) & 0xFFFFFFFF) != y_crc:
        raise ATICBitstreamError("ATIC y payload checksum validation failed")

    return ATICBitstream(
        flags=flags,
        original_width=original_width,
        original_height=original_height,
        coded_width=coded_width,
        coded_height=coded_height,
        y_width=y_width,
        y_height=y_height,
        z_width=z_width,
        z_height=z_height,
        y_channels=y_channels,
        bit_depth=bit_depth,
        colorspace=colorspace,
        entropy_coder=entropy_coder,
        quality_id=quality_id,
        model_hash=model_hash,
        z_string=z_string,
        y_string=y_string,
        num_bytes=len(raw),
    )


def write_atic_bytes(path: PathLike, data: Union[bytes, bytearray, memoryview]) -> int:
    """Validate and atomically write a complete ATIC container."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("ATIC data must be bytes-like")
    if memoryview(data).nbytes > MAX_ATIC_FILE_BYTES:
        raise ATICBitstreamError("ATIC file exceeds the version-1 safety limit")
    encoded = bytes(data)
    unpack_atic(encoded)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return len(encoded)


def read_atic_file(path: PathLike) -> ATICBitstream:
    """Read, validate, and unpack an ATIC file with a bounded allocation."""

    target = Path(path)
    file_size = target.stat().st_size
    if file_size > MAX_ATIC_FILE_BYTES:
        raise ATICBitstreamError("ATIC file exceeds the version-1 safety limit")
    with open(target, "rb") as handle:
        encoded = handle.read(MAX_ATIC_FILE_BYTES + 1)
    if len(encoded) > MAX_ATIC_FILE_BYTES:
        raise ATICBitstreamError("ATIC file exceeds the version-1 safety limit")
    return unpack_atic(encoded)


def bpp_from_num_bytes(
    num_bytes: Union[int, float],
    image_or_height: Any,
    width: int | None = None,
    *,
    batch_size: int = 1,
) -> float:
    """Calculate actual file BPP, including the complete 128-byte header."""

    if width is None:
        shape = getattr(image_or_height, "shape", None)
        if shape is None or len(shape) != 4:
            raise TypeError(
                "Expected a tensor-like (N,C,H,W) object or explicit height/width"
            )
        batch_size, _, height, width = (
            _integer("shape value", value) for value in shape
        )
    else:
        height = _integer("height", image_or_height)
        width = _integer("width", width)
        batch_size = _integer("batch_size", batch_size)

    try:
        num_bytes = float(num_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError("num_bytes must be a finite number") from exc
    if not math.isfinite(num_bytes) or num_bytes < 0:
        raise ValueError("num_bytes must be finite and non-negative")
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError("batch size, height, and width must be positive")
    return float((num_bytes * 8.0) / (batch_size * height * width))
