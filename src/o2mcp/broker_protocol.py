"""Length-prefixed protocol shared by the local and remote O2 brokers.

The protocol deliberately does not use newline-delimited JSON. Remote command
output may contain arbitrary newlines, partial UTF-8 sequences, or text that
looks like JSON; a fixed-width length header keeps every request and response
unambiguous on one long-lived byte stream.

Only the Python standard library is used because the helper source is embedded
in the single SSH session and executed on an O2 login node. Keeping this module
small also makes the exact remote program easy to audit and test locally.
"""

from __future__ import annotations

import json
import struct
import textwrap
from collections.abc import Mapping
from typing import Any, BinaryIO

# Version 2 adds the local dispatch acknowledgement. A client must never send a
# command to a version-1 daemon because that older server could execute it while
# the newer client mistakes the direct result for a pre-dispatch failure.
PROTOCOL_VERSION = 2
FRAME_MAGIC = b"O2B1"
HEADER_SIZE = len(FRAME_MAGIC) + 4
MAX_FRAME_BYTES = 16 * 1024 * 1024
# JSON escaping can expand arbitrary command bytes substantially (for example,
# every NUL becomes six ASCII characters). A one-MiB per-stream source bound
# therefore keeps even worst-case escaped stdout+stderr below the 16-MiB frame.
MAX_OUTPUT_BYTES = 1 * 1024 * 1024
MAX_STARTUP_PREAMBLE_BYTES = 64 * 1024


class BrokerProtocolError(RuntimeError):
    """Raised when a peer sends a malformed, oversized, or truncated frame."""


def encode_frame(payload: Mapping[str, Any], *, max_bytes: int = MAX_FRAME_BYTES) -> bytes:
    """Serialize one JSON object with a magic marker and network-order length.

    Args:
        payload: JSON-compatible mapping to serialize.
        max_bytes: Maximum permitted UTF-8 payload size. Tests may lower this
            bound to exercise rejection without allocating a large object.
    """

    try:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BrokerProtocolError(f"broker payload is not JSON serializable: {exc}") from exc
    if not body:
        raise BrokerProtocolError("broker frames may not be empty")
    if len(body) > max_bytes:
        raise BrokerProtocolError(f"broker frame is {len(body)} bytes; maximum is {max_bytes}")
    return FRAME_MAGIC + struct.pack("!I", len(body)) + body


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    """Read exactly ``size`` bytes or raise a protocol error on premature EOF."""

    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            received = size - remaining
            raise BrokerProtocolError(f"broker stream ended after {received} of {size} expected bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_magic(stream: BinaryIO, *, resynchronize: bool) -> None:
    """Read the frame marker, optionally skipping a bounded startup banner."""

    if not resynchronize:
        marker = _read_exact(stream, len(FRAME_MAGIC))
        if marker != FRAME_MAGIC:
            raise BrokerProtocolError(f"broker frame has invalid magic marker: {marker!r}")
        return

    window = bytearray()
    for _ in range(MAX_STARTUP_PREAMBLE_BYTES + len(FRAME_MAGIC)):
        window.extend(_read_exact(stream, 1))
        if len(window) > len(FRAME_MAGIC):
            del window[0]
        if bytes(window) == FRAME_MAGIC:
            return
    raise BrokerProtocolError(f"broker protocol marker not found within {MAX_STARTUP_PREAMBLE_BYTES} startup bytes")


def read_frame(
    stream: BinaryIO,
    *,
    max_bytes: int = MAX_FRAME_BYTES,
    resynchronize: bool = False,
) -> dict[str, Any]:
    """Read one framed JSON object, optionally after bounded startup output.

    ``resynchronize`` is used only for the first remote hello because an SSH
    login shell can emit a site or user banner before executing Python. Every
    later frame is strict: command output is captured by the helper and any
    unframed byte indicates channel corruption.
    """

    _read_magic(stream, resynchronize=resynchronize)
    header = _read_exact(stream, 4)
    (length,) = struct.unpack("!I", header)
    if length == 0:
        raise BrokerProtocolError("broker frames may not be empty")
    if length > max_bytes:
        raise BrokerProtocolError(f"broker frame declares {length} bytes; maximum is {max_bytes}")
    raw = _read_exact(stream, length)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrokerProtocolError(f"broker frame is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BrokerProtocolError("broker frame payload must be a JSON object")
    return payload


def write_frame(stream: BinaryIO, payload: Mapping[str, Any], *, max_bytes: int = MAX_FRAME_BYTES) -> None:
    """Write and flush one validated frame to a binary stream."""

    stream.write(encode_frame(payload, max_bytes=max_bytes))
    stream.flush()


def remote_helper_source() -> str:
    """Return the auditable remote program used by the persistent SSH session.

    The helper captures command stdout/stderr so those bytes never mix with the
    framing stream. Commands are intentionally serialized: this MVP optimizes
    the authentication boundary, not concurrent scheduling on a login node.
    A bounded response prevents one noisy command from exhausting the local MCP
    process or corrupting the next frame.
    """

    # Keep constants literal in the generated program. Importing ``o2mcp`` on
    # the remote host would require a deployment step and could silently select
    # a different package version than the workstation broker.
    return textwrap.dedent(
        f"""\
        import json
        import math
        import os
        import signal
        import struct
        import subprocess
        import sys
        import threading
        import time
        from contextlib import suppress

        PROTOCOL_VERSION = {PROTOCOL_VERSION}
        FRAME_MAGIC = b"O2B1"
        MAX_FRAME_BYTES = {MAX_FRAME_BYTES}
        MAX_OUTPUT_BYTES = {MAX_OUTPUT_BYTES}

        def read_exact(size):
            chunks = []
            remaining = size
            while remaining:
                chunk = sys.stdin.buffer.read(remaining)
                if not chunk:
                    raise EOFError
                chunks.append(chunk)
                remaining -= len(chunk)
            return b\"\".join(chunks)

        def read_frame():
            marker = read_exact(4)
            if marker != FRAME_MAGIC:
                raise ValueError(\"invalid frame marker\")
            length = struct.unpack(\"!I\", read_exact(4))[0]
            if length <= 0 or length > MAX_FRAME_BYTES:
                raise ValueError(\"invalid frame length\")
            payload = json.loads(read_exact(length).decode(\"utf-8\"))
            if not isinstance(payload, dict):
                raise ValueError(\"frame payload must be an object\")
            return payload

        def write_frame(payload):
            body = json.dumps(payload, ensure_ascii=False, separators=(\",\", \":\")).encode(\"utf-8\")
            if not body or len(body) > MAX_FRAME_BYTES:
                raise ValueError(\"response frame exceeds protocol limit\")
            sys.stdout.buffer.write(FRAME_MAGIC + struct.pack(\"!I\", len(body)) + body)
            sys.stdout.buffer.flush()

        def drain_bounded(stream, captures, key):
            parts = []
            retained = 0
            truncated = False
            try:
                while True:
                    chunk = stream.read(65536)
                    if not chunk:
                        break
                    room = max(0, MAX_OUTPUT_BYTES - retained)
                    if room:
                        kept = chunk[:room]
                        parts.append(kept)
                        retained += len(kept)
                    if len(chunk) > room:
                        truncated = True
            except OSError:
                pass
            finally:
                with suppress(OSError):
                    stream.close()
                captures[key] = (b\"\".join(parts), truncated)

        def feed_stdin(stream, value):
            try:
                if value is not None:
                    stream.write(value.encode(\"utf-8\"))
                    stream.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                with suppress(OSError):
                    stream.close()

        def kill_process_group(process):
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            if process.poll() is None:
                process.kill()

        def run_bounded(command, stdin_text, timeout):
            process = subprocess.Popen(
                [\"/bin/bash\", \"-lc\", command],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            captures = {{}}
            stdout_thread = threading.Thread(
                target=drain_bounded, args=(process.stdout, captures, \"stdout\"), daemon=True
            )
            stderr_thread = threading.Thread(
                target=drain_bounded, args=(process.stderr, captures, \"stderr\"), daemon=True
            )
            stdin_thread = threading.Thread(target=feed_stdin, args=(process.stdin, stdin_text), daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            stdin_thread.start()
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_process_group(process)
                process.wait()

            # Correctly detached children redirect these descriptors and let the
            # drains end immediately. A child that accidentally inherits them
            # must not keep the broker hung or make output accumulation unbounded.
            for thread in (stdout_thread, stderr_thread):
                thread.join(timeout=0.5)
            if stdout_thread.is_alive() or stderr_thread.is_alive():
                kill_process_group(process)
                for stream in (process.stdout, process.stderr):
                    with suppress(OSError):
                        os.close(stream.fileno())
                stdout_thread.join(timeout=0.5)
                stderr_thread.join(timeout=0.5)
            stdin_thread.join(timeout=0.5)

            stdout_raw, stdout_truncated = captures.get(\"stdout\", (b\"\", False))
            stderr_raw, stderr_truncated = captures.get(\"stderr\", (b\"\", False))
            stdout = stdout_raw.decode(\"utf-8\", errors=\"replace\")
            stderr = stderr_raw.decode(\"utf-8\", errors=\"replace\")
            if timed_out:
                stderr += (\"\\n\" if stderr else \"\") + \"command timed out inside persistent O2 broker\"
            return (
                124 if timed_out else process.returncode,
                stdout,
                stderr,
                timed_out,
                stdout_truncated,
                stderr_truncated,
            )

        write_frame({{"type": "hello", "protocol": PROTOCOL_VERSION}})
        while True:
            try:
                request = read_frame()
            except EOFError:
                break
            request_id = request.get(\"id\")
            if request.get(\"type\") != \"exec\" or not isinstance(request_id, str):
                write_frame({{"type": "error", "id": request_id, "error": "invalid_request"}})
                continue
            command = request.get(\"command\")
            timeout = request.get(\"timeout_seconds\")
            stdin_text = request.get(\"stdin\")
            valid_timeout = (isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
                             and math.isfinite(timeout) and timeout > 0)
            if not isinstance(command, str) or not command or not valid_timeout:
                write_frame({{"type": "error", "id": request_id, "error": "invalid_request"}})
                continue
            if stdin_text is not None and not isinstance(stdin_text, str):
                write_frame({{"type": "error", "id": request_id, "error": "invalid_request"}})
                continue
            started = time.monotonic()
            returncode, stdout, stderr, timed_out, stdout_truncated, stderr_truncated = run_bounded(
                command, stdin_text, float(timeout)
            )
            response = {{
                "type": "result",
                "id": request_id,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": timed_out,
                "duration_seconds": time.monotonic() - started,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }}
            write_frame(response)
        """
    )
