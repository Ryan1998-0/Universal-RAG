from __future__ import annotations

import json
import re
import unicodedata
import socket
import struct
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from rag_demo.document_pipeline import SUPPORTED_FORMATS, decode_text_payload


class FileSecurityError(ValueError):
    pass


class MalwareDetectedError(FileSecurityError):
    pass


@dataclass(frozen=True)
class FileInspection:
    extension: str
    detected_mime_type: str
    prompt_injection_findings: tuple[str, ...]


class ClamAvScanner:
    def __init__(self, host: str, port: int = 3310, timeout_seconds: int = 30):
        self.host = str(host or "").strip()
        self.port = int(port)
        self.timeout_seconds = int(timeout_seconds)

    def ping(self) -> None:
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout_seconds
        ) as connection:
            connection.sendall(b"zPING\0")
            response = _receive_clamd_response(connection)
        if "PONG" not in response:
            raise RuntimeError("ClamAV did not return PONG")

    def scan(self, payload: bytes) -> None:
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout_seconds
        ) as connection:
            connection.sendall(b"zINSTREAM\0")
            view = memoryview(payload)
            for offset in range(0, len(view), 64 * 1024):
                block = view[offset : offset + 64 * 1024]
                connection.sendall(struct.pack("!I", len(block)))
                connection.sendall(block)
            connection.sendall(struct.pack("!I", 0))
            response = _receive_clamd_response(connection)
        if response.endswith(" OK") or response == "stream: OK":
            return
        if " FOUND" in response:
            raise MalwareDetectedError("malware was detected in the uploaded document")
        raise RuntimeError("ClamAV could not complete the scan")


class NoOpMalwareScanner:
    def ping(self) -> None:
        return None

    def scan(self, payload: bytes) -> None:
        return None


def inspect_file(
    *,
    filename: str,
    payload: bytes,
    declared_mime_type: str,
    max_uncompressed_bytes: int = 250 * 1024 * 1024,
) -> FileInspection:
    clean_name = Path(str(filename or "").replace("\\", "/")).name
    extension = Path(clean_name).suffix.lower()
    if extension not in SUPPORTED_FORMATS:
        raise FileSecurityError("file extension is not supported")
    expected_mime = SUPPORTED_FORMATS[extension]["mime_type"]
    if str(declared_mime_type or "").lower() != expected_mime:
        raise FileSecurityError("declared MIME type does not match the file extension")

    if extension == ".docx":
        _inspect_docx(payload, max_uncompressed_bytes=max_uncompressed_bytes)
    elif extension == ".pdf" and not payload.startswith(b"%PDF-"):
        raise FileSecurityError("PDF signature is invalid")
    elif extension == ".png" and not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise FileSecurityError("PNG signature is invalid")
    elif extension in {".jpg", ".jpeg"} and not (
        payload.startswith(b"\xff\xd8\xff") and payload.endswith(b"\xff\xd9")
    ):
        raise FileSecurityError("JPEG signature is invalid")
    elif extension == ".webp" and not (
        payload.startswith(b"RIFF") and payload[8:12] == b"WEBP"
    ):
        raise FileSecurityError("WebP signature is invalid")
    elif extension == ".bmp" and not payload.startswith(b"BM"):
        raise FileSecurityError("BMP signature is invalid")
    elif extension in {".tif", ".tiff"} and not payload.startswith(
        (b"II*\x00", b"MM\x00*")
    ):
        raise FileSecurityError("TIFF signature is invalid")
    elif extension == ".heic" and b"ftyp" not in payload[:32]:
        raise FileSecurityError("HEIC signature is invalid")

    text_for_scan = ""
    if extension in {".txt", ".md", ".markdown", ".json"}:
        text_for_scan = decode_text_payload(payload)
        if extension == ".json":
            try:
                json.loads(text_for_scan)
            except json.JSONDecodeError as exc:
                raise FileSecurityError("JSON payload is invalid") from exc
    return FileInspection(
        extension=extension,
        detected_mime_type=expected_mime,
        prompt_injection_findings=tuple(scan_prompt_injection(text_for_scan)),
    )


def scan_prompt_injection(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", normalized)
    patterns = (
        ("ignore_previous_instructions", r"ignore\s+(all\s+)?previous\s+instructions"),
        ("system_prompt_exfiltration", r"(show|reveal|print|output).{0,40}system\s+prompt"),
        ("tool_execution_instruction", r"(execute|run|call).{0,30}(tool|command|shell)"),
        ("chinese_instruction_override", r"忽略.{0,12}(先前|以上|原本).{0,12}(指令|規則)"),
        ("role_spoofing", r"(?:\[im_start\]\s*(?:system|developer)|<\|im_start\|>\s*(?:system|developer)|^\s*#{1,4}\s*(?:system|developer)\s*[:：])"),
        ("evidence_boundary_escape", r"</\s*(?:evidence|trusted_evidence)\s*>"),
        ("chinese_answer_override", r"(?:回答時|作答時|接下來的回答).{0,30}(?:一律|只輸出|不要引用|忽略來源)"),
        ("english_answer_override", r"(?:assistant|model|chatbot).{0,30}(?:must|should|shall).{0,30}(?:answer|respond|output)"),
        ("answer_exfiltration", r"(?:send|post|upload|傳送|回傳|寄到).{0,45}(?:https?://|[\w.+-]+@[\w.-]+)"),
    )
    findings = []
    for label, pattern in patterns:
        if re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE):
            findings.append(label)
    return findings


def _inspect_docx(payload: bytes, max_uncompressed_bytes: int) -> None:
    if not zipfile.is_zipfile(BytesIO(payload)):
        raise FileSecurityError("DOCX container is invalid")
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        members = archive.infolist()
        if len(members) > 5000:
            raise FileSecurityError("DOCX contains too many archive members")
        names = set()
        total_uncompressed = 0
        total_compressed = 0
        for member in members:
            normalized = member.filename.replace("\\", "/")
            if normalized.startswith("/") or any(
                part in {"", ".", ".."} for part in normalized.split("/")
            ):
                raise FileSecurityError("DOCX contains an unsafe archive path")
            names.add(normalized)
            total_uncompressed += int(member.file_size)
            total_compressed += max(1, int(member.compress_size))
        if total_uncompressed > max_uncompressed_bytes:
            raise FileSecurityError("DOCX uncompressed size exceeds the safety limit")
        if total_uncompressed / max(1, total_compressed) > 100:
            raise FileSecurityError("DOCX compression ratio exceeds the safety limit")
        if "[Content_Types].xml" not in names or "word/document.xml" not in names:
            raise FileSecurityError("DOCX required parts are missing")


def _receive_clamd_response(connection) -> str:
    parts = []
    while True:
        block = connection.recv(4096)
        if not block:
            break
        parts.append(block)
        if b"\0" in block:
            break
    return b"".join(parts).rstrip(b"\0\n").decode("utf-8", errors="replace")
