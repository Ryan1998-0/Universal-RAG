from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalScope:
    """Trusted server-side boundary for every retrieval branch."""

    tenant_id: str
    knowledge_base_id: str
    index_version_id: str = ""
    artifact_attempt: int = 0

    def __post_init__(self):
        for field_name in ("tenant_id", "knowledge_base_id"):
            value = str(getattr(self, field_name) or "").strip()
            if not value or len(value) > 64:
                raise ValueError(f"{field_name} must contain 1 to 64 characters")
            object.__setattr__(self, field_name, value)
        clean_index_version = str(self.index_version_id or "").strip()
        if len(clean_index_version) > 64:
            raise ValueError("index_version_id must contain at most 64 characters")
        object.__setattr__(self, "index_version_id", clean_index_version)
        if type(self.artifact_attempt) is not int or self.artifact_attempt < 0:
            raise ValueError("artifact_attempt must be a non-negative integer")

    def matches(self, chunk: dict) -> bool:
        if not self.index_version_id:
            return False
        if str(chunk.get("tenant_id") or "") != self.tenant_id:
            return False
        if str(chunk.get("knowledge_base_id") or "") != self.knowledge_base_id:
            return False
        if str(chunk.get("index_version_id") or "") != self.index_version_id:
            return False
        if self.artifact_attempt == 0:
            return "artifact_attempt" not in chunk
        return (
            type(chunk.get("artifact_attempt")) is int
            and chunk["artifact_attempt"] == self.artifact_attempt
        )
