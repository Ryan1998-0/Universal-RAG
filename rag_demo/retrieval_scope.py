from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalScope:
    """Trusted server-side boundary for every retrieval branch."""

    tenant_id: str
    knowledge_base_id: str
    index_version_id: str = ""

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

    def matches(self, chunk: dict) -> bool:
        if not self.index_version_id:
            return False
        if str(chunk.get("tenant_id") or "") != self.tenant_id:
            return False
        if str(chunk.get("knowledge_base_id") or "") != self.knowledge_base_id:
            return False
        if str(chunk.get("index_version_id") or "") != self.index_version_id:
            return False
        return True
