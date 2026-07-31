import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


DEFAULT_RAW_DIR = "data/raw"
DEFAULT_INDEX_DIR = "data/index"
DEFAULT_ALIAS_PATH = "data/entities/default_aliases.json"
DEFAULT_GRAPH_PATH = "data/graph/graph.json"
DEFAULT_PROFILE = "default"
PROFILE_MANIFEST = "knowledge_base.json"


@dataclass(frozen=True)
class KnowledgeBaseProfile:
    name: str
    project_root: Path
    raw_dir: Path
    index_dir: Path
    alias_path: Path
    graph_path: Path
    profile_root: Optional[Path] = None
    manifest_path: Optional[Path] = None

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        project_root: Optional[Path] = None,
    ) -> "KnowledgeBaseProfile":
        values = os.environ if env is None else env
        root = project_root or Path(__file__).resolve().parents[1]
        profile_name = str(values.get("RAG_PROFILE") or "").strip() or DEFAULT_PROFILE
        profile_root = root / "profiles" / profile_name
        manifest_path = profile_root / PROFILE_MANIFEST if profile_root else None
        manifest = _load_manifest(manifest_path)

        raw_dir = _path_from_env_manifest_or_default(
            values,
            "RAG_KB_DIR",
            manifest,
            "raw_dir",
            profile_root,
            root,
            DEFAULT_RAW_DIR,
            "raw",
        )
        index_dir = _path_from_env_manifest_or_default(
            values,
            "RAG_INDEX_DIR",
            manifest,
            "index_dir",
            profile_root,
            root,
            DEFAULT_INDEX_DIR,
            "index",
        )
        alias_path = _path_from_env_manifest_or_default(
            values,
            "RAG_ENTITY_ALIASES",
            manifest,
            "alias_path",
            profile_root,
            root,
            DEFAULT_ALIAS_PATH,
            "entities/aliases.json",
        )
        graph_path = _path_from_env_manifest_or_default(
            values,
            "RAG_GRAPH_PATH",
            manifest,
            "graph_path",
            profile_root,
            root,
            DEFAULT_GRAPH_PATH,
            "graph/graph.json",
        )

        return cls(
            name=profile_name,
            project_root=root,
            raw_dir=raw_dir,
            index_dir=index_dir,
            alias_path=alias_path,
            graph_path=graph_path,
            profile_root=profile_root,
            manifest_path=manifest_path if manifest_path and manifest_path.exists() else None,
        )


def active_knowledge_base(
    env: Optional[Mapping[str, str]] = None,
    project_root: Optional[Path] = None,
) -> KnowledgeBaseProfile:
    return KnowledgeBaseProfile.from_env(env=env, project_root=project_root)


def _load_manifest(path: Optional[Path]) -> Mapping[str, object]:
    if not path or not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Knowledge base manifest must contain a JSON object: {path}")
    return loaded


def _path_from_env_manifest_or_default(
    values: Mapping[str, str],
    env_key: str,
    manifest: Mapping[str, object],
    manifest_key: str,
    profile_root: Optional[Path],
    project_root: Path,
    default_relative_path: str,
    profile_relative_path: str,
) -> Path:
    configured = values.get(env_key)
    if configured:
        return Path(configured).expanduser()

    manifest_value = manifest.get(manifest_key)
    if isinstance(manifest_value, str) and manifest_value.strip():
        return _resolve_profile_path(profile_root or project_root, manifest_value)

    if profile_root:
        return profile_root / profile_relative_path

    return project_root / default_relative_path


def _resolve_profile_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve(strict=False)
