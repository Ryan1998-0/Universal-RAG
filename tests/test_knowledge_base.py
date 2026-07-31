from pathlib import Path

from rag_demo.config import resolve_profile_path
from rag_demo.knowledge_base import KnowledgeBaseProfile


def test_default_profile_uses_generic_profile_directory(tmp_path: Path) -> None:
    profile_root = tmp_path / "profiles" / "default"
    profile_root.mkdir(parents=True)
    (profile_root / "knowledge_base.json").write_text(
        '{"raw_dir":"raw","index_dir":"index",'
        '"alias_path":"entities/aliases.json","graph_path":"graph/graph.json"}',
        encoding="utf-8",
    )

    profile = KnowledgeBaseProfile.from_env(env={}, project_root=tmp_path)

    assert profile.name == "default"
    assert profile.profile_root == profile_root
    assert profile.raw_dir == profile_root / "raw"
    assert profile.index_dir == profile_root / "index"


def test_blank_profile_name_also_uses_generic_default(tmp_path: Path) -> None:
    profile = KnowledgeBaseProfile.from_env(
        env={"RAG_PROFILE": "   "},
        project_root=tmp_path,
    )

    assert profile.name == "default"
    assert profile.profile_root == tmp_path / "profiles" / "default"

    (profile.profile_root / "index").mkdir(parents=True)
    resolved = resolve_profile_path(
        "RAG_INDEX_DIR",
        "data/index",
        "index",
        env={"RAG_PROFILE": "   "},
        project_root=tmp_path,
    )
    assert resolved == profile.profile_root / "index"


def test_active_runtime_is_not_bound_to_the_ifrs17_sample() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runtime_roots = (
        project_root / "rag_demo",
        project_root / "frontend",
        project_root / "docs" / "rag-demo",
    )
    inspected_suffixes = {".py", ".js", ".html", ".css"}

    offenders = []
    for runtime_root in runtime_roots:
        for path in runtime_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in inspected_suffixes:
                if "ifrs17" in path.read_text(encoding="utf-8").lower():
                    offenders.append(path.relative_to(project_root).as_posix())

    assert offenders == []
