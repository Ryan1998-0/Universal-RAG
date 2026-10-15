"""Exercise backup/restore orchestration with a fake Docker command."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
if not BASH and Path("C:/Program Files/Git/bin/bash.exe").exists():
    BASH = "C:/Program Files/Git/bin/bash.exe"

VOLUMES = ("postgres-data", "redis-data", "qdrant-data", "object-data")
STUB = r'''
docker() {
  printf '<%s>' "$@" >> "$FAKE_DOCKER_LOG"
  printf '\n' >> "$FAKE_DOCKER_LOG"
  if [[ $1 == run && "$*" == *:/source:ro* ]]; then
    printf 'fixture archive' > "$backup_root/$volume.tar.gz"
  fi
}
export -f docker
exec bash "$@"
'''


@pytest.mark.skipif(not BASH, reason="Bash is required for shell orchestration tests")
@pytest.mark.parametrize("operation", ["cold-backup.sh", "cold-restore.sh"])
def test_backup_and_restore_keep_same_environment_and_project(tmp_path, operation):
    env_path = tmp_path / "deployment settings.env"
    env_path.write_text("# fake Docker does not read secrets\n", encoding="utf-8")
    backup_path = tmp_path / "backup"
    backup_path.mkdir()
    log_path = tmp_path / "docker-commands.txt"
    arguments = [backup_path.as_posix()]
    if operation == "cold-restore.sh":
        checksums = []
        for volume in VOLUMES:
            payload = b"fixture archive"
            filename = f"{volume}.tar.gz"
            (backup_path / filename).write_bytes(payload)
            checksums.append(f"{hashlib.sha256(payload).hexdigest()}  {filename}")
        (backup_path / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8", newline="\n")
        arguments.insert(0, "--confirm-destroy-existing")
    env = {
        **os.environ, "FAKE_DOCKER_LOG": log_path.as_posix(),
        "RAG_COMPOSE_ENV_FILE": env_path.as_posix(), "COMPOSE_PROJECT_NAME": "test-rag-backup",
    }
    result = subprocess.run(
        [BASH, "--noprofile", "--norc", "-c", STUB, "backup-test", (ROOT / "scripts" / operation).as_posix(), *arguments],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    commands = log_path.read_text(encoding="utf-8").splitlines()
    compose_commands = [line for line in commands if line.startswith("<compose>") and "<version>" not in line]
    assert any("<stop>" in line for line in compose_commands)
    assert any("<start>" in line for line in compose_commands)
    for line in compose_commands:
        assert f"<--env-file><{env_path.as_posix()}>" in line
        assert "<-p><test-rag-backup>" in line
        assert "<up>" not in line
    for volume in VOLUMES:
        assert any(f"test-rag-backup_{volume}" in line for line in commands)


@pytest.mark.skipif(not BASH, reason="Bash is required for shell orchestration tests")
def test_restore_requires_explicit_confirmation(tmp_path):
    result = subprocess.run(
        [BASH, (ROOT / "scripts" / "cold-restore.sh").as_posix(), tmp_path.as_posix()],
        capture_output=True, text=True, encoding="utf-8", timeout=10,
    )
    assert result.returncode == 2
    assert "--confirm-destroy-existing" in result.stderr
