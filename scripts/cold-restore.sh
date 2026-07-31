#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s --confirm-destroy-existing BACKUP_DIRECTORY\n' "$0"
}

if [[ ${1:-} != "--confirm-destroy-existing" || -z ${2:-} ]]; then
  usage >&2
  exit 2
fi

command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 1; }
docker compose version >/dev/null

backup_root=$(cd "$2" && pwd)
project_name=${COMPOSE_PROJECT_NAME:-ifrs17-rag}
helper_image=${BACKUP_HELPER_IMAGE:-python:3.12.13-slim-bookworm}
volumes=(postgres-data redis-data qdrant-data object-data)
services=(caddy api worker beat postgres redis qdrant object-storage)

for volume in "${volumes[@]}"; do
  test -f "${backup_root}/${volume}.tar.gz" || {
    echo "Missing backup archive: ${volume}.tar.gz" >&2
    exit 1
  }
done

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$backup_root" && sha256sum --check SHA256SUMS)
else
  (cd "$backup_root" && shasum -a 256 -c SHA256SUMS)
fi

echo "Stopping services before destructive restore..."
docker compose stop "${services[@]}"

for volume in "${volumes[@]}"; do
  docker volume inspect "${project_name}_${volume}" >/dev/null
  echo "Restoring ${volume}..."
  docker run --rm \
    -v "${project_name}_${volume}:/target" \
    -v "${backup_root}:/backup:ro" \
    "$helper_image" \
    sh -euc "find /target -mindepth 1 -maxdepth 1 -exec rm -rf {} +; tar -C /target -xzf /backup/${volume}.tar.gz"
done

echo "Starting restored stack..."
docker compose up -d
echo "Restore complete. Verify /health/ready and run the smoke test before reopening traffic."
