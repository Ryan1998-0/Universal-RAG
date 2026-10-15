#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s [backup-directory]\n' "$0"
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 1; }
docker compose version >/dev/null

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
backup_root=${1:-"./backups/${timestamp}"}
mkdir -p "$backup_root"
backup_root=$(cd "$backup_root" && pwd)
chmod 700 "$backup_root"

project_name=${COMPOSE_PROJECT_NAME:-universal-rag}
repo_root=$(cd "$(dirname "$0")/.." && pwd)
env_file=${RAG_COMPOSE_ENV_FILE:-"${repo_root}/.env.production"}
test -f "$env_file" || { echo "Compose environment file is missing" >&2; exit 1; }
compose=(docker compose --project-directory "$repo_root" --env-file "$env_file" -f "${repo_root}/compose.yaml" -p "$project_name")
"${compose[@]}" config --quiet
helper_image=${BACKUP_HELPER_IMAGE:-python:3.12.13-slim-bookworm}
volumes=(postgres-data redis-data qdrant-data object-data)
services=(caddy api worker beat postgres redis qdrant object-storage)
restart_required=0

restart_stack() {
  if [[ $restart_required -eq 1 ]]; then
    "${compose[@]}" start "${services[@]}" >/dev/null || {
      echo "Failed to restart services after backup; run docker compose start with the same environment file" >&2
      exit 1
    }
  fi
}
trap restart_stack EXIT

echo "Stopping write paths for a consistent cold backup..."
restart_required=1
"${compose[@]}" stop "${services[@]}"

for volume in "${volumes[@]}"; do
  docker volume inspect "${project_name}_${volume}" >/dev/null
  echo "Archiving ${volume}..."
  docker run --rm \
    -v "${project_name}_${volume}:/source:ro" \
    -v "${backup_root}:/backup" \
    "$helper_image" \
    tar -C /source -czf "/backup/${volume}.tar.gz" .
done

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$backup_root" && sha256sum ./*.tar.gz > SHA256SUMS)
else
  (cd "$backup_root" && shasum -a 256 ./*.tar.gz > SHA256SUMS)
fi

cat >"${backup_root}/metadata.json" <<EOF
{"created_at":"${timestamp}","compose_project":"${project_name}","format":"cold-volume-v1","volumes":["postgres-data","redis-data","qdrant-data","object-data"]}
EOF
chmod 600 "${backup_root}"/*

echo "Backup complete: ${backup_root}"
