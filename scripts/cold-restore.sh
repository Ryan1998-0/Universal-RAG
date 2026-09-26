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
command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }
command -v timeout >/dev/null 2>&1 || { echo "timeout is required" >&2; exit 1; }
docker compose version >/dev/null

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
compose_env_file=${RAG_COMPOSE_ENV_FILE:-"${project_dir}/.env.production"}
if [[ $compose_env_file != /* ]]; then
  compose_env_file="$(pwd)/${compose_env_file}"
fi
[[ -f $compose_env_file && -r $compose_env_file ]] || {
  echo "Readable Compose environment file is required: ${compose_env_file}" >&2
  exit 1
}
project_name=$(
  docker compose --project-directory "$project_dir" --env-file "$compose_env_file" \
    -f "$project_dir/compose.yaml" config --format json \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["name"])'
)
[[ $project_name =~ ^[a-z0-9][a-z0-9_-]*$ ]] || {
  echo "Compose project name is invalid" >&2
  exit 1
}
compose() {
  docker compose --project-directory "$project_dir" --env-file "$compose_env_file" \
    -f "$project_dir/compose.yaml" --project-name "$project_name" "$@"
}
compose config --quiet

backup_root=$(cd "$2" && pwd)
helper_image=${BACKUP_HELPER_IMAGE:-python:3.12.13-slim-bookworm}
volumes=(postgres-data redis-data qdrant-data object-data)
services=(caddy api worker beat postgres redis qdrant object-storage)

for volume in "${volumes[@]}"; do
  test -f "${backup_root}/${volume}.tar.gz" || {
    echo "Missing backup archive: ${volume}.tar.gz" >&2
    exit 1
  }
done

[[ -f ${backup_root}/SHA256SUMS ]] || {
  echo "Missing backup checksum manifest" >&2
  exit 1
}
checksum_pattern='^[[:xdigit:]]{64}  \./(postgres-data|redis-data|qdrant-data|object-data)\.tar\.gz$'
declare -A seen_checksum_files=()
while IFS= read -r checksum_line; do
  [[ $checksum_line =~ $checksum_pattern ]] || {
    echo "Backup checksum manifest has an unexpected entry" >&2
    exit 1
  }
  checksum_name=${BASH_REMATCH[1]}
  [[ -z ${seen_checksum_files[$checksum_name]+x} ]] || {
    echo "Backup checksum manifest has a duplicate entry" >&2
    exit 1
  }
  seen_checksum_files[$checksum_name]=1
done <"${backup_root}/SHA256SUMS"
[[ ${#seen_checksum_files[@]} -eq ${#volumes[@]} ]] || {
  echo "Backup checksum manifest is incomplete" >&2
  exit 1
}

python3 - "${backup_root}/metadata.json" "$project_name" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
except (OSError, ValueError) as exc:
    raise SystemExit(f"Backup metadata cannot be read: {exc}") from exc
if (
    metadata.get("format") != "cold-volume-v1"
    or metadata.get("compose_project") != sys.argv[2]
    or metadata.get("volumes")
    != ["postgres-data", "redis-data", "qdrant-data", "object-data"]
):
    raise SystemExit("Backup metadata does not match this Compose project")
PY

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$backup_root" && sha256sum --check SHA256SUMS)
else
  (cd "$backup_root" && shasum -a 256 -c SHA256SUMS)
fi

for volume in "${volumes[@]}"; do
  docker volume inspect "${project_name}_${volume}" >/dev/null
done

for service in migrate bootstrap provision; do
  running_writer=$(docker ps --quiet \
    --filter "label=com.docker.compose.project=${project_name}" \
    --filter "label=com.docker.compose.service=${service}") || {
    echo "Cannot inspect running Compose writers" >&2
    exit 1
  }
  if [[ -n $running_writer ]]; then
    echo "Cannot restore while ${service} is running" >&2
    exit 1
  fi
done

restart_ready_timeout=${RAG_RESTART_READY_TIMEOUT_SECONDS:-600}
[[ $restart_ready_timeout =~ ^[1-9][0-9]{0,2}$ ]] \
  && (( restart_ready_timeout <= 600 )) || {
  echo "RAG_RESTART_READY_TIMEOUT_SECONDS must be between 1 and 600" >&2
  exit 1
}

compose_timed() {
  timeout 10s docker compose --project-directory "$project_dir" --env-file "$compose_env_file" \
    -f "$project_dir/compose.yaml" --project-name "$project_name" "$@"
}

wait_for_stack_ready() {
  local deadline=$((SECONDS + restart_ready_timeout))
  local api_id worker_id beat_id caddy_id api_state worker_state beat_state caddy_state
  while :; do
    api_id=$(compose_timed ps -q api) || { echo "Cannot inspect API container" >&2; return 1; }
    worker_id=$(compose_timed ps -q worker) || { echo "Cannot inspect Worker container" >&2; return 1; }
    beat_id=$(compose_timed ps -q beat) || { echo "Cannot inspect Beat container" >&2; return 1; }
    caddy_id=$(compose_timed ps -q caddy) || { echo "Cannot inspect Caddy container" >&2; return 1; }
    if [[ -n $api_id && -n $worker_id && -n $beat_id && -n $caddy_id ]]; then
      api_state=$(timeout 10s docker inspect --format \
        '{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}' \
        "$api_id") || { echo "Cannot inspect API health" >&2; return 1; }
      worker_state=$(timeout 10s docker inspect --format '{{.State.Running}}' \
        "$worker_id") || { echo "Cannot inspect Worker state" >&2; return 1; }
      beat_state=$(timeout 10s docker inspect --format '{{.State.Running}}' \
        "$beat_id") || { echo "Cannot inspect Beat state" >&2; return 1; }
      caddy_state=$(timeout 10s docker inspect --format '{{.State.Running}}' \
        "$caddy_id") || { echo "Cannot inspect Caddy state" >&2; return 1; }
      if [[ $api_state == "true healthy" && $worker_state == true \
        && $beat_state == true && $caddy_state == true ]]; then
        return 0
      fi
    fi
    if (( SECONDS >= deadline )); then
      echo "Stack did not become ready within ${restart_ready_timeout} seconds" >&2
      return 1
    fi
    sleep 5 || { echo "Readiness polling was interrupted" >&2; return 1; }
  done
}

for volume in "${volumes[@]}"; do
  echo "Checking ${volume} archive before stopping services..."
  docker run --rm \
    -v "${backup_root}:/backup:ro" \
    "$helper_image" \
    tar -tzf "/backup/${volume}.tar.gz" >/dev/null
done

restore_started=0
restore_exit() {
  exit_status=$?
  if [[ $restore_started -eq 1 && $exit_status -ne 0 ]]; then
    if ! compose stop "${services[@]}" >/dev/null; then
      echo "Could not stop every service after restore failure; isolate traffic immediately." >&2
    fi
    echo "Restore failed; data may be partial. Recover from a verified backup before reopening traffic." >&2
  fi
}
trap restore_exit EXIT

echo "Stopping services before destructive restore..."
restore_started=1
compose stop "${services[@]}"

for volume in "${volumes[@]}"; do
  echo "Restoring ${volume}..."
  docker run --rm \
    -v "${project_name}_${volume}:/target" \
    -v "${backup_root}:/backup:ro" \
    "$helper_image" \
    sh -euc "find /target -mindepth 1 -maxdepth 1 -exec rm -rf {} +; tar -C /target -xzf /backup/${volume}.tar.gz"
done

echo "Starting restored stack..."
compose up -d
wait_for_stack_ready
restore_started=0
echo "Restore complete; API health passed and Caddy, Worker and Beat are running. Run the smoke test before reopening traffic."
