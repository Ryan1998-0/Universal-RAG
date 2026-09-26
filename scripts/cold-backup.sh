#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
  printf 'Usage: %s [backup-directory]\n' "$0"
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
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

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
backup_root=${1:-"./backups/${timestamp}"}
mkdir -p "$backup_root"
backup_root=$(cd "$backup_root" && pwd)
first_entry=$(find "$backup_root" -mindepth 1 -maxdepth 1 -print -quit)
[[ -z $first_entry ]] || {
  echo "Backup directory must be empty: ${backup_root}" >&2
  exit 1
}
chmod 700 "$backup_root"

helper_image=${BACKUP_HELPER_IMAGE:-python:3.12.13-slim-bookworm}
volumes=(postgres-data redis-data qdrant-data object-data)
services=(caddy api worker beat postgres redis qdrant object-storage)
restart_required=0

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
    echo "Cannot take a cold backup while ${service} is running" >&2
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

restart_stack() {
  exit_status=$?
  trap - EXIT
  if [[ $restart_required -eq 1 ]]; then
    if ! compose up -d; then
      echo "Failed to restart the stack after backup; manual recovery is required." >&2
      exit_status=1
    elif ! wait_for_stack_ready; then
      echo "Stack restart did not become ready after backup; manual recovery is required." >&2
      exit_status=1
    fi
  fi
  if [[ $exit_status -eq 0 ]]; then
    echo "Backup complete; API health passed and Caddy, Worker and Beat are running: ${backup_root}"
  fi
  exit "$exit_status"
}
trap restart_stack EXIT

echo "Stopping write paths for a consistent cold backup..."
restart_required=1
compose stop "${services[@]}"

for volume in "${volumes[@]}"; do
  echo "Archiving ${volume}..."
  docker run --rm \
    -v "${project_name}_${volume}:/source:ro" \
    "$helper_image" \
    tar -C /source -czf - . >"${backup_root}/${volume}.tar.gz"
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

