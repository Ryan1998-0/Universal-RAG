#!/usr/bin/env bash

# Source this file from host-side production operations. The lock path must be
# stable across checkouts and shared by every operator of the same Compose host.
acquire_maintenance_lock() {
  local lock_file=${RAG_MAINTENANCE_LOCK_FILE:-}
  local lock_dir lock_name script_dir project_dir owner mode

  command -v flock >/dev/null 2>&1 || {
    echo "flock is required for production maintenance" >&2
    return 1
  }
  command -v stat >/dev/null 2>&1 || {
    echo "stat is required for production maintenance" >&2
    return 1
  }
  [[ $lock_file == /* ]] || {
    echo "RAG_MAINTENANCE_LOCK_FILE must be an absolute path outside the checkout" >&2
    return 1
  }
  lock_dir=${lock_file%/*}
  lock_name=${lock_file##*/}
  [[ -n $lock_name ]] || {
    echo "Maintenance lock path must name a file" >&2
    return 1
  }
  [[ -d $lock_dir && -w $lock_dir ]] || {
    echo "Maintenance lock directory must already exist and be writable: ${lock_dir}" >&2
    return 1
  }
  lock_dir=$(cd "$lock_dir" && pwd -P) || return 1
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P) || return 1
  project_dir=$(cd "$script_dir/.." && pwd -P) || return 1
  [[ $lock_dir != "$project_dir" && $lock_dir != "$project_dir/"* ]] || {
    echo "Maintenance lock directory must be outside the checkout" >&2
    return 1
  }
  read -r owner mode < <(stat -c '%u %a' -- "$lock_dir") || return 1
  [[ $owner == "$(id -u)" ]] && (( (8#$mode & 8#22) == 0 )) || {
    echo "Maintenance lock directory must belong to the deployment account and not be group/world writable" >&2
    return 1
  }
  lock_file="${lock_dir}/${lock_name}"
  [[ ! -L $lock_file && ( ! -e $lock_file || -f $lock_file ) ]] || {
    echo "Maintenance lock path must be a regular file, not a symlink" >&2
    return 1
  }
  if [[ -e $lock_file ]]; then
    read -r owner mode < <(stat -c '%u %a' -- "$lock_file") || return 1
    [[ $owner == "$(id -u)" ]] && (( (8#$mode & 8#22) == 0 )) || {
      echo "Maintenance lock file must belong to the deployment account and not be group/world writable" >&2
      return 1
    }
  fi

  umask 077
  exec {rag_maintenance_lock_fd}>>"$lock_file" || return 1
  flock -n "$rag_maintenance_lock_fd" || {
    echo "Another production maintenance operation holds ${lock_file}" >&2
    return 1
  }
}
