#!/usr/bin/env bash
set -euo pipefail
umask 077

if (( $# == 0 )); then
  echo "Usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/maintenance-lock.sh"
acquire_maintenance_lock
"$@"
