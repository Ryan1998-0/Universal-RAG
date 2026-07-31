#!/bin/sh
set -eu

if [ "${RAG_RUN_MIGRATIONS:-false}" = "true" ]; then
  alembic upgrade head
fi

exec "$@"
