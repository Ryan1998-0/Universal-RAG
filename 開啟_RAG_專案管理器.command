#!/bin/zsh
set -eu

project_dir="${0:A:h}"
page_url="http://127.0.0.1:8765/architecture.html"
log_file="$project_dir/.local/rag-project-manager.log"

cd "$project_dir"

if ! curl -fsS "$page_url" >/dev/null 2>&1; then
  if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
    echo "找不到專案 Python 環境：$project_dir/.venv/bin/python"
    echo "請先依照 README.md 完成本機環境安裝。"
    read -r "?按 Enter 關閉…"
    exit 1
  fi

  mkdir -p "$project_dir/.local"
  nohup "$project_dir/.venv/bin/python" -m rag_demo.web_app >"$log_file" 2>&1 &
  server_pid=$!

  for attempt in {1..30}; do
    if curl -fsS "$page_url" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "RAG 本機服務啟動失敗，最近的紀錄如下："
      tail -n 20 "$log_file"
      read -r "?按 Enter 關閉…"
      exit 1
    fi
    sleep 0.5
  done
fi

if curl -fsS "$page_url" >/dev/null 2>&1; then
  open "$page_url"
  echo "RAG 專案管理器已開啟：$page_url"
else
  echo "服務未能在預期時間內啟動，請查看：$log_file"
  read -r "?按 Enter 關閉…"
  exit 1
fi
