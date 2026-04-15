#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  start_video2book_download_and_transcribe_tmux.sh [options passed through to worker]

Launcher-only options:
  --session-name <name>   tmux session name. Defaults to video2book-download-transcribe.
  -h, --help              Show this message.

All other arguments are forwarded to:
  scripts/process_video2book_download_and_transcribe.sh
EOF
}

session_name="${SESSION_NAME:-video2book-download-transcribe}"
worker_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-name)
      session_name="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      worker_args+=("$1")
      shift
      ;;
  esac
done

repo_root="${TRANSCRIPTION_REPO_ROOT:-$(pwd -P)}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
worker_script="$script_dir/process_video2book_download_and_transcribe.sh"

mkdir -p "$repo_root/downloads/logs"
log_file="$repo_root/downloads/logs/${session_name}_$(date +%Y%m%d_%H%M%S).log"

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "Session already exists: $session_name"
  tmux list-panes -t "$session_name" -F '#S:#I.#P #{pane_current_command}'
  exit 0
fi

quoted_args=()
for arg in "${worker_args[@]}"; do
  quoted_args+=("$(printf '%q' "$arg")")
done

tmux_command="cd $(printf '%q' "$repo_root") && bash $(printf '%q' "$worker_script")"
if [[ "${#quoted_args[@]}" -gt 0 ]]; then
  tmux_command+=" ${quoted_args[*]}"
fi
tmux_command+=" 2>&1 | tee $(printf '%q' "$log_file")"

tmux new-session -d -s "$session_name" "$tmux_command"

echo "Started tmux session: $session_name"
echo "Repo root: $repo_root"
echo "Log file: $log_file"
echo "Attach with: tmux attach -t $session_name"
