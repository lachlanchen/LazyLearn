#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  start_video2book_download_and_transcribe_monitor_tmux.sh [options] [-- <worker args>]

Launcher options:
  --session-name <name>           Monitor tmux session name.
  --worker-session-name <name>    Worker tmux session name to supervise.
  --download-session-name <name>  Download tmux session name to observe.
  --source-root <path>            Download source root watched by the worker.
  --check-interval <seconds>      Monitor poll interval. Defaults to 180.
  --stale-seconds <seconds>       Restart threshold. Defaults to 2700.
  -h, --help                      Show this message.

Everything after `--` is passed to the worker launcher when the monitor restarts it.
EOF
}

session_name="${VIDEO2BOOK_MONITOR_SESSION_NAME:-video2book-download-transcribe-monitor}"
worker_session_name="video2book-download-transcribe"
download_session_name=""
source_root=""
check_interval=""
stale_seconds=""
worker_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-name)
      session_name="$2"
      shift 2
      ;;
    --worker-session-name)
      worker_session_name="$2"
      shift 2
      ;;
    --download-session-name)
      download_session_name="$2"
      shift 2
      ;;
    --source-root)
      source_root="$2"
      shift 2
      ;;
    --check-interval)
      check_interval="$2"
      shift 2
      ;;
    --stale-seconds)
      stale_seconds="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      worker_args=("$@")
      break
      ;;
    *)
      worker_args+=("$1")
      shift
      ;;
  esac
done

repo_root="${TRANSCRIPTION_REPO_ROOT:-$(pwd -P)}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
monitor_script="$script_dir/monitor_video2book_download_and_transcribe.sh"
mkdir -p "$repo_root/downloads/logs"
log_file="$repo_root/downloads/logs/${session_name}_$(date +%Y%m%d_%H%M%S).log"

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "Session already exists: $session_name"
  tmux list-panes -t "$session_name" -F '#S:#I.#P #{pane_current_command}'
  exit 0
fi

cmd=(bash "$monitor_script" --repo-root "$repo_root" --worker-session-name "$worker_session_name")
if [[ -n "$download_session_name" ]]; then
  cmd+=(--download-session-name "$download_session_name")
fi
if [[ -n "$source_root" ]]; then
  cmd+=(--source-root "$source_root")
fi
if [[ -n "$check_interval" ]]; then
  cmd+=(--check-interval "$check_interval")
fi
if [[ -n "$stale_seconds" ]]; then
  cmd+=(--stale-seconds "$stale_seconds")
fi
if [[ "${#worker_args[@]}" -gt 0 ]]; then
  cmd+=(-- "${worker_args[@]}")
fi

quoted_cmd=()
for arg in "${cmd[@]}"; do
  quoted_cmd+=("$(printf '%q' "$arg")")
done

tmux_command="cd $(printf '%q' "$repo_root") && ${quoted_cmd[*]} 2>&1 | tee $(printf '%q' "$log_file")"
tmux new-session -d -s "$session_name" "$tmux_command"

echo "Started monitor tmux session: $session_name"
echo "Worker session: $worker_session_name"
if [[ -n "$download_session_name" ]]; then
  echo "Download session: $download_session_name"
fi
echo "Log file: $log_file"
echo "Attach with: tmux attach -t $session_name"
