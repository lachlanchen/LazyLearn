#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  monitor_video2book_download_and_transcribe.sh [options] [-- <worker args>]

Options:
  --worker-session-name <name>   Worker tmux session to supervise.
  --download-session-name <name> Download tmux session to observe.
  --source-root <path>           Download source root watched by the worker.
  --repo-root <path>             Host repo root. Defaults to current directory.
  --video2book-root <path>       Video2Book checkout. Defaults to /home/lachlan/ProjectsLFS/Video2Book.
  --check-interval <seconds>     Monitor poll interval. Defaults to 180.
  --stale-seconds <seconds>      Restart worker if its log is older than this. Defaults to 2700.
  -h, --help                     Show this message.

Everything after `--` is passed to the worker launcher on restart.
EOF
}

abs_path() {
  python3 -c 'import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$1"
}

repo_root="$(pwd -P)"
video2book_root="${VIDEO2BOOK_ROOT:-/home/lachlan/ProjectsLFS/Video2Book}"
worker_session_name="video2book-download-transcribe"
download_session_name=""
source_root=""
check_interval="${VIDEO2BOOK_MONITOR_INTERVAL_SECONDS:-180}"
stale_seconds="${VIDEO2BOOK_MONITOR_STALE_SECONDS:-2700}"
worker_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --repo-root)
      repo_root="$2"
      shift 2
      ;;
    --video2book-root)
      video2book_root="$2"
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
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(abs_path "$repo_root")"
video2book_root="$(abs_path "$video2book_root")"
if [[ -n "$source_root" ]]; then
  source_root="$(abs_path "$source_root")"
fi

log_dir="$repo_root/downloads/logs"
mkdir -p "$log_dir"
monitor_log="$log_dir/${worker_session_name}-monitor_$(date +%Y%m%d_%H%M%S).log"
launcher_script="$repo_root/scripts/start_video2book_download_and_transcribe_tmux.sh"

log_line() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$monitor_log"
}

latest_worker_log() {
  ls -1t "$log_dir"/"${worker_session_name}"_*.log 2>/dev/null | head -n 1 || true
}

pending_video() {
  python3 "$video2book_root/videos2subtitles/transcribe_video.py" \
    --repo-root "$repo_root" \
    --source-root "$source_root" \
    --print-next
}

part_count() {
  if [[ -z "$source_root" || ! -d "$source_root" ]]; then
    echo 0
    return
  fi
  find "$source_root" -type f -name '*.part' | wc -l | tr -d ' '
}

session_is_busy() {
  local session_name="$1"
  local pane_commands

  if ! tmux has-session -t "$session_name" 2>/dev/null; then
    return 1
  fi

  pane_commands="$(tmux list-panes -t "$session_name" -F '#{pane_current_command}' 2>/dev/null || true)"
  while IFS= read -r pane_command; do
    case "$pane_command" in
      ""|bash|sh|zsh|fish|tmux)
        ;;
      *)
        return 0
        ;;
    esac
  done <<< "$pane_commands"

  return 1
}

download_active() {
  if [[ -z "$download_session_name" ]]; then
    return 1
  fi
  session_is_busy "$download_session_name"
}

restart_worker() {
  local quoted_args=()
  for arg in "${worker_args[@]}"; do
    quoted_args+=("$(printf '%q' "$arg")")
  done

  if tmux has-session -t "$worker_session_name" 2>/dev/null; then
    tmux kill-session -t "$worker_session_name"
    log_line "Killed existing stale worker session: $worker_session_name"
  fi

  local cmd=(bash "$launcher_script" --session-name "$worker_session_name")
  if [[ "${#worker_args[@]}" -gt 0 ]]; then
    cmd+=("${worker_args[@]}")
  fi

  log_line "Restarting worker: ${cmd[*]}"
  "${cmd[@]}" | tee -a "$monitor_log"
}

while true; do
  next_video="$(pending_video || true)"
  active_parts="$(part_count)"
  download_state="idle"
  if download_active; then
    download_state="running"
  fi

  if [[ -z "$next_video" && "$active_parts" == "0" && "$download_state" != "running" ]]; then
    log_line "No pending videos, no .part files, and no active download job. Monitor exiting."
    exit 0
  fi

  status="healthy"
  action="none"
  reason="worker active"

  if [[ -n "$next_video" || "$active_parts" != "0" || "$download_state" == "running" ]]; then
    work_remaining=1
  else
    work_remaining=0
  fi

  if (( work_remaining == 0 )); then
    status="complete"
    action="none"
    reason="no work remaining"
  elif ! tmux has-session -t "$worker_session_name" 2>/dev/null; then
    status="missing"
    action="restart"
    reason="worker tmux session missing"
  else
    worker_log="$(latest_worker_log)"
    if [[ -n "$worker_log" ]]; then
      now_epoch="$(date +%s)"
      log_epoch="$(stat -c %Y "$worker_log")"
      age_seconds=$(( now_epoch - log_epoch ))
      if (( age_seconds > stale_seconds )); then
        status="stalled"
        action="restart"
        reason="worker log stale for ${age_seconds}s"
      fi
    fi
  fi

  log_line "Monitor decision: status=$status action=$action download=$download_state part_files=$active_parts next=${next_video:-<none>} reason=$reason"

  if [[ "$action" == "restart" ]]; then
    restart_worker
  fi

  sleep "$check_interval"
done
