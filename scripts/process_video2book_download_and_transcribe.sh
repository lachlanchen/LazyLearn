#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  process_video2book_download_and_transcribe.sh [options] [-- <extra yt-dlp args>]

Modes:
  1. Download a playlist, then transcribe it:
     --playlist-url <youtube-playlist-url>

  2. Skip downloading and transcribe an existing source tree:
     --source-root <path-to-downloaded-videos>

Options:
  --repo-root <path>            Host repo root. Defaults to current directory.
  --video2book-root <path>      Video2Book checkout. Defaults to /home/lachlan/ProjectsLFS/Video2Book.
  --workspace <path>            Downloader workspace for yt-dlp discovery.
  --download-root <path>        Download root. Defaults to <repo-root>/downloads.
  --download-log-root <path>    Download log root. Defaults to <download-root>/logs.
  --playlist-start <n>          First playlist item to download.
  --playlist-end <n>            Last playlist item to download.
  --transcribe-model <name>     Whisper model. Defaults to large-v3.
  --min-free-gpu-mib <n>        Wait until at least this much GPU memory is free.
  --watch                       Keep polling for newly completed videos.
  --poll-seconds <n>            Poll interval in watch mode. Defaults to 60.
  --idle-polls-before-exit <n>  Exit after this many empty polls in watch mode. Defaults to 10.
  --force                       Rebuild transcripts even if outputs already exist.
  --dry-run                     Print resolved commands and exit.
  -h, --help                    Show this message.

Examples:
  process_video2book_download_and_transcribe.sh \
    --playlist-url 'https://www.youtube.com/playlist?list=...' \
    --playlist-start 1 --playlist-end 3

  process_video2book_download_and_transcribe.sh \
    --source-root ./downloads/how-to-speak-and-write
EOF
}

abs_path() {
  python3 -c 'import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$1"
}

extract_playlist_id() {
  python3 - "$1" <<'PY'
from urllib.parse import parse_qs, urlparse
import sys

url = sys.argv[1]
query = parse_qs(urlparse(url).query)
playlist_ids = query.get("list")
if not playlist_ids or not playlist_ids[0]:
    raise SystemExit(f"Could not extract a playlist id from URL: {url}")
print(playlist_ids[0])
PY
}

default_min_free_gpu_mib() {
  case "$1" in
    large|large-v1|large-v2|large-v3)
      echo 14000
      ;;
    medium|medium.en)
      echo 9000
      ;;
    *)
      echo 4000
      ;;
  esac
}

wait_for_gpu_memory() {
  local min_free_gpu_mib="$1"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; skipping GPU memory gate."
    return 0
  fi

  while true; do
    local free_mib
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ -n "$free_mib" ]] && (( free_mib >= min_free_gpu_mib )); then
      return 0
    fi
    echo "Waiting for GPU memory: ${free_mib:-unknown} MiB free, need ${min_free_gpu_mib} MiB"
    sleep 60
  done
}

repo_root="$(pwd -P)"
video2book_root="${VIDEO2BOOK_ROOT:-/home/lachlan/ProjectsLFS/Video2Book}"
workspace="${VIDEO2BOOK_WORKSPACE:-/home/lachlan/ProjectsLFS/YoutubeDownloader}"
download_root=""
download_log_root=""
playlist_url=""
source_root="${SOURCE_ROOT:-}"
playlist_start=""
playlist_end=""
transcribe_model="${TRANSCRIBE_MODEL:-large-v3}"
min_free_gpu_mib="${MIN_FREE_GPU_MIB:-}"
watch_mode=0
poll_seconds="${POLL_SECONDS:-60}"
idle_polls_before_exit="${IDLE_POLLS_BEFORE_EXIT:-10}"
force=0
dry_run=0
download_extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      repo_root="$2"
      shift 2
      ;;
    --video2book-root)
      video2book_root="$2"
      shift 2
      ;;
    --workspace)
      workspace="$2"
      shift 2
      ;;
    --download-root)
      download_root="$2"
      shift 2
      ;;
    --download-log-root)
      download_log_root="$2"
      shift 2
      ;;
    --playlist-url)
      playlist_url="$2"
      shift 2
      ;;
    --source-root)
      source_root="$2"
      shift 2
      ;;
    --playlist-start)
      playlist_start="$2"
      shift 2
      ;;
    --playlist-end)
      playlist_end="$2"
      shift 2
      ;;
    --transcribe-model)
      transcribe_model="$2"
      shift 2
      ;;
    --min-free-gpu-mib)
      min_free_gpu_mib="$2"
      shift 2
      ;;
    --watch)
      watch_mode=1
      shift
      ;;
    --poll-seconds)
      poll_seconds="$2"
      shift 2
      ;;
    --idle-polls-before-exit)
      idle_polls_before_exit="$2"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      download_extra_args=("$@")
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
workspace="$(abs_path "$workspace")"
download_root="${download_root:-$repo_root/downloads}"
download_root="$(abs_path "$download_root")"
download_log_root="${download_log_root:-$download_root/logs}"
download_log_root="$(abs_path "$download_log_root")"

if [[ -n "$playlist_url" && -n "$source_root" ]]; then
  echo "Use either --playlist-url or --source-root, not both." >&2
  exit 2
fi

if [[ -z "$playlist_url" && -z "$source_root" ]]; then
  echo "Either --playlist-url or --source-root is required." >&2
  exit 2
fi

if [[ -n "$source_root" ]]; then
  source_root="$(abs_path "$source_root")"
fi

if [[ -z "$min_free_gpu_mib" ]]; then
  min_free_gpu_mib="$(default_min_free_gpu_mib "$transcribe_model")"
fi

mkdir -p "$repo_root/subtitles" "$repo_root/markdown" "$download_root" "$download_log_root"

download_cmd=()
if [[ -n "$playlist_url" ]]; then
  playlist_id="$(extract_playlist_id "$playlist_url")"
  source_root="$download_root/$playlist_id"
  download_cmd=(
    python3
    "$video2book_root/playlist2videos/download_playlist.py"
    --playlist-url "$playlist_url"
    --workspace "$workspace"
    --download-root "$download_root"
    --log-root "$download_log_root"
  )
  if [[ -n "$playlist_start" ]]; then
    download_cmd+=(--playlist-start "$playlist_start")
  fi
  if [[ -n "$playlist_end" ]]; then
    download_cmd+=(--playlist-end "$playlist_end")
  fi
  if [[ "${#download_extra_args[@]}" -gt 0 ]]; then
    download_cmd+=(-- "${download_extra_args[@]}")
  fi
fi

if [[ $dry_run -eq 1 ]]; then
  echo "repo_root: $repo_root"
  echo "video2book_root: $video2book_root"
  echo "workspace: $workspace"
  echo "download_root: $download_root"
  echo "download_log_root: $download_log_root"
  echo "source_root: $source_root"
  echo "transcribe_model: $transcribe_model"
  echo "min_free_gpu_mib: $min_free_gpu_mib"
  echo "watch_mode: $watch_mode"
  echo "poll_seconds: $poll_seconds"
  echo "idle_polls_before_exit: $idle_polls_before_exit"
  echo "force: $force"
  if [[ "${#download_cmd[@]}" -gt 0 ]]; then
    printf 'download_cmd:'
    printf ' %q' "${download_cmd[@]}"
    printf '\n'
  else
    echo "download_cmd: <skipped>"
  fi
  echo "transcribe_cmd: python3 $video2book_root/videos2subtitles/transcribe_video.py --repo-root $repo_root --source-root $source_root --model $transcribe_model --video <next-video>"
  exit 0
fi

if [[ "${#download_cmd[@]}" -gt 0 ]]; then
  printf 'Starting download command:'
  printf ' %q' "${download_cmd[@]}"
  printf '\n'
  "${download_cmd[@]}"
fi

idle_polls=0
while true; do
  next_video="$(python3 "$video2book_root/videos2subtitles/transcribe_video.py" \
    --repo-root "$repo_root" \
    --source-root "$source_root" \
    --print-next)"

  if [[ -z "$next_video" ]]; then
    if [[ $watch_mode -eq 1 ]]; then
      ((idle_polls+=1))
      echo "No pending videos remain. Watch poll ${idle_polls}/${idle_polls_before_exit}."
      if (( idle_polls >= idle_polls_before_exit )); then
        echo "Idle limit reached; exiting watch mode."
        exit 0
      fi
      sleep "$poll_seconds"
      continue
    fi
    echo "No pending videos remain."
    exit 0
  fi

  idle_polls=0
  wait_for_gpu_memory "$min_free_gpu_mib"
  echo "Processing $next_video with model $transcribe_model"

  transcribe_cmd=(
    python3
    "$video2book_root/videos2subtitles/transcribe_video.py"
    --repo-root "$repo_root"
    --source-root "$source_root"
    --model "$transcribe_model"
    --video "$next_video"
  )
  if [[ $force -eq 1 ]]; then
    transcribe_cmd+=(--force)
  fi

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${transcribe_cmd[@]}"
done
