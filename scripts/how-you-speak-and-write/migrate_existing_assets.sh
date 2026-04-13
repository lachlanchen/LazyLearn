#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
course_rel="lazylearn/how-you-speak-and-write"
old_download_dir="$repo_root/downloads/how-to-speak-and-write"
new_download_dir="$repo_root/downloads/$course_rel"
new_log_dir="$repo_root/downloads/logs/lazylearn/how-you-speak-and-write"
old_archive_file="$old_download_dir/logs/archive.txt"
new_archive_file="$new_log_dir/urls.archive"

mkdir -p \
  "$new_download_dir" \
  "$new_log_dir" \
  "$repo_root/subtitles/$course_rel" \
  "$repo_root/markdown/$course_rel"

shopt -s nullglob

for video in "$old_download_dir"/*; do
  [[ -f "$video" ]] || continue
  case "$video" in
    *.mp4|*.mkv|*.webm|*.m4a|*.mp3)
      destination="$new_download_dir/$(basename "$video")"
      if [[ ! -e "$destination" ]]; then
        mv "$video" "$destination"
      fi
      ;;
  esac
done

if [[ -f "$old_archive_file" && ! -f "$new_archive_file" ]]; then
  cp "$old_archive_file" "$new_archive_file"
fi

for video in "$new_download_dir"/*; do
  [[ -f "$video" ]] || continue
  case "$video" in
    *.mp4|*.mkv|*.webm|*.m4a|*.mp3) ;;
    *) continue ;;
  esac

  basename_with_ext="$(basename "$video")"
  stem="${basename_with_ext%.*}"

  old_subtitle="$repo_root/subtitles/$stem.srt"
  new_subtitle="$repo_root/subtitles/$course_rel/$stem.srt"
  if [[ -f "$old_subtitle" && ! -e "$new_subtitle" ]]; then
    mv "$old_subtitle" "$new_subtitle"
  fi

  old_markdown="$repo_root/markdown/$stem.md"
  new_markdown="$repo_root/markdown/$course_rel/$stem.md"
  if [[ -f "$old_markdown" && ! -e "$new_markdown" ]]; then
    mv "$old_markdown" "$new_markdown"
  fi

  if [[ -f "$new_markdown" ]]; then
    python3 - "$new_markdown" "$course_rel/$basename_with_ext" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source_rel = sys.argv[2]
lines = path.read_text(encoding="utf-8").splitlines()
updated = False
for index, line in enumerate(lines):
    if line.startswith("Source: "):
        lines[index] = f"Source: {source_rel}"
        updated = True
        break
if not updated:
    insert_at = 2 if len(lines) >= 2 else len(lines)
    lines[insert_at:insert_at] = ["", f"Source: {source_rel}"]
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
  fi
done

printf 'download_dir=%s\n' "$new_download_dir"
printf 'subtitle_dir=%s\n' "$repo_root/subtitles/$course_rel"
printf 'markdown_dir=%s\n' "$repo_root/markdown/$course_rel"
printf 'videos=%s\n' "$(find "$new_download_dir" -maxdepth 1 -type f | wc -l | tr -d ' ')"
printf 'subtitles=%s\n' "$(find "$repo_root/subtitles/$course_rel" -maxdepth 1 -type f -name '*.srt' | wc -l | tr -d ' ')"
printf 'markdown=%s\n' "$(find "$repo_root/markdown/$course_rel" -maxdepth 1 -type f -name '*.md' | wc -l | tr -d ' ')"
