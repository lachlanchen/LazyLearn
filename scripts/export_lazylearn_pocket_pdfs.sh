#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$repo_root/generated_course_notes/lazylearn"
output_dir="$repo_root/pocket_books/lazylearn"
nutstore_root="${NUTSTORE_ROOT:-/home/lachlan/Nutstore Files/Projects/LazyingArtBooks/lazylearn}"

"$repo_root/Video2Book/scripts/export_course_pocket_pdfs.sh" \
  --host-root "$repo_root" \
  --source-dir "$source_dir" \
  --output-dir "$output_dir" \
  --no-rsync \
  "$@"

mkdir -p \
  "$nutstore_root/how-to-speak-and-write" \
  "$nutstore_root/justice-with-michael-sandel"

install -m 0644 \
  "$output_dir/how-you-speak-and-write_pocket.pdf" \
  "$nutstore_root/how-to-speak-and-write/how-to-speak-and-write-pocket.pdf"

install -m 0644 \
  "$output_dir/justice-with-michael-sandel_pocket.pdf" \
  "$nutstore_root/justice-with-michael-sandel/justice-with-michael-sandel-pocket.pdf"

printf 'Exported pocket PDFs to: %s\n' "$output_dir"
printf 'Synced pocket PDFs to: %s\n' "$nutstore_root"
