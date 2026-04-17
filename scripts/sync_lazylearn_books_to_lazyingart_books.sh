#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest_root="${1:-/home/lachlan/Nutstore Files/Projects/LazyingArtBooks/lazylearn}"

courses=(
  "how-to-speak-and-write"
  "justice-with-michael-sandel"
)

for course in "${courses[@]}"; do
  src_dir="$repo_root/$course"
  dest_dir="$dest_root/$course"

  if [[ ! -d "$src_dir" ]]; then
    printf 'Skipping missing course folder: %s\n' "$src_dir" >&2
    continue
  fi

  rm -rf "$dest_dir"
  mkdir -p "$dest_dir"

  found_any=0
  while IFS= read -r pdf_path; do
    found_any=1
    install -m 0644 "$pdf_path" "$dest_dir/$(basename "$pdf_path")"
  done < <(find "$src_dir" -maxdepth 1 -type f -name '*.pdf' | sort)

  if [[ "$found_any" -eq 0 ]]; then
    printf 'No PDFs found in: %s\n' "$src_dir" >&2
    continue
  fi

  printf 'Synced LazyLearn PDFs to: %s\n' "$dest_dir"
done
