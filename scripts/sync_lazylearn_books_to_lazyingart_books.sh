#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest_root="${1:-/home/lachlan/Nutstore Files/Projects/LazyingArtBooks/lazylearn}"
full_size_dir="$dest_root/full size"
pocket_size_1_0_dir="$dest_root/pocket size 1.0"
pocket_size_1_2_dir="$dest_root/pocket size 1.2"

courses=(
  "how-to-speak-and-write"
  "justice-with-michael-sandel"
)

rm -rf \
  "$full_size_dir" \
  "$pocket_size_1_0_dir" \
  "$pocket_size_1_2_dir" \
  "$dest_root/how-to-speak-and-write" \
  "$dest_root/justice-with-michael-sandel"

mkdir -p "$full_size_dir" "$pocket_size_1_0_dir" "$pocket_size_1_2_dir"

for course in "${courses[@]}"; do
  src_dir="$repo_root/$course"

  if [[ ! -d "$src_dir" ]]; then
    printf 'Skipping missing course folder: %s\n' "$src_dir" >&2
    continue
  fi

  found_any=0
  while IFS= read -r pdf_path; do
    pdf_name="$(basename "$pdf_path")"
    found_any=1
    case "$pdf_name" in
      *-pocket-1_2x.pdf)
        install -m 0644 "$pdf_path" "$pocket_size_1_2_dir/$pdf_name"
        ;;
      *-pocket.pdf)
        install -m 0644 "$pdf_path" "$pocket_size_1_0_dir/$pdf_name"
        ;;
      *.pdf)
        install -m 0644 "$pdf_path" "$full_size_dir/$pdf_name"
        ;;
    esac
  done < <(find "$src_dir" -maxdepth 1 -type f -name '*.pdf' | sort)

  if [[ "$found_any" -eq 0 ]]; then
    printf 'No PDFs found in: %s\n' "$src_dir" >&2
    continue
  fi

  printf 'Collected LazyLearn PDFs from: %s\n' "$src_dir"
done

printf 'Synced LazyLearn full-size PDFs to: %s\n' "$full_size_dir"
printf 'Synced LazyLearn pocket 1.0 PDFs to: %s\n' "$pocket_size_1_0_dir"
printf 'Synced LazyLearn pocket 1.2 PDFs to: %s\n' "$pocket_size_1_2_dir"
