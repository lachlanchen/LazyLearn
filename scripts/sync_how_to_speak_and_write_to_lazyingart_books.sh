#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest_root="${1:-/home/lachlan/Nutstore Files/Projects/LazyingArtBooks/lazylearn/full size}"
slug="how-to-speak-and-write"

publish_root="$repo_root/$slug"
dest_dir="$dest_root"

mkdir -p "$dest_dir"
install -m 0644 "$publish_root/$slug.pdf" "$dest_dir/$slug.pdf"
install -m 0644 "$publish_root/${slug}-zh.pdf" "$dest_dir/${slug}-zh.pdf"
install -m 0644 "$publish_root/${slug}-jp.pdf" "$dest_dir/${slug}-jp.pdf"
echo "Synced publish PDFs to: $dest_dir"
