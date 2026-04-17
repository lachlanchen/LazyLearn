#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/publish_lazylearn_pocket_pdfs.sh [options]

Publish generated LazyLearn pocket PDFs by:
1. hardlinking them into the root course folders
2. syncing them into Nutstore

Options:
  --course <slug>         Restrict to one published course slug
  --repo-root <path>      Repo root (default: parent of this script)
  --nutstore-root <path>  Nutstore LazyingArtBooks/lazylearn root
  -h, --help              Show this help
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nutstore_root="${NUTSTORE_ROOT:-/home/lachlan/Nutstore Files/Projects/LazyingArtBooks/lazylearn}"
course_filter=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --course)
      course_filter="${2:-}"
      shift 2
      ;;
    --repo-root)
      repo_root="${2:-}"
      shift 2
      ;;
    --nutstore-root)
      nutstore_root="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

link_file() {
  local src="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  rm -f "$dest"
  ln "$src" "$dest"
}

copy_file() {
  local src="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  install -m 0644 "$src" "$dest"
}

publish_course() {
  local source_slug="$1"
  local publish_slug="$2"

  if [[ -n "$course_filter" && "$course_filter" != "$publish_slug" ]]; then
    return 0
  fi

  local base_src="$repo_root/all_notes/pocket_books/${source_slug}_pocket.pdf"
  local onepointtwo_src="$repo_root/all_notes/pocket_books_1_2x/${source_slug}_pocket_1_2x.pdf"
  local publish_dir="$repo_root/$publish_slug"
  local nutstore_dir="$nutstore_root/$publish_slug"
  local base_dest="$publish_dir/${publish_slug}-pocket.pdf"
  local onepointtwo_dest="$publish_dir/${publish_slug}-pocket-1_2x.pdf"

  if [[ -f "$base_src" ]]; then
    link_file "$base_src" "$base_dest"
    copy_file "$base_src" "$nutstore_dir/${publish_slug}-pocket.pdf"
  fi

  if [[ -f "$onepointtwo_src" ]]; then
    link_file "$onepointtwo_src" "$onepointtwo_dest"
    copy_file "$onepointtwo_src" "$nutstore_dir/${publish_slug}-pocket-1_2x.pdf"
  fi
}

publish_course "how-you-speak-and-write" "how-to-speak-and-write"
publish_course "justice-with-michael-sandel" "justice-with-michael-sandel"

printf 'Published LazyLearn pocket PDFs into root course folders and Nutstore: %s\n' "$nutstore_root"
