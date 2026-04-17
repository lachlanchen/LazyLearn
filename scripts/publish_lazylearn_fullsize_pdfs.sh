#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/publish_lazylearn_fullsize_pdfs.sh [options]

Rebuild and publish LazyLearn full-size English PDFs by:
1. syncing the shared Video2Book A4 header preamble into generated courses
2. rebuilding the generated full-size source PDFs
3. publishing the rebuilt PDFs into the root book folders
4. mirroring the full-size publish set into all_notes/

Options:
  --course <slug>           Restrict to one published course slug
  --repo-root <path>        Repo root (default: parent of this script)
  --video2book-root <path>  Video2Book root (default: <repo-root>/Video2Book)
  -h, --help                Show this help
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
video2book_root=""
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
    --video2book-root)
      video2book_root="${2:-}"
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

if [[ -z "$video2book_root" ]]; then
  video2book_root="$repo_root/Video2Book"
fi

shared_preamble="$video2book_root/subtitles2notes/templates/lecture_notes_common_preamble.tex"
if [[ ! -f "$shared_preamble" ]]; then
  echo "Missing shared preamble: $shared_preamble" >&2
  exit 1
fi

hardlink_or_copy() {
  local src="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  rm -f "$dest"
  if ! ln "$src" "$dest" 2>/dev/null; then
    cp -f "$src" "$dest"
  fi
}

sync_shared_preamble() {
  local course_dir="$1"
  install -m 0644 "$shared_preamble" "$course_dir/common_preamble.tex"
}

rebuild_course_pdf() {
  local course_dir="$1"
  local build_dir="$course_dir/build"
  mkdir -p "$build_dir"

  for _ in 1 2; do
    (
      cd "$course_dir"
      pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
        -output-directory="$build_dir" course.tex
    ) >/dev/null
  done

  if [[ ! -f "$build_dir/course.pdf" ]]; then
    echo "Missing rebuilt PDF: $build_dir/course.pdf" >&2
    exit 1
  fi

  cp -f "$build_dir/course.pdf" "$course_dir/course.pdf"
}

publish_course() {
  local source_slug="$1"
  local publish_slug="$2"

  if [[ -n "$course_filter" && "$course_filter" != "$publish_slug" ]]; then
    return 0
  fi

  local course_dir="$repo_root/generated_course_notes/lazylearn/$source_slug"
  local publish_dir="$repo_root/$publish_slug"
  local source_pdf="$course_dir/course.pdf"
  local publish_pdf="$publish_dir/$publish_slug.pdf"
  local all_notes_pdf="$repo_root/all_notes/$publish_slug.pdf"

  sync_shared_preamble "$course_dir"
  rebuild_course_pdf "$course_dir"

  hardlink_or_copy "$source_pdf" "$publish_pdf"
  hardlink_or_copy "$source_pdf" "$all_notes_pdf"

  if [[ "$publish_slug" == "how-to-speak-and-write" ]]; then
    for translated in \
      "$publish_dir/$publish_slug-zh.pdf" \
      "$publish_dir/$publish_slug-jp.pdf"; do
      if [[ -f "$translated" ]]; then
        hardlink_or_copy "$translated" "$repo_root/all_notes/$(basename "$translated")"
      fi
    done
  fi

  printf 'Published full-size PDF: %s -> %s and %s\n' "$source_slug" "$publish_pdf" "$all_notes_pdf"
}

mkdir -p "$repo_root/all_notes"

publish_course "how-you-speak-and-write" "how-to-speak-and-write"
publish_course "justice-with-michael-sandel" "justice-with-michael-sandel"

printf 'Synced LazyLearn full-size PDFs into %s/all_notes\n' "$repo_root"
