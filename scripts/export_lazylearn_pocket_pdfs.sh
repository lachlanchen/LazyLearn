#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/export_lazylearn_pocket_pdfs.sh [options]

Export LazyLearn pocket PDFs into canonical root-level all_notes folders,
then publish hardlinks into the root course folders and sync copies to Nutstore.

Options:
  --course <relpath>      Restrict export to one course path under generated_course_notes/lazylearn
  --variant <name>        all (default) | normal | onepointtwo
  --repo-root <path>      Repo root (default: parent of this script)
  --nutstore-root <path>  Nutstore LazyingArtBooks/lazylearn root
  -h, --help              Show this help
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$repo_root/generated_course_notes/lazylearn"
course_filter=""
variant="all"
nutstore_root="${NUTSTORE_ROOT:-/home/lachlan/Nutstore Files/Projects/LazyingArtBooks/lazylearn}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --course)
      course_filter="${2:-}"
      shift 2
      ;;
    --variant)
      variant="${2:-}"
      shift 2
      ;;
    --repo-root)
      repo_root="${2:-}"
      source_dir="$repo_root/generated_course_notes/lazylearn"
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

case "$variant" in
  all|normal|onepointtwo)
    ;;
  *)
    echo "Unknown variant: $variant" >&2
    exit 1
    ;;
esac

mkdir -p \
  "$repo_root/all_notes/pocket_books" \
  "$repo_root/all_notes/pocket_books_1_2x"

export_variant() {
  local font_mode="$1"
  local output_dir="$2"
  local suffix="$3"

  local cmd=(
    bash "$repo_root/Video2Book/scripts/export_course_pocket_pdfs.sh"
    --host-root "$repo_root"
    --source-dir "$source_dir"
    --output-dir "$output_dir"
    --font-mode "$font_mode"
    --suffix "$suffix"
    --no-rsync
  )

  if [[ -n "$course_filter" ]]; then
    cmd+=(--course "$course_filter")
  fi

  "${cmd[@]}"
}

if [[ "$variant" == "all" || "$variant" == "normal" ]]; then
  export_variant "normal" "$repo_root/all_notes/pocket_books" "pocket"
fi

if [[ "$variant" == "all" || "$variant" == "onepointtwo" ]]; then
  export_variant "onepointtwo" "$repo_root/all_notes/pocket_books_1_2x" "pocket_1_2x"
fi

publish_args=(--repo-root "$repo_root" --nutstore-root "$nutstore_root")
if [[ -n "$course_filter" ]]; then
  case "$course_filter" in
    lazylearn/how-you-speak-and-write)
      publish_args+=(--course "how-to-speak-and-write")
      ;;
    lazylearn/justice-with-michael-sandel)
      publish_args+=(--course "justice-with-michael-sandel")
      ;;
  esac
fi

bash "$repo_root/scripts/publish_lazylearn_pocket_pdfs.sh" "${publish_args[@]}"

printf 'Exported LazyLearn pocket PDFs into %s/all_notes\n' "$repo_root"
