#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/fix_lazylearn_pocket_overflow.sh [options]

Run the shared Video2Book pocket overflow fixer for the two LazyLearn books,
then export final 1.0x and 1.2x PDFs into all_notes/, publish hardlinks into
the root course folders, and sync copies to Nutstore.

Options:
  --course <relpath>        Restrict to one course under generated_course_notes/lazylearn
  --variant <name>          all (default) | normal | onepointtwo
  --model <name>            Codex model for the fixer (default: gpt-5.4)
  --reasoning <level>       low|medium|high|xhigh (default: high)
  --max-iterations <n>      Maximum edit passes per course/variant (default: 4)
  --repo-root <path>        Repo root (default: parent of this script)
  --nutstore-root <path>    Nutstore LazyingArtBooks/lazylearn root
  --skip-export             Only run the fixer, skip final export/publish
  -h, --help                Show this help
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
course_filter=""
variant="all"
model="${NOTE_MODEL:-gpt-5.4}"
reasoning="${NOTE_REASONING:-high}"
max_iterations=4
skip_export=0
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
    --model)
      model="${2:-}"
      shift 2
      ;;
    --reasoning)
      reasoning="${2:-}"
      shift 2
      ;;
    --max-iterations)
      max_iterations="${2:-4}"
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
    --skip-export)
      skip_export=1
      shift
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

courses=()
if [[ -n "$course_filter" ]]; then
  courses+=("$course_filter")
else
  courses+=("lazylearn/how-you-speak-and-write")
  courses+=("lazylearn/justice-with-michael-sandel")
fi

session_dir="$repo_root/.lecture-notes-work/codex_sessions"
mkdir -p "$session_dir"
session_file="$session_dir/lazylearn-pocket-fixer.session_id"
session_doc="$session_dir/lazylearn-pocket-fixer.session.md"

run_fix() {
  local course="$1"
  local font_mode="$2"

  bash "$repo_root/Video2Book/scripts/fix_course_pocket_overfulls.sh" \
    --host-root "$repo_root" \
    --source-dir "$repo_root/generated_course_notes" \
    --course "$course" \
    --font-mode "$font_mode" \
    --model "$model" \
    --reasoning "$reasoning" \
    --max-iterations "$max_iterations" \
    --session-file "$session_file" \
    --session-doc "$session_doc" \
    --skip-commit
}

for course in "${courses[@]}"; do
  if [[ "$variant" == "all" || "$variant" == "normal" ]]; then
    run_fix "$course" "normal"
  fi
  if [[ "$variant" == "all" || "$variant" == "onepointtwo" ]]; then
    run_fix "$course" "onepointtwo"
  fi
done

if [[ "$skip_export" -ne 1 ]]; then
  export_args=(--repo-root "$repo_root" --nutstore-root "$nutstore_root" --variant "$variant")
  if [[ -n "$course_filter" ]]; then
    export_args+=(--course "$course_filter")
  fi
  bash "$repo_root/scripts/export_lazylearn_pocket_pdfs.sh" "${export_args[@]}"
fi

printf 'Completed LazyLearn pocket overflow fixing for variant=%s\n' "$variant"
