# Video2Book Integration

`Video2Book` can be used from another repository as a Git submodule. In this workspace, the intended layout is to mount it at `Video2Book/` from the host repo root and run its helper scripts from there.

## Repository

- HTTPS: `https://github.com/lachlanchen/Video2Book`
- SSH: `git@github.com:lachlanchen/Video2Book.git`

## Add as a submodule

```bash
git submodule add git@github.com:lachlanchen/Video2Book.git Video2Book
git submodule update --init --recursive
```

## Expected host layout

The host repository should provide these directories at its root:

```text
<host-repo>/
|- Video2Book/
|- subtitles/
|- markdown/
|- generated_course_notes/
`- .lecture-notes-work/
```

`Video2Book` itself lives inside the host repo, but the generated outputs are written into the host repo root rather than into the submodule.

## Pipeline overview

There are three main stages:

- `playlist2videos`: downloads the playlist into the external media workspace.
- `videos2subtitles`: writes `subtitles/` and `markdown/` in the host repo.
- `subtitles2notes`: writes `generated_course_notes/` and `.lecture-notes-work/` in the host repo.

## Typical commands

Run these from the host repo root:

```bash
./Video2Book/scripts/download_susskind_playlist.sh
./Video2Book/scripts/start_transcription_tmux.sh
./Video2Book/scripts/start_transcription_monitor_tmux.sh
./Video2Book/scripts/start_course_notes_tmux.sh
./Video2Book/scripts/start_course_notes_monitor_tmux.sh
```

Suggested sequence:

1. Download the playlist.
2. Start the transcription tmux session.
3. Start the transcription monitor.
4. Start the course-notes tmux session.
5. Start the course-notes monitor.

## Requirements

- `tmux`
- `ffmpeg`
- `pdflatex`
- `pdfunite`
- `pdftotext`
- Codex CLI for the notes pipeline
- A working Whisper conda environment for transcription
- `whisper_with_lang_detect` if available

If `whisper_with_lang_detect` is not available, the transcription step can fall back to direct Whisper in some cases.

## Notes

- The scripts are designed to be invoked from the host repo root, not from inside the `Video2Book` submodule.
- The media download stage uses an external media workspace.
- The note-generation stage depends on the local Codex CLI setup being functional before launching the tmux workers.
