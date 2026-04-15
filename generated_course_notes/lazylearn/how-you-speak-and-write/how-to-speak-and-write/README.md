# How to Speak and Write

This folder is the full source package for the curated book
**How to Speak and Write**.

## What Lives Here

- reordered English book source
- cover art and cover-generation trace
- Traditional Chinese source edition
- Japanese source edition
- translation manifests and translation work logs
- compiled book PDFs produced from this source package

## Source Structure

- [how-to-speak-and-write.tex](/home/lachlan/ProjectsLFS/LazyLearn/generated_course_notes/lazylearn/how-you-speak-and-write/how-to-speak-and-write/how-to-speak-and-write.tex)
  is the reordered English wrapper
- [assets/](/home/lachlan/ProjectsLFS/LazyLearn/generated_course_notes/lazylearn/how-you-speak-and-write/how-to-speak-and-write/assets)
  holds the cover art used by the published edition
- [zh/](/home/lachlan/ProjectsLFS/LazyLearn/generated_course_notes/lazylearn/how-you-speak-and-write/how-to-speak-and-write/zh)
  holds the Traditional Chinese source edition
- [jp/](/home/lachlan/ProjectsLFS/LazyLearn/generated_course_notes/lazylearn/how-you-speak-and-write/how-to-speak-and-write/jp)
  holds the Japanese source edition
- [.translation-work/](/home/lachlan/ProjectsLFS/LazyLearn/generated_course_notes/lazylearn/how-you-speak-and-write/how-to-speak-and-write/.translation-work)
  keeps prompt traces, shared Codex session notes, and translation logs

## Relationship To The Course Tree

This source package sits inside the canonical course-note tree:

- [../](/home/lachlan/ProjectsLFS/LazyLearn/generated_course_notes/lazylearn/how-you-speak-and-write)

That parent directory still holds the reusable lecture-note source:

- chapter `content.tex` files under `chapters/`
- shared TeX preamble
- figures and generated assets
- course-level note outputs

The book wrapper here is the curated multilingual edition built on top of those
chapter sources.

## Published Outputs

This source package compiles to:

- [how-to-speak-and-write.pdf](/home/lachlan/ProjectsLFS/LazyLearn/generated_course_notes/lazylearn/how-you-speak-and-write/how-to-speak-and-write/how-to-speak-and-write.pdf)
- [zh/how-to-speak-and-write.pdf](/home/lachlan/ProjectsLFS/LazyLearn/generated_course_notes/lazylearn/how-you-speak-and-write/how-to-speak-and-write/zh/how-to-speak-and-write.pdf)
- [jp/how-to-speak-and-write.pdf](/home/lachlan/ProjectsLFS/LazyLearn/generated_course_notes/lazylearn/how-you-speak-and-write/how-to-speak-and-write/jp/how-to-speak-and-write.pdf)

The root-level publication shelf is separate and keeps only the final PDFs for
easy browsing.
