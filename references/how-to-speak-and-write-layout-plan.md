# How to Speak and Write Layout

This note records the live structure now used by the `How to Speak and Write`
book inside this repository.

## Source Of Truth

The canonical editable course-note tree is:

- [generated_course_notes/lazylearn/how-you-speak-and-write](/home/lachlan/ProjectsLFS/LazyLearn/generated_course_notes/lazylearn/how-you-speak-and-write)

That tree keeps:

- per-lecture chapter `content.tex`
- shared TeX preamble
- figures
- course-order note compilation

## Curated Book Source

The full multilingual source package for the curated book lives inside that
course tree:

- [generated_course_notes/lazylearn/how-you-speak-and-write/how-to-speak-and-write](/home/lachlan/ProjectsLFS/LazyLearn/generated_course_notes/lazylearn/how-you-speak-and-write/how-to-speak-and-write)

That nested package keeps:

- the reordered English wrapper
- cover art
- cover-generation trace
- Traditional Chinese source edition
- Japanese source edition
- translation manifests
- translation work logs and prompts

## Publication Shelf

The root-level publication folder is now intentionally slim:

- [how-to-speak-and-write](/home/lachlan/ProjectsLFS/LazyLearn/how-to-speak-and-write)

It keeps only the final PDFs:

- [how-to-speak-and-write/how-to-speak-and-write.pdf](/home/lachlan/ProjectsLFS/LazyLearn/how-to-speak-and-write/how-to-speak-and-write.pdf)
- [how-to-speak-and-write/how-to-speak-and-write-zh.pdf](/home/lachlan/ProjectsLFS/LazyLearn/how-to-speak-and-write/how-to-speak-and-write-zh.pdf)
- [how-to-speak-and-write/how-to-speak-and-write-jp.pdf](/home/lachlan/ProjectsLFS/LazyLearn/how-to-speak-and-write/how-to-speak-and-write-jp.pdf)

## Rule Of Thumb

- edit under `generated_course_notes/...`
- publish from `how-to-speak-and-write/`
