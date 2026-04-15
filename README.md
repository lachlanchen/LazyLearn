[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)

![LazyingArt logo](https://lazying.art/logos/logo.png)

# LazyLearn

Knowledge + skill, unrushed.

[![Site](https://img.shields.io/badge/website-learn.lazying.art-0a7ea4?style=for-the-badge&logo=githubpages&logoColor=white)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Learning-16a34a?style=for-the-badge&logo=target&logoColor=white)
![Repo Type](https://img.shields.io/badge/Repo-Type-Mixed%20Format-6b7280?style=for-the-badge)
![Docs](https://img.shields.io/badge/Docs-Static%20Microsite-0ea5e9?style=for-the-badge&logo=markdown&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-11-0f766e?style=for-the-badge&logo=googletranslate&logoColor=white)

## Featured Book

**How to Speak and Write** is now the most prominent published book from this repo.

![How to Speak and Write cover](docs/how-to-speak-and-write-cover.png)

- Read the English local edition: [how-to-speak-and-write/how-to-speak-and-write.pdf](how-to-speak-and-write/how-to-speak-and-write.pdf)
- Read the Traditional Chinese local edition: [how-to-speak-and-write/how-to-speak-and-write-zh.pdf](how-to-speak-and-write/how-to-speak-and-write-zh.pdf)
- Read the Japanese local edition: [how-to-speak-and-write/how-to-speak-and-write-jp.pdf](how-to-speak-and-write/how-to-speak-and-write-jp.pdf)
- Read the website edition: [learn.lazying.art/how-to-speak-and-write.pdf](https://learn.lazying.art/how-to-speak-and-write.pdf)

## Dedicated Physics Book Repo

There is also a separate repository dedicated to the Leonard Susskind lecture-note book collection.

[![Classical Mechanics Stanford partial cover](https://github.com/lachlanchen/leonardsusskind/raw/main/figs/readme-covers/classical_mechanics_stanford_partial.png)](https://github.com/lachlanchen/leonardsusskind)

- Browse the repo: [lachlanchen/leonardsusskind](https://github.com/lachlanchen/leonardsusskind)
- Use that repo for the larger physics catalog, including classical mechanics, advanced quantum mechanics, particle physics, entanglement, and string theory

## Book Layout

The book now has a strict split between source and publication:

- canonical editable lecture-note source lives in [generated_course_notes/lazylearn/how-you-speak-and-write](/home/lachlan/ProjectsLFS/LazyLearn/generated_course_notes/lazylearn/how-you-speak-and-write)
- the curated multilingual book source lives in [generated_course_notes/lazylearn/how-you-speak-and-write/how-to-speak-and-write](/home/lachlan/ProjectsLFS/LazyLearn/generated_course_notes/lazylearn/how-you-speak-and-write/how-to-speak-and-write)
- the root-level publication shelf lives in [how-to-speak-and-write](/home/lachlan/ProjectsLFS/LazyLearn/how-to-speak-and-write)

Current behavior:

- `generated_course_notes/...` keeps the editable source, cover assets, chapter structure, and translation source trees
- `how-to-speak-and-write/` keeps only the published PDFs:
  - English
  - Traditional Chinese
  - Japanese

Layout note:

- [references/how-to-speak-and-write-layout-plan.md](references/how-to-speak-and-write-layout-plan.md)

## 📌 At a glance

| Focus                | What this repo does                                    |
| -------------------- | ------------------------------------------------------ |
| Workflow type        | Reproducible physics + chemistry learning workspace    |
| Deliverables         | Scripts, notebooks, generated figures, and static docs |
| Collaboration model  | Root experiments + public site publishing              |
| Translation coverage | README mirror files in `i18n/`                         |

This repository is the code + notebook half of **LazyLearn**: an intentionally slow, practical learning log for knowledge and skill building. The living notes, wins, and TODOs are published at [learn.lazying.art](https://learn.lazying.art) (served from `docs/` in this repo), while runnable artifacts stay here so experiments always have a reproducible home.

## Overview 🧭

### LazyLearn

- **Home base:** [learn.lazying.art](https://learn.lazying.art) - the public-facing site with weekly focuses, backlog, and highlights.
- **Source of truth:** everything the site links to lives in `examples/`, `comp_physics/`, `comp_physics_python/`, `multiwfn/`, or `figures/`.
- **Update flow:** ship code/notebooks first, regenerate plots if needed, and then add an entry to `docs/` so the site reflects the latest work.

This repository is intentionally mixed-format, not a single packaged app. It combines executable scripts, notebooks, references, and a static docs site under one versioned workspace.

## Features ✨

- Quantum example scripts (QAOA + VQE) that run on commodity laptops.
- Computational physics notebooks and helper solvers (e.g., Numerov-based workflows).
- Chapter-by-chapter Python ports of textbook computational physics programs.
- Multiwfn source/manual bundle for local quantum chemistry post-processing reference.
- Versioned generated figures for reports/slides (`figures/`).
- Built-in multilingual README set under `i18n/`.
- Static microsite in `docs/` (custom domain: `learn.lazying.art`).

## Project structure 🗂️

### What lives here

| Path                   | Purpose                                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `examples/`            | Focused Python scripts (QAOA + VQE) that run with Qiskit or PennyLane.                                                                             |
| `comp_physics/`        | Computational physics notebooks, helper scripts like `numerov.py`, and supporting data/figures.                                                    |
| `comp_physics_python/` | Python ports of Jos Thijssen's _Computational Physics_, organized by chapter (see [comp_physics_python/README.md](comp_physics_python/README.md)). |
| `multiwfn/`            | Multiwfn 3.8 developer source bundle plus manuals for local reference.                                                                             |
| `figures/`             | Static PNG/SVG outputs used in reports/slides and README.                                                                                          |
| `figs/`                | Logo and banner assets.                                                                                                                            |
| `docs/`                | LazyLearn microsite content (served by GitHub Pages or any static host).                                                                           |
| `i18n/`                | Localized README files.                                                                                                                            |
| `generated_course_notes/` | Canonical generated lecture-note source trees, chapter TeX, figures, compiled course artifacts, and book-source wrappers.                      |
| `how-to-speak-and-write/` | Publish-only shelf for the featured English, Traditional Chinese, and Japanese PDFs.                                                           |

Representative layout:

```text
LazyLearn/
|- README.md
|- docs/
|- i18n/
|- examples/
|- comp_physics/
|- comp_physics_python/
|- multiwfn/
|- figures/
|- figs/
|- Gaussian -> ../Gaussian/ (symlink)
|- ComputationalPhysics -> ../ComputationalPhysics/ (symlink)
|- leonardsusskind -> ../leonardsusskind/ (symlink)
`- the_theoretical_minimum -> ../the_theoretical_minimum/ (symlink)
```

> [!IMPORTANT]
> Several top-level entries are symlinks to directories outside this repository. Editing under those paths affects external targets.

## Prerequisites 🧰

| Requirement                        | Notes                                             |
| ---------------------------------- | ------------------------------------------------- |
| Python 3.x                         | Required for root scripts and most notebook work. |
| `pip` (or Conda)                   | Package/environment management.                   |
| Jupyter Lab/Notebook (optional)    | Needed for notebook workflows.                    |
| Gaussian 16 + GaussView (optional) | Needed for Gaussian workflows.                    |

## Installation ⚙️

### Minimal Python setup (root examples)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

### ✅ Quick setup checklist

| Step | Command                                         | Goal                                |
| ---- | ----------------------------------------------- | ----------------------------------- |
| 1    | `python -m venv .venv`                          | Create an isolated environment      |
| 2    | `source .venv/bin/activate` (or OS-equivalent)  | Avoid dependency conflicts          |
| 3    | `pip install --upgrade pip`                     | Ensure current package tooling      |
| 4    | `pip install qiskit pennylane numpy matplotlib` | Install the core experimental stack |
| 5    | Run one script in `examples/`                   | Validate installation end-to-end    |

Jupyter notebooks inside `comp_physics/` use the same environment. Launch with:

```bash
jupyter lab
# or
jupyter notebook
```

### Optional chapter-port dependencies (`comp_physics_python/`)

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## Usage 🚀

### Example workflows

- **QAOA with Qiskit**

```bash
python examples/qaoa_qiskit_maxcut.py
```

No Aer dependency; uses a pure statevector backend.

- **QAOA with PennyLane**

```bash
python examples/qaoa_pennylane_maxcut.py
```

Uses `default.qubit`.

- **VQE for H2**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

Reproduces `figures/pennylane_h2_vqe_convergence.png`.

All scripts log intermediate metrics so you can reuse plots or extend to new molecules/graphs.

## Computational physics notebooks 📓

The `comp_physics/` directory mirrors working notes:

- `comp_physics_textbook_code/` - reusable routines extracted from notebooks.
- Standalone notebooks such as `chapter1.ipynb`, `chapter2.ipynb`, `numerov.ipynb`, and `numpy_1ddft.ipynb`.
- Topic folders (`bosonscattering/`, `lensless/`, `lightscattering/`, etc.) with data and helpers per experiment.

If extra dependencies are needed, record them in `comp_physics/environments.yaml`.

## Textbook code translations 📚

`comp_physics_python/` is a growing Python translation of the classic Fortran programs from _Computational Physics_. Example chapter mapping:

- `ch4/`: Hartree-Fock examples.
- `ch8/`: molecular dynamics solvers.
- `ch10/`: Monte Carlo samplers.

Refer to [comp_physics_python/README.md](comp_physics_python/README.md) for full chapter coverage and CLI commands.

## Multiwfn references 🔬

`multiwfn/` keeps `Multiwfn_3.8_dev_src_Linux` plus the PDF manual and quick-start guide. No compiled binaries are committed.

## Figures 🖼️

Generated PNG/SVG assets live in `figures/` so outputs are versioned alongside producing scripts/notebooks.

## Configuration 🛠️

### Python and notebooks

- Root scripts assume the venv shown above.
- Notebook environment details are distributed across project docs; no single lockfile currently exists at repo root.

### Gaussian runner (symlinked path)

`Gaussian/run_gaussian.sh` supports:

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

Behavior:

- Writes `<basename>.log` next to input.
- Uses `GAUSS_SCRDIR` if set, otherwise defaults to `~/gaussian/scr`.
- Detects `%chk=...` in input; if checkpoint exists, GaussView opens `.chk`, otherwise `.log`.
- If available, prefers `~/gaussian/gv/gview_safe.sh`, then `gview.sh`.

Recommended GaussView wrapper:

```bash
#!/usr/bin/env bash
set -euo pipefail
GV_SH="$HOME/gaussian/gv/gview.sh"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec "$GV_SH" "$@"
```

## Development notes 🧪

### 🎬 Navigation map

Use this as a launchpad for daily work:

| Area                      | Start here             |
| ------------------------- | ---------------------- |
| Quantum demos             | `examples/`            |
| Physics notebooks         | `comp_physics/`        |
| Textbook translations     | `comp_physics_python/` |
| Quantum chemistry tools   | `multiwfn/`            |
| Published outputs         | `docs/`                |
| Figures and illustrations | `figures/`, `figs/`    |

### Version control notes

- Heavy paths are ignored via `.gitignore`, including `books/`, external symlink targets (`Gaussian`, `ComputationalPhysics`, `leonardsusskind`, `the_theoretical_minimum`), and local artifacts such as `*.chk`.
- Keep contributions focused on tracked folders for lightweight clone/update workflows.
- For website updates: edit `docs/`, preview locally, then push.

Local docs preview:

```bash
python -m http.server --directory docs
```

`docs/CNAME` is configured for `learn.lazying.art`.

## Troubleshooting 🩺

- Gaussian success criterion: `Normal termination of Gaussian` near the end of the `.log`.
- If GaussView fails under Wayland/remote sessions, use `gview_safe.sh` and pass `--gview` explicitly.
- If Gaussian scratch errors occur, verify free disk and permissions in `GAUSS_SCRDIR`.
- If notebook dependencies drift, treat subproject READMEs as source-of-truth and capture missing packages in environment files before sharing.
- `comp_physics/environments.yaml` appears to be a placeholder in the current repo state; rely on explicit install commands until it is corrected.

## Roadmap 🛣️

- Continue expanding `comp_physics_python/` chapter coverage (transfer matrices, DMC/PIMC, FEM, and beyond).
- Harmonize output/plot conventions across scripts and notebooks.
- Add lightweight, repeatable validation checks for key examples.
- Keep `docs/` and multilingual READMEs aligned with new experiments.

## Contribution 🤝

Issues and pull requests are welcome, especially for:

- Numerical correctness checks and reproducibility improvements.
- Better environment specifications for notebooks/scripts.
- Additional textbook chapter ports and CLI refinements.
- Documentation clarity across languages in `i18n/`.

Before submitting major content updates, keep generated figures in `figures/` and ensure commands are runnable from repository root unless otherwise documented.

## ❤️ Support

| Donate                                                                                                                                                                                                                                                                                                                                                     | PayPal                                                                                                                                                                                                                                                                                                                                                          | Stripe                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License 📄

No root `LICENSE` file is currently present in this repository. Until a license is added, treat usage/redistribution rights as unspecified and request clarification from the maintainer before reusing substantial content.
