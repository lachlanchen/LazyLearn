[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)

<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>


# LazyPhysics and Chemistry

[![Site](https://img.shields.io/badge/site-learn.lazying.art-0a7ea4)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-active%20learning-22c55e)
![Repo Type](https://img.shields.io/badge/repo-mixed--format-6b7280)
![Docs](https://img.shields.io/badge/docs-static%20microsite-0ea5e9)

LazyPhysics and Chemistry is the code + notebook half of **LazyLearn**: an intentionally slow, practical learning log for physics and chemistry. The living notes, wins, and TODOs are published at [learn.lazying.art](https://learn.lazying.art) (served from `docs/` in this repo), while runnable artifacts stay here so experiments always have a reproducible home.

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

| Path | Purpose |
| --- | --- |
| `examples/` | Focused Python scripts (QAOA + VQE) that run with Qiskit or PennyLane. |
| `comp_physics/` | Computational physics notebooks, helper scripts like `numerov.py`, and supporting data/figures. |
| `comp_physics_python/` | Python ports of Jos Thijssen's *Computational Physics*, organized by chapter (see [comp_physics_python/README.md](comp_physics_python/README.md)). |
| `multiwfn/` | Multiwfn 3.8 developer source bundle plus manuals for local reference. |
| `figures/` | Static PNG/SVG outputs used in reports/slides and README. |
| `figs/` | Logo and banner assets. |
| `docs/` | LazyLearn microsite content (served by GitHub Pages or any static host). |
| `i18n/` | Localized README files. |

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

| Requirement | Notes |
| --- | --- |
| Python 3.x | Required for root scripts and most notebook work. |
| `pip` (or Conda) | Package/environment management. |
| Jupyter Lab/Notebook (optional) | Needed for notebook workflows. |
| Gaussian 16 + GaussView (optional) | Needed for Gaussian workflows. |

## Installation ⚙️

### Minimal Python setup (root examples)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

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

`comp_physics_python/` is a growing Python translation of the classic Fortran programs from *Computational Physics*. Example chapter mapping:

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

## Support LazyLearn ❤️

Helping LazyLearn keeps experiments, documentation, and open tooling flowing:

- Cover hosting/inference/storage for public demos and notebooks.
- Fund focused hack-weeks on EchoMind, LazyEdit, and quantum/physics utilities here.
- Prototype optics + wearables (IdeasGlass, LightMind) that feed future chapters.
- Sponsor free deployments for students, community labs, and creators.

### Donate

<div align="center">
<table style="margin:0 auto; text-align:center; border-collapse:collapse;">
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://chat.lazying.art/donate">https://chat.lazying.art/donate</a>
    </td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://chat.lazying.art/donate"><img src="figures/donate_button.svg" alt="Donate" height="44"></a>
    </td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://paypal.me/RongzhouChen">
        <img src="https://img.shields.io/badge/PayPal-Donate-003087?logo=paypal&logoColor=white" alt="Donate with PayPal">
      </a>
    </td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400">
        <img src="https://img.shields.io/badge/Stripe-Donate-635bff?logo=stripe&logoColor=white" alt="Donate with Stripe">
      </a>
    </td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><strong>WeChat</strong></td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><strong>Alipay</strong></td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><img alt="WeChat QR" src="figures/donate_wechat.png" width="240"/></td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><img alt="Alipay QR" src="figures/donate_alipay.png" width="240"/></td>
  </tr>
</table>
</div>

**支援 / Donate**

- ご支援は研究・開発と運用の継続に役立ち、より多くのオープンなプロジェクトを皆さんに届ける力になります。
- 你的支持将用于研发与运维，帮助我持续公开分享更多项目与改进。
- Your support sustains my research, development, and ops so I can keep sharing more open projects and improvements.

## License 📄

No root `LICENSE` file is currently present in this repository. Until a license is added, treat usage/redistribution rights as unspecified and request clarification from the maintainer before reusing substantial content.
