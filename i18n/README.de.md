[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# LazyPhysics and Chemistry

[![Site](https://img.shields.io/badge/site-learn.lazying.art-0a7ea4)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-active%20learning-22c55e)
![Repo Type](https://img.shields.io/badge/repo-mixed--format-6b7280)
![Docs](https://img.shields.io/badge/docs-static%20microsite-0ea5e9)

LazyPhysics and Chemistry ist die Code- und Notebook-Hälfte von **LazyLearn**: ein bewusst langsames, praxisorientiertes Lernprotokoll für Physik und Chemie. Laufende Notizen, Fortschritte und TODOs werden unter [learn.lazying.art](https://learn.lazying.art) veröffentlicht (aus `docs/` in diesem Repository bereitgestellt), während ausführbare Artefakte hier bleiben, damit Experimente immer ein reproduzierbares Zuhause haben.

## Überblick 🧭

### LazyLearn

- **Homebase:** [learn.lazying.art](https://learn.lazying.art) - die öffentliche Seite mit Wochenschwerpunkten, Backlog und Highlights.
- **Single Source of Truth:** Alles, worauf die Website verweist, liegt in `examples/`, `comp_physics/`, `comp_physics_python/`, `multiwfn/` oder `figures/`.
- **Update-Ablauf:** Zuerst Code/Notebooks veröffentlichen, bei Bedarf Plots neu generieren und danach einen Eintrag in `docs/` ergänzen, damit die Website den neuesten Stand zeigt.

Dieses Repository ist absichtlich ein Mischformat und keine einzelne paketierte App. Es kombiniert ausführbare Skripte, Notebooks, Referenzen und eine statische Doku-Website in einem versionierten Workspace.

## Features ✨

- Quanten-Beispielskripte (QAOA + VQE), die auf handelsüblichen Laptops laufen.
- Notebooks zur Computational Physics und Hilfslöser (z. B. Numerov-basierte Workflows).
- Kapitelweise Python-Portierungen von Lehrbuchprogrammen zur Computational Physics.
- Multiwfn-Quellcode-/Handbuch-Bundle als lokale Referenz für Post-Processing in der Quantenchemie.
- Versionierte, generierte Abbildungen für Berichte/Folien (`figures/`).
- Integrierter mehrsprachiger README-Satz unter `i18n/`.
- Statische Microsite in `docs/` (Custom Domain: `learn.lazying.art`).

## Projektstruktur 🗂️

### Was hier liegt

| Path | Purpose |
| --- | --- |
| `examples/` | Fokussierte Python-Skripte (QAOA + VQE), die mit Qiskit oder PennyLane laufen. |
| `comp_physics/` | Computational-Physics-Notebooks, Hilfsskripte wie `numerov.py` sowie unterstützende Daten/Abbildungen. |
| `comp_physics_python/` | Python-Portierungen von Jos Thijssens *Computational Physics*, nach Kapiteln organisiert (siehe [comp_physics_python/README.md](../comp_physics_python/README.md)). |
| `multiwfn/` | Multiwfn-3.8-Developer-Source-Bundle plus Handbücher als lokale Referenz. |
| `figures/` | Statische PNG/SVG-Ausgaben für Berichte/Folien und README. |
| `figs/` | Logo- und Banner-Assets. |
| `docs/` | Inhalte der LazyLearn-Microsite (bereitgestellt über GitHub Pages oder jeden statischen Host). |
| `i18n/` | Lokalisierte README-Dateien. |

Beispielhafter Aufbau:

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
> Mehrere Einträge auf oberster Ebene sind Symlinks auf Verzeichnisse außerhalb dieses Repositories. Änderungen unter diesen Pfaden wirken sich auf externe Ziele aus.

## Voraussetzungen 🧰

| Requirement | Notes |
| --- | --- |
| Python 3.x | Erforderlich für Skripte im Root und die meisten Notebook-Workflows. |
| `pip` (oder Conda) | Paket-/Umgebungsverwaltung. |
| Jupyter Lab/Notebook (optional) | Erforderlich für Notebook-Workflows. |
| Gaussian 16 + GaussView (optional) | Erforderlich für Gaussian-Workflows. |

## Installation ⚙️

### Minimales Python-Setup (Root-Beispiele)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

Jupyter-Notebooks in `comp_physics/` verwenden dieselbe Umgebung. Start mit:

```bash
jupyter lab
# or
jupyter notebook
```

### Optionale Kapitel-Portierungs-Abhängigkeiten (`comp_physics_python/`)

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## Nutzung 🚀

### Beispiel-Workflows

- **QAOA mit Qiskit**

```bash
python examples/qaoa_qiskit_maxcut.py
```

Keine Aer-Abhängigkeit; nutzt ein reines Statevector-Backend.

- **QAOA mit PennyLane**

```bash
python examples/qaoa_pennylane_maxcut.py
```

Verwendet `default.qubit`.

- **VQE für H2**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

Reproduziert `figures/pennylane_h2_vqe_convergence.png`.

Alle Skripte protokollieren Zwischenmetriken, sodass du Plots wiederverwenden oder auf neue Moleküle/Graphen erweitern kannst.

## Computational-Physics-Notebooks 📓

Das Verzeichnis `comp_physics/` spiegelt laufende Arbeitsnotizen wider:

- `comp_physics_textbook_code/` - wiederverwendbare Routinen, die aus Notebooks extrahiert wurden.
- Eigenständige Notebooks wie `chapter1.ipynb`, `chapter2.ipynb`, `numerov.ipynb` und `numpy_1ddft.ipynb`.
- Themenordner (`bosonscattering/`, `lensless/`, `lightscattering/` usw.) mit Daten und Hilfsdateien pro Experiment.

Wenn zusätzliche Abhängigkeiten benötigt werden, dokumentiere sie in `comp_physics/environments.yaml`.

## Lehrbuch-Codeübersetzungen 📚

`comp_physics_python/` ist eine wachsende Python-Übersetzung der klassischen Fortran-Programme aus *Computational Physics*. Beispielhafte Kapitelzuordnung:

- `ch4/`: Hartree-Fock-Beispiele.
- `ch8/`: Molekulardynamik-Solver.
- `ch10/`: Monte-Carlo-Sampler.

Siehe [comp_physics_python/README.md](../comp_physics_python/README.md) für die vollständige Kapitelabdeckung und CLI-Befehle.

## Multiwfn-Referenzen 🔬

`multiwfn/` enthält `Multiwfn_3.8_dev_src_Linux` sowie das PDF-Handbuch und die Quick-Start-Anleitung. Es werden keine kompilierten Binärdateien versioniert.

## Abbildungen 🖼️

Generierte PNG/SVG-Assets liegen in `figures/`, damit Outputs gemeinsam mit den erzeugenden Skripten/Notebooks versioniert sind.

## Konfiguration 🛠️

### Python und Notebooks

- Root-Skripte setzen das oben gezeigte venv voraus.
- Details zu Notebook-Umgebungen sind über Projekt-Dokumente verteilt; derzeit gibt es im Repository-Root keine einzelne Lockfile.

### Gaussian-Runner (symlinked path)

`Gaussian/run_gaussian.sh` unterstützt:

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

Verhalten:

- Schreibt `<basename>.log` neben die Eingabedatei.
- Verwendet `GAUSS_SCRDIR`, falls gesetzt, sonst standardmäßig `~/gaussian/scr`.
- Erkennt `%chk=...` in der Eingabe; existiert der Checkpoint, öffnet GaussView `.chk`, sonst `.log`.
- Falls verfügbar, wird `~/gaussian/gv/gview_safe.sh`, danach `gview.sh` bevorzugt.

Empfohlener GaussView-Wrapper:

```bash
#!/usr/bin/env bash
set -euo pipefail
GV_SH="$HOME/gaussian/gv/gview.sh"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec "$GV_SH" "$@"
```

## Entwicklungshinweise 🧪

### Hinweise zur Versionskontrolle

- Umfangreiche Pfade werden über `.gitignore` ignoriert, darunter `books/`, externe Symlink-Ziele (`Gaussian`, `ComputationalPhysics`, `leonardsusskind`, `the_theoretical_minimum`) und lokale Artefakte wie `*.chk`.
- Halte Beiträge auf verfolgte Ordner fokussiert, damit Clone-/Update-Workflows schlank bleiben.
- Für Website-Updates: `docs/` bearbeiten, lokal vorschauen, dann pushen.

Lokale Doku-Vorschau:

```bash
python -m http.server --directory docs
```

`docs/CNAME` ist für `learn.lazying.art` konfiguriert.

## Fehlerbehebung 🩺

- Gaussian-Erfolgskriterium: `Normal termination of Gaussian` nahe dem Ende der `.log`.
- Wenn GaussView unter Wayland/Remote-Sitzungen nicht startet, `gview_safe.sh` verwenden und `--gview` explizit übergeben.
- Bei Gaussian-Scratch-Fehlern freien Speicherplatz und Berechtigungen in `GAUSS_SCRDIR` prüfen.
- Wenn Notebook-Abhängigkeiten auseinanderlaufen, Unterprojekt-READMEs als Source of Truth behandeln und fehlende Pakete vor dem Teilen in Umgebungsdateien erfassen.
- `comp_physics/environments.yaml` scheint im aktuellen Repo-Stand ein Platzhalter zu sein; bis zur Korrektur auf explizite Installationsbefehle verlassen.

## Roadmap 🛣️

- Kapitelabdeckung in `comp_physics_python/` weiter ausbauen (Transfermatrizen, DMC/PIMC, FEM und mehr).
- Ausgabe-/Plot-Konventionen über Skripte und Notebooks hinweg harmonisieren.
- Leichte, wiederholbare Validierungschecks für zentrale Beispiele ergänzen.
- `docs/` und mehrsprachige READMEs mit neuen Experimenten synchron halten.

## Beitrag 🤝

Issues und Pull Requests sind willkommen, besonders für:

- Prüfungen numerischer Korrektheit und Verbesserungen der Reproduzierbarkeit.
- Bessere Umgebungsspezifikationen für Notebooks/Skripte.
- Zusätzliche Lehrbuch-Kapitelportierungen und CLI-Verbesserungen.
- Klarere Dokumentation über die Sprachen in `i18n/` hinweg.

Vor dem Einreichen größerer Inhalts-Updates generierte Abbildungen in `figures/` behalten und sicherstellen, dass Befehle vom Repository-Root aus lauffähig sind, sofern nicht anders dokumentiert.

## LazyLearn unterstützen ❤️

Mit Unterstützung für LazyLearn bleiben Experimente, Dokumentation und offene Tooling-Arbeit im Fluss:

- Hosting/Inference/Storage für öffentliche Demos und Notebooks finanzieren.
- Fokussierte Hack-Weeks zu EchoMind, LazyEdit und Quanten-/Physik-Utilities in diesem Repo ermöglichen.
- Optik- + Wearables-Prototypen (IdeasGlass, LightMind) entwickeln, die in zukünftige Kapitel einfließen.
- Kostenlose Deployments für Studierende, Community-Labs und Creator sponsern.

### Spenden

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
- Deine Unterstützung trägt meine Forschung, Entwicklung und den laufenden Betrieb, damit ich weiterhin mehr offene Projekte und Verbesserungen teilen kann.

## Lizenz 📄

Im Root dieses Repositories ist derzeit keine `LICENSE`-Datei vorhanden. Bis eine Lizenz ergänzt wird, sollten Nutzungs- und Weitergaberechte als nicht festgelegt betrachtet werden; vor der Wiederverwendung wesentlicher Inhalte bitte Rücksprache mit dem Maintainer halten.
