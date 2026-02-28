[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# LazyPhysics and Chemistry

[![Site](https://img.shields.io/badge/website-learn.lazying.art-0a7ea4?style=for-the-badge&logo=githubpages&logoColor=white)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Learning-16a34a?style=for-the-badge&logo=target&logoColor=white)
![Repo Type](https://img.shields.io/badge/Repo-Type-Mixed%20Format-6b7280?style=for-the-badge)
![Docs](https://img.shields.io/badge/Docs-Static%20Microsite-0ea5e9?style=for-the-badge&logo=markdown&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-11-0f766e?style=for-the-badge&logo=googletranslate&logoColor=white)

## 📌 Kurzüberblick

| Fokus | Was dieses Repo macht |
| --- | --- |
| Workflow-Typ | Reproduzierbarer Lernarbeitsbereich für Physik + Chemie |
| Ergebnisse | Skripte, Notebooks, generierte Abbildungen und statische Doku |
| Kollaborationsmodell | Arbeiten im Root + Veröffentlichung auf öffentlicher Seite |
| Übersetzungen | README-Spiegelungen in `i18n/` |

LazyPhysics and Chemistry ist die Code- und Notebook-Hälfte von **LazyLearn**: ein bewusst langsames, praxisnahes Lernjournal für Physik und Chemie. Die laufenden Notizen, Fortschritte und TODOs werden auf [learn.lazying.art](https://learn.lazying.art) veröffentlicht (in diesem Repo aus `docs/` ausgeliefert), während ausführbare Artefakte hier bleiben, damit Experimente immer einen reproduzierbaren Ort haben.

## Überblick 🧭

### LazyLearn

- **Startpunkt:** [learn.lazying.art](https://learn.lazying.art) - die öffentliche Seite mit wöchentlichen Schwerpunkten, Backlog und Highlights.
- **Single Source of Truth:** Alles, worauf die Website verweist, liegt in `examples/`, `comp_physics/`, `comp_physics_python/`, `multiwfn/` oder `figures/`.
- **Update-Prozess:** Erst Code/Notebooks liefern, bei Bedarf Plots neu erzeugen und anschließend einen Beitrag in `docs/` ergänzen, damit die Website den neuesten Stand widerspiegelt.

Dieses Repository ist bewusst ein Mischformat und keine einzelne paketierte App. Es kombiniert ausführbare Skripte, Notebooks, Referenzen und eine statische Dokumentationsseite in einem versionierten Workspace.

## Funktionen ✨

- Quanten-Beispielskripte (QAOA + VQE), die auf Standard-Laptops laufen.
- Notebooks zur Computational Physics und Hilfslöser (z. B. Numerov-basierte Workflows).
- Kapitelweise Python-Portierungen von Lehrbuchprogrammen zur Computational Physics.
- Multiwfn-Source-/Manual-Bundle als lokale Referenz für Post-Processing in der Quantenchemie.
- Versionierte, generierte Abbildungen für Berichte/Folien (`figures/`).
- Integrierter mehrsprachiger README-Satz unter `i18n/`.
- Statische Microsite in `docs/` (eigene Domain: `learn.lazying.art`).

## Projektstruktur 🗂️

### Was hier liegt

| Pfad | Zweck |
| --- | --- |
| `examples/` | Fokussierte Python-Skripte (QAOA + VQE), die mit Qiskit oder PennyLane laufen. |
| `comp_physics/` | Computational-Physics-Notebooks, Hilfsskripte wie `numerov.py` sowie unterstützende Daten/Abbildungen. |
| `comp_physics_python/` | Python-Portierungen von Jos Thijssens *Computational Physics*, nach Kapiteln organisiert (siehe [comp_physics_python/README.md](comp_physics_python/README.md)). |
| `multiwfn/` | Multiwfn-3.8-Entwicklerquelle plus Handbücher als lokale Referenz. |
| `figures/` | Statische PNG/SVG-Ausgaben für Berichte/Folien und README. |
| `figs/` | Logo- und Bannerdateien. |
| `docs/` | Inhalte der LazyLearn-Microsite (über GitHub Pages oder jeden statischen Host bereitgestellt). |
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

| Requirement | Hinweise |
| --- | --- |
| Python 3.x | Erforderlich für Skripte im Root und die meisten Notebook-Workflows. |
| `pip` (oder Conda) | Paket-/Umgebungsverwaltung |
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

### ✅ Schneller Checklisten-Ablauf

| Schritt | Befehl | Ziel |
| --- | --- | --- |
| 1 | `python -m venv .venv` | Isolierte Umgebung erstellen |
| 2 | `source .venv/bin/activate` (oder OS-äquivalent) | Abhängigkeitskonflikte vermeiden |
| 3 | `pip install --upgrade pip` | Aktuelle Paketwerkzeuge sicherstellen |
| 4 | `pip install qiskit pennylane numpy matplotlib` | Kernepakete für Experimente installieren |
| 5 | Skript aus `examples/` ausführen | Installation end-to-end validieren |

Jupyter-Notebooks in `comp_physics/` verwenden dieselbe Umgebung. Start mit:

```bash
jupyter lab
# or
jupyter notebook
```

### Optionale Kapitel-Portierungs-Abhängigkeiten (`comp_physics_python/`)

```bash
# conda activate quantum  # üblicher lokaler Env-Name in Unterprojekt-Dokumenten
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

Nutzt `default.qubit`.

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
- Themenordner (`bosonscattering/`, `lensless/`, `lightscattering/`, usw.) mit Daten und Hilfsdateien pro Experiment.

Wenn zusätzliche Abhängigkeiten benötigt werden, dokumentiere sie in `comp_physics/environments.yaml`.

## Lehrbuch-Codeübersetzungen 📚

`comp_physics_python/` ist eine wachsende Python-Übersetzung der klassischen Fortran-Programme aus *Computational Physics*. Beispielhafte Kapitelzuordnung:

- `ch4/`: Hartree-Fock-Beispiele.
- `ch8/`: Molekulardynamik-Solver.
- `ch10/`: Monte-Carlo-Sampler.

Siehe [comp_physics_python/README.md](comp_physics_python/README.md) für die vollständige Kapitelabdeckung und CLI-Befehle.

## Multiwfn-Referenzen 🔬

`multiwfn/` enthält `Multiwfn_3.8_dev_src_Linux` sowie das PDF-Handbuch und die Quick-Start-Anleitung. Es werden keine kompilierten Binärdateien versioniert.

## Abbildungen 🖼️

Generierte PNG/SVG-Assets liegen in `figures/`, damit Outputs gemeinsam mit den erzeugenden Skripten/Notebooks versioniert sind.

## Konfiguration 🛠️

### Python und Notebooks

- Root-Skripte setzen das oben gezeigte virtuelle Environment voraus.
- Details zu Notebook-Umgebungen sind über Projektdokumente verteilt; aktuell gibt es im Repo-Root keine einzelne Lockfile.

### Gaussian-Runner (symlinked path)

`Gaussian/run_gaussian.sh` unterstützt:

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

Verhalten:

- Schreibt `<basename>.log` neben der Eingabedatei.
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

### 🎬 Navigationskarte

Nutze dies als Einstieg für die tägliche Arbeit:

| Bereich | Einstiegspunkt |
| --- | --- |
| Quantum Demos | `examples/` |
| Physik-Notebooks | `comp_physics/` |
| Textbook-Übersetzungen | `comp_physics_python/` |
| Quantum-Chemie-Tools | `multiwfn/` |
| Veröffentlichte Outputs | `docs/` |
| Abbildungen und Illustrationen | `figures/`, `figs/` |

### Versionierungs-Hinweise

- Große Pfade werden über `.gitignore` ignoriert, darunter `books/`, externe Symlink-Ziele (`Gaussian`, `ComputationalPhysics`, `leonardsusskind`, `the_theoretical_minimum`) und lokale Artefakte wie `*.chk`.
- Halte Beiträge auf verfolgte Ordner fokussiert, damit Clone-/Update-Workflows schlank bleiben.
- Für Website-Updates: `docs/` bearbeiten, lokal vorschauen, dann pushen.

Lokale Doku-Vorschau:

```bash
python -m http.server --directory docs
```

`docs/CNAME` ist für `learn.lazying.art` konfiguriert.

## Fehlerbehebung 🩺

- Erfolgsnachweis für Gaussian: `Normal termination of Gaussian` nahe dem Ende der `.log`.
- Wenn GaussView unter Wayland/Remote-Sitzungen nicht startet, `gview_safe.sh` verwenden und `--gview` explizit übergeben.
- Wenn Gaussian-Scratch-Fehler auftreten, freien Speicherplatz und Berechtigungen in `GAUSS_SCRDIR` prüfen.
- Wenn Notebook-Abhängigkeiten auseinanderlaufen, Unterprojekt-READMEs als Source of Truth verwenden und fehlende Pakete vor dem Teilen in Umgebungsdateien erfassen.
- `comp_physics/environments.yaml` scheint im aktuellen Repo-Stand ein Platzhalter zu sein; bis zur Korrektur auf explizite Installationsbefehle zurückgreifen.

## Roadmap 🛣️

- Abdeckung der Kapitel in `comp_physics_python/` weiter ausbauen (Transfermatrizen, DMC/PIMC, FEM und mehr).
- Ausgabe-/Plot-Konventionen über Skripte und Notebooks hinweg harmonisieren.
- Leichte, wiederholbare Validierungschecks für zentrale Beispiele ergänzen.
- `docs/` und mehrsprachige READMEs mit neuen Experimenten synchron halten.

## Beitrag 🤝

Issues und Pull Requests sind willkommen, besonders für:

- Prüfungen numerischer Korrektheit und Verbesserungen der Reproduzierbarkeit.
- Bessere Umgebungsspezifikationen für Notebooks/Skripte.
- Zusätzliche Kapitel-Portierungen aus dem Lehrbuch und Verbesserungen der CLI.
- Klarere Dokumentation über die Sprachen in `i18n/` hinweg.

Vor dem Einreichen größerer Inhalts-Updates generierte Abbildungen in `figures/` behalten und sicherstellen, dass Befehle vom Repository-Root aus lauffähig sind, sofern nicht anders dokumentiert.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License 📄

Es gibt aktuell im Root dieses Repositories keine `LICENSE`-Datei. Bis eine Lizenz ergänzt wird, sollten Nutzungs- und Weitergaberechte als nicht festgelegt betrachtet werden; vor der Wiederverwendung wesentlicher Inhalte bitte Rücksprache mit dem Maintainer halten.
