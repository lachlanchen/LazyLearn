[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# LazyPhysics et Chemistry

[![Site](https://img.shields.io/badge/site-learn.lazying.art-0a7ea4)](https://learn.lazying.art)
![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-active%20learning-22c55e)
![Repo Type](https://img.shields.io/badge/repo-mixed--format-6b7280)
![Docs](https://img.shields.io/badge/docs-static%20microsite-0ea5e9)

LazyPhysics and Chemistry est la moitié code + notebooks de **LazyLearn** : un journal d'apprentissage volontairement lent et pratique pour la physique et la chimie. Les notes vivantes, progrès et TODO sont publiés sur [learn.lazying.art](https://learn.lazying.art) (servi depuis `docs/` dans ce dépôt), tandis que les artefacts exécutables restent ici pour que les expériences aient toujours un foyer reproductible.

## Vue d'ensemble 🧭

### LazyLearn

- **Base principale :** [learn.lazying.art](https://learn.lazying.art) - le site public avec les axes hebdomadaires, le backlog et les points forts.
- **Source de vérité :** tout ce que le site référence se trouve dans `examples/`, `comp_physics/`, `comp_physics_python/`, `multiwfn/` ou `figures/`.
- **Flux de mise à jour :** livrer d'abord le code/les notebooks, régénérer les figures si nécessaire, puis ajouter une entrée dans `docs/` pour que le site reflète le travail le plus récent.

Ce dépôt est volontairement en format mixte, pas une application packagée unique. Il combine scripts exécutables, notebooks, références et site statique de documentation dans un même espace versionné.

## Fonctionnalités ✨

- Scripts d'exemple quantiques (QAOA + VQE) exécutables sur des ordinateurs portables standards.
- Notebooks de physique computationnelle et solveurs auxiliaires (par ex. workflows basés sur Numerov).
- Portages Python chapitre par chapitre de programmes de physique computationnelle issus de manuels.
- Bundle source/manuels Multiwfn pour référence locale en post-traitement de chimie quantique.
- Figures générées versionnées pour rapports/slides (`figures/`).
- Ensemble README multilingue intégré dans `i18n/`.
- Microsite statique dans `docs/` (domaine personnalisé : `learn.lazying.art`).

## Structure du projet 🗂️

### Ce qui se trouve ici

| Path | Purpose |
| --- | --- |
| `examples/` | Scripts Python ciblés (QAOA + VQE) exécutables avec Qiskit ou PennyLane. |
| `comp_physics/` | Notebooks de physique computationnelle, scripts utilitaires comme `numerov.py`, et données/figures de support. |
| `comp_physics_python/` | Portages Python de *Computational Physics* de Jos Thijssen, organisés par chapitre (voir [comp_physics_python/README.md](comp_physics_python/README.md)). |
| `multiwfn/` | Bundle source développeur Multiwfn 3.8 avec manuels pour référence locale. |
| `figures/` | Sorties PNG/SVG statiques utilisées dans les rapports/slides et le README. |
| `figs/` | Ressources logo et bannière. |
| `docs/` | Contenu du microsite LazyLearn (servi par GitHub Pages ou tout hébergeur statique). |
| `i18n/` | Fichiers README localisés. |

Arborescence représentative :

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
> Plusieurs entrées de premier niveau sont des symlinks vers des répertoires hors de ce dépôt. Toute modification sous ces chemins affecte des cibles externes.

## Prérequis 🧰

| Requirement | Notes |
| --- | --- |
| Python 3.x | Requis pour les scripts à la racine et la plupart des notebooks. |
| `pip` (ou Conda) | Gestion des paquets/environnements. |
| Jupyter Lab/Notebook (optionnel) | Nécessaire pour les workflows notebooks. |
| Gaussian 16 + GaussView (optionnel) | Nécessaire pour les workflows Gaussian. |

## Installation ⚙️

### Configuration Python minimale (exemples à la racine)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qiskit pennylane numpy matplotlib
```

Les notebooks Jupyter dans `comp_physics/` utilisent le même environnement. Lancez avec :

```bash
jupyter lab
# or
jupyter notebook
```

### Dépendances optionnelles des portages de chapitres (`comp_physics_python/`)

```bash
# conda activate quantum  # common local env name in subproject docs
pip install numpy scipy matplotlib
```

## Utilisation 🚀

### Workflows d'exemple

- **QAOA avec Qiskit**

```bash
python examples/qaoa_qiskit_maxcut.py
```

Aucune dépendance Aer ; utilise un backend statevector pur.

- **QAOA avec PennyLane**

```bash
python examples/qaoa_pennylane_maxcut.py
```

Utilise `default.qubit`.

- **VQE pour H2**

```bash
python examples/pennylane_chemistry_h2_vqe.py
```

Reproduit `figures/pennylane_h2_vqe_convergence.png`.

Tous les scripts journalisent des métriques intermédiaires afin de réutiliser les figures ou d'étendre à de nouvelles molécules/graphes.

## Notebooks de physique computationnelle 📓

Le répertoire `comp_physics/` reflète les notes de travail :

- `comp_physics_textbook_code/` - routines réutilisables extraites des notebooks.
- Notebooks autonomes comme `chapter1.ipynb`, `chapter2.ipynb`, `numerov.ipynb` et `numpy_1ddft.ipynb`.
- Dossiers thématiques (`bosonscattering/`, `lensless/`, `lightscattering/`, etc.) avec données et utilitaires par expérience.

Si des dépendances supplémentaires sont nécessaires, enregistrez-les dans `comp_physics/environments.yaml`.

## Traductions de code de manuel 📚

`comp_physics_python/` est une traduction Python croissante des programmes Fortran classiques de *Computational Physics*. Exemple de correspondance par chapitre :

- `ch4/` : exemples Hartree-Fock.
- `ch8/` : solveurs de dynamique moléculaire.
- `ch10/` : échantillonneurs Monte Carlo.

Consultez [comp_physics_python/README.md](comp_physics_python/README.md) pour la couverture complète des chapitres et les commandes CLI.

## Références Multiwfn 🔬

`multiwfn/` conserve `Multiwfn_3.8_dev_src_Linux` ainsi que le manuel PDF et le guide de démarrage rapide. Aucun binaire compilé n'est versionné.

## Figures 🖼️

Les ressources PNG/SVG générées vivent dans `figures/` afin que les sorties soient versionnées avec les scripts/notebooks qui les produisent.

## Configuration 🛠️

### Python et notebooks

- Les scripts à la racine supposent le venv indiqué ci-dessus.
- Les détails d'environnement notebooks sont répartis dans la documentation du projet ; aucun lockfile unique n'existe actuellement à la racine du dépôt.

### Runner Gaussian (chemin symlinké)

`Gaussian/run_gaussian.sh` prend en charge :

```bash
Gaussian/run_gaussian.sh [--no-view] [--g16 <path_to_g16>] [--gview <path_to_gview.sh>] <input.com|input.gjf>
```

Comportement :

- Écrit `<basename>.log` à côté de l'entrée.
- Utilise `GAUSS_SCRDIR` s'il est défini, sinon utilise `~/gaussian/scr`.
- Détecte `%chk=...` dans l'entrée ; si le checkpoint existe, GaussView ouvre `.chk`, sinon `.log`.
- Si disponible, préfère `~/gaussian/gv/gview_safe.sh`, puis `gview.sh`.

Wrapper GaussView recommandé :

```bash
#!/usr/bin/env bash
set -euo pipefail
GV_SH="$HOME/gaussian/gv/gview.sh"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
exec "$GV_SH" "$@"
```

## Notes de développement 🧪

### Notes de contrôle de version

- Les chemins lourds sont ignorés via `.gitignore`, y compris `books/`, les cibles de symlink externes (`Gaussian`, `ComputationalPhysics`, `leonardsusskind`, `the_theoretical_minimum`) et les artefacts locaux tels que `*.chk`.
- Gardez les contributions concentrées sur les dossiers suivis pour conserver des workflows de clone/mise à jour légers.
- Pour les mises à jour du site web : modifiez `docs/`, prévisualisez localement, puis poussez.

Prévisualisation locale de la doc :

```bash
python -m http.server --directory docs
```

`docs/CNAME` est configuré pour `learn.lazying.art`.

## Dépannage 🩺

- Critère de réussite Gaussian : `Normal termination of Gaussian` près de la fin du `.log`.
- Si GaussView échoue sous Wayland/session distante, utilisez `gview_safe.sh` et passez `--gview` explicitement.
- En cas d'erreur d'espace scratch Gaussian, vérifiez l'espace disque libre et les permissions dans `GAUSS_SCRDIR`.
- Si les dépendances notebooks dérivent, considérez les README des sous-projets comme source de vérité et consignez les paquets manquants dans les fichiers d'environnement avant partage.
- `comp_physics/environments.yaml` semble être un placeholder dans l'état actuel du dépôt ; fiez-vous aux commandes d'installation explicites jusqu'à correction.

## Feuille de route 🛣️

- Continuer à étendre la couverture des chapitres de `comp_physics_python/` (matrices de transfert, DMC/PIMC, FEM, etc.).
- Harmoniser les conventions de sortie/figures entre scripts et notebooks.
- Ajouter des vérifications de validation légères et répétables pour les exemples clés.
- Maintenir `docs/` et les README multilingues alignés sur les nouvelles expériences.

## Contribution 🤝

Les issues et pull requests sont bienvenues, en particulier pour :

- Les vérifications de justesse numérique et les améliorations de reproductibilité.
- De meilleures spécifications d'environnement pour notebooks/scripts.
- Des portages supplémentaires de chapitres de manuels et des raffinements CLI.
- La clarté de la documentation entre langues dans `i18n/`.

Avant de soumettre des mises à jour majeures de contenu, conservez les figures générées dans `figures/` et assurez-vous que les commandes sont exécutables depuis la racine du dépôt, sauf mention contraire documentée.

## Soutenir LazyLearn ❤️

Aider LazyLearn permet de faire avancer les expériences, la documentation et l'outillage ouvert :

- Couvrir l'hébergement/l'inférence/le stockage pour les démos et notebooks publics.
- Financer des hack-weeks ciblées sur EchoMind, LazyEdit et les utilitaires quantique/physique ici.
- Prototyper l'optique + wearables (IdeasGlass, LightMind) qui alimentent les futurs chapitres.
- Sponsoriser des déploiements gratuits pour les étudiants, laboratoires communautaires et créateurs.

### Faire un don

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
- Votre soutien soutient mes activités de recherche, de développement et d'exploitation afin que je puisse continuer à partager davantage de projets ouverts et d'améliorations.

## Licence 📄

Aucun fichier `LICENSE` à la racine n'est actuellement présent dans ce dépôt. Tant qu'une licence n'est pas ajoutée, considérez les droits d'usage/redistribution comme non spécifiés et demandez une clarification au mainteneur avant de réutiliser un contenu substantiel.
